from __future__ import annotations

import asyncio
import io
from pathlib import Path

import aiohttp
from aiohttp import web

import discord

from ai import agent
from config import config
from connector import connector
from motor import motor


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

client = discord.Client(
    intents=intents
)


# ============================================================
# HTTP HEALTH SERVER
# ============================================================

async def health(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "service": config.bot_name,
            "provider": connector.provider,
            "model": config.ai_model,
        }
    )


async def root(_: web.Request) -> web.Response:
    return web.Response(
        text=f"{config.bot_name} online ({connector.provider})"
    )


async def metrics(_: web.Request) -> web.Response:
    try:
        data = motor.snapshot()
    except Exception as exc:
        return web.json_response(
            {"error": f"{type(exc).__name__}: {exc}"},
            status=500,
        )
    return web.json_response(data)


async def start_http() -> web.AppRunner:

    app = web.Application()

    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", health)
    app.router.add_get("/metrics", metrics)

    runner = web.AppRunner(app)

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        config.port,
    )

    await site.start()

    print(
        f"HTTP server escuchando en puerto {config.port}"
    )

    return runner


# ============================================================
# HELPERS
# ============================================================

async def send_long(
    chat: discord.Messageable,
    text: str,
) -> None:

    text = text or "Sin respuesta."

    for i in range(0, len(text), 1900):
        await chat.send(text[i:i + 1900])


async def should_reply(
    message: discord.Message,
) -> bool:

    if message.author.bot:
        return False

    if message.author.id != config.allowed_user_id:
        return False

    if isinstance(message.channel, discord.DMChannel):
        return True

    return (
        client.user is not None
        and client.user in message.mentions
    )


async def extract_prompt(
    message: discord.Message,
) -> tuple[str, bool]:

    content = message.content.strip()

    flag_image = False

    lowered = content.lower()

    prefixes = (
        "!image ",
        "/image ",
        "image: ",
        "!imagen ",
        "/imagen ",
        "imagen: ",
        "!draw ",
        "/draw ",
        "genera: ",
        "dibuja: ",
    )

    for prefix in prefixes:

        if lowered.startswith(prefix):
            return (
                content[len(prefix):].strip(),
                True,
            )

    if message.attachments:
        flag_image = True

    return (content.strip(), flag_image)


# ============================================================
# ADJUNTOS
# ============================================================

async def download_attachments(
    message: discord.Message,
) -> list[dict]:

    if not message.attachments:
        return []

    uploads = Path(config.workspace) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []

    async with aiohttp.ClientSession() as session:

        for att in message.attachments:

            safe_name = f"{message.id}_{att.filename}"
            target = uploads / safe_name

            try:

                async with session.get(att.url) as resp:

                    if resp.status >= 400:
                        continue

                    data = await resp.read()
                    target.write_bytes(data)

                    items.append(
                        {
                            "path": str(target),
                            "filename": att.filename,
                            "content_type": att.content_type or "",
                            "url": att.url,
                            "size": len(data),
                            "is_image": (
                                (att.content_type or "")
                                .startswith("image/")
                            ),
                        }
                    )

                    print(f"Adjunto guardado: {target}")

            except Exception as exc:

                print(
                    f"Error descargando "
                    f"{att.filename}: {exc}"
                )

    return items


# ============================================================
# CONTEXTO DE USUARIO / SERVIDOR
# ============================================================

def build_user_context(
    message: discord.Message,
    attachments: list[dict],
) -> str:

    user = message.author

    lines = [
        "[Contexto Discord]",
        f"- Usuario: {user.display_name}",
        f"- Nombre de usuario: {user.name}",
        f"- ID: {user.id}",
        f"- Avatar: {user.display_avatar.url}",
    ]

    if message.guild:

        lines.append(f"- Servidor: {message.guild.name}")
        lines.append(f"- Servidor ID: {message.guild.id}")
        lines.append(
            f"- Miembros: {message.guild.member_count}"
        )

        if message.guild.icon:
            lines.append(
                f"- Icono servidor: {message.guild.icon.url}"
            )

    if isinstance(message.channel, discord.TextChannel):

        lines.append(f"- Canal: #{message.channel.name}")
        lines.append(f"- Canal ID: {message.channel.id}")

        if message.channel.topic:
            lines.append(
                f"- Tema del canal: {message.channel.topic}"
            )

    elif isinstance(message.channel, discord.DMChannel):
        lines.append("- Canal: DM")

    if attachments:

        lines.append("- Archivos adjuntos del usuario:")

        for att in attachments:

            kind = "imagen" if att["is_image"] else "archivo"

            lines.append(
                f"  * [{kind}] {att['filename']} "
                f"({att['content_type']}) "
                f"-> ruta local: {att['path']}"
            )

        has_image = any(a["is_image"] for a in attachments)
        has_pdf = any(
            (a["content_type"] or "") == "application/pdf"
            for a in attachments
        )
        has_text = any(
            (a["content_type"] or "").startswith("text/")
            for a in attachments
        )

        lines.append("")
        lines.append("[Instrucciones]")

        if has_image:
            lines.append(
                "- El usuario adjuntó una imagen. "
                "Analízala directamente."
            )

        if has_pdf:
            lines.append(
                "- El usuario adjuntó un PDF. "
                "Si necesitas leerlo, usa la herramienta "
                "file_read_pdf con la ruta local indicada."
            )

        if has_text:
            lines.append(
                "- El usuario adjuntó un archivo de texto. "
                "Si necesitas leerlo, usa file_read_text "
                "o file_read_any con la ruta local."
            )

    return "\n".join(lines)


# ============================================================
# ENVÍO DE ARCHIVOS
# ============================================================

async def send_files(
    chat: discord.Messageable,
    files: list,
) -> None:

    for item in files:

        try:

            if isinstance(item, (str, Path)):

                path = Path(item)

                if path.exists():
                    await chat.send(file=discord.File(path))

            elif (
                isinstance(item, tuple)
                and len(item) == 2
            ):

                name, data = item

                if isinstance(data, bytes):

                    await chat.send(
                        file=discord.File(
                            io.BytesIO(data),
                            filename=name,
                        )
                    )

        except Exception as exc:

            print(f"Error enviando archivo: {exc}")


# ============================================================
# IMAGEN GENERADA (descarga y reenvía)
# ============================================================

async def send_generated_image(
    chat: discord.Messageable,
    image_url: str,
) -> None:

    try:

        async with aiohttp.ClientSession() as session:

            async with session.get(image_url) as resp:

                if resp.status < 400:

                    data = await resp.read()

                    await chat.send(
                        file=discord.File(
                            io.BytesIO(data),
                            filename="subtom.png",
                        )
                    )

                    return

                await chat.send(
                    f"Imagen generada: {image_url}"
                )

    except Exception as exc:

        print(f"Error descargando imagen generada: {exc}")

        try:
            await chat.send(
                f"Imagen generada: {image_url}"
            )
        except Exception:
            pass


# ============================================================
# DISCORD EVENTS
# ============================================================

@client.event
async def on_ready() -> None:

    print(
        f"{config.bot_name} | "
        f"proveedor={connector.provider} | "
        f"modelo={config.ai_model}"
    )


@client.event
async def on_message(
    message: discord.Message,
) -> None:

    if not await should_reply(message):
        return

    prompt, flag_image = await extract_prompt(message)

    attachments = await download_attachments(message)

    if not prompt and not attachments:

        await message.channel.send(
            "No entendí tu mensaje."
        )

        return

    user_context = build_user_context(
        message,
        attachments,
    )

    full_prompt = (
        f"{user_context}\n\n{prompt}"
    ).strip()

    try:
        await message.add_reaction("⏳")
    except Exception:
        pass

    try:

        result = await agent.ask(
            message.author.id,
            message.channel.id,
            full_prompt,
            force_image=flag_image,
        )

    except Exception as exc:

        print(f"Error procesando mensaje: {exc}")

        await message.channel.send(f"Error: {exc}")

        return

    finally:

        try:
            await message.remove_reaction(
                "⏳",
                client.user,
            )
        except Exception:
            pass

    files = None

    if isinstance(result, tuple) and len(result) == 3:
        response, image_url, files = result
    else:
        response, image_url = result

    await send_long(message.channel, response)

    if image_url:
        await send_generated_image(
            message.channel,
            image_url,
        )

    if files:
        await send_files(message.channel, files)


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    runner = None

    try:

        await agent.init()

        motor.start_healthcheck(interval=300)

        runner = await start_http()

        await client.start(config.discord_token)

    finally:

        print("Cerrando Subtom...")

        motor.stop_healthcheck()

        await agent.close()

        await client.close()

        if runner is not None:
            await runner.cleanup()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
