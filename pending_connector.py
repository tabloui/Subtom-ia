from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import aiohttp
from aiohttp import web

import discord

from ai import agent
from config import config
from connector import connector
from motor import motor


# ============================================================
# UPTIME
# ============================================================

_START_TIME = time.monotonic()


def _uptime_seconds() -> float:
    return time.monotonic() - _START_TIME


def _uptime_human() -> str:
    secs = int(_uptime_seconds())
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if mins or hours or days:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)


# ============================================================
# HTTP HEALTH SERVER
# ============================================================

async def health(_: web.Request) -> web.Response:
    return web.json_response({
        "ok": True,
        "service": config.bot_name,
        "provider": connector.provider,
        "model": config.ai_model,
        "uptime_seconds": round(_uptime_seconds(), 1),
        "uptime_human": _uptime_human(),
    })


async def root(_: web.Request) -> web.Response:
    return web.Response(
        text=f"{config.bot_name} online ({connector.provider})"
    )


async def metrics(_: web.Request) -> web.Response:
    try:
        data = motor.snapshot()
        data["rotador"] = connector.stats()
        data["uptime_seconds"] = round(_uptime_seconds(), 1)
        data["uptime_human"] = _uptime_human()
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

    print(f"HTTP server escuchando en puerto {config.port}")
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


async def should_reply(message: discord.Message) -> bool:
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
        "!image ", "/image ", "image: ",
        "!imagen ", "/imagen ", "imagen: ",
        "!draw ", "/draw ",
        "genera: ", "dibuja: ",
    )

    for prefix in prefixes:
        if lowered.startswith(prefix):
            return (content[len(prefix):].strip(), True)

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

                    items.append({
                        "path": str(target),
                        "filename": att.filename,
                        "content_type": att.content_type or "",
                        "url": att.url,
                        "size": len(data),
                        "is_image": (
                            (att.content_type or "").startswith("image/")
                        ),
                    })
                    print(f"Adjunto guardado: {target}")

            except Exception as exc:
                print(f"Error descargando {att.filename}: {exc}")

    return items


# ============================================================
# CONTEXTO
# ============================================================

def build_user_context(
    message: discord.Message,
    attachments: list[dict],
) -> str:

    user = message.author

    lines = [
        "[Contexto Discord]",
        f"- Usuario: {user.display_name}",
        f"- Nombre: {user.name}",
        f"- ID: {user.id}",
        f"- Avatar: {user.display_avatar.url}",
        f"- Canal ID actual: {message.channel.id}",
    ]

    if message.guild:
        lines.append(f"- Servidor: {message.guild.name}")
        lines.append(f"- Servidor ID: {message.guild.id}")
        lines.append(f"- Miembros: {message.guild.member_count}")

        if message.guild.icon:
            lines.append(f"- Icono servidor: {message.guild.icon.url}")

    if isinstance(message.channel, discord.TextChannel):
        lines.append(f"- Canal: #{message.channel.name}")

        if message.channel.topic:
            lines.append(f"- Tema del canal: {message.channel.topic}")

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
                "- El usuario adjuntó una imagen. Analízala directamente."
            )

        if has_pdf:
            lines.append(
                "- El usuario adjuntó un PDF. Si necesitas leerlo, "
                "usa file_read_pdf con la ruta local indicada."
            )

        if has_text:
            lines.append(
                "- El usuario adjuntó un archivo de texto. Si necesitas "
                "leerlo, usa file_read_text o file_read_any con la "
                "ruta local."
            )

    return "\n".join(lines)


# ============================================================
# ENVÍO DE ARCHIVOS
# ============================================================

def _resolve_workspace_path(raw: str | Path) -> Path | None:
    """
    Resuelve una ruta de archivo intentando varias ubicaciones:
    1) Ruta absoluta tal cual
    2) Ruta relativa al CWD
    3) Ruta relativa al workspace
    4) Solo el nombre del archivo dentro del workspace
    5) Dentro de workspace/generated (imágenes)
    6) Dentro de workspace/uploads (adjuntos)
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None

    p = Path(raw)

    # 1) Absoluta
    if p.is_absolute():
        if p.exists() and p.is_file():
            return p
        return None

    # 2) Relativa al CWD
    if p.exists() and p.is_file():
        return p.resolve()

    # 3) Relativa al workspace
    workspace_root = Path(config.workspace)
    candidate = workspace_root / raw
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # 4) Solo el nombre dentro del workspace
    candidate = workspace_root / p.name
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # 5) Dentro de workspace/generated
    candidate = workspace_root / "generated" / p.name
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # 6) Dentro de workspace/uploads
    candidate = workspace_root / "uploads" / p.name
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    return None


async def send_files(
    chat: discord.Messageable,
    files: list,
) -> None:

    valid: list[discord.File] = []

    for item in files:
        try:
            if isinstance(item, (str, Path)):
                resolved = _resolve_workspace_path(item)
                if resolved is not None:
                    valid.append(discord.File(resolved))
                    print(f"[SEND_FILES] OK: {item} -> {resolved}")
                else:
                    print(f"[SEND_FILES] NO ENCONTRADO: {item}")
                    try:
                        await chat.send(f"⚠️ No pude encontrar el archivo: `{item}`")
                    except Exception:
                        pass

            elif isinstance(item, tuple) and len(item) == 2:
                name, data = item
                if isinstance(data, bytes):
                    valid.append(
                        discord.File(io.BytesIO(data), filename=name)
                    )

        except Exception as exc:
            print(f"Error preparando archivo: {exc}")

    for i in range(0, len(valid), 10):
        batch = valid[i:i + 10]
        try:
            await chat.send(files=batch)
            print(f"[SEND_FILES] Enviados {len(batch)} archivos")
        except Exception as exc:
            print(f"Error enviando lote: {exc}")


# ============================================================
# IMAGEN GENERADA
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

                await chat.send(f"Imagen generada: {image_url}")

    except Exception as exc:
        print(f"Error descargando imagen generada: {exc}")
        try:
            await chat.send(f"Imagen generada: {image_url}")
        except Exception:
            pass


# ============================================================
# DISCORD EVENTS
# ============================================================

@client.event
async def on_ready() -> None:
    print(
        f"{config.bot_name} | "
        f"{len(config.ai_slots)} slots | "
        f"principal: {connector.provider} | "
        f"modelo: {config.ai_model}"
    )


@client.event
async def on_message(message: discord.Message) -> None:

    if not await should_reply(message):
        return

    prompt, flag_image = await extract_prompt(message)
    attachments = await download_attachments(message)

    if not prompt and not attachments:
        await message.channel.save("No entendí tu mensaje.")
        return

    user_context = build_user_context(message, attachments)
    full_prompt = f"{user_context}\n\n{prompt}".strip()

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
            await message.remove_reaction("⏳", client.user)
        except Exception:
            pass

    files = None

    if isinstance(result, tuple) and len(result) == 3:
        response, image_url, files = result
    else:
        response, image_url = result

    # Enviar la respuesta de texto
    if response:
        await send_long(message.channel, response)

    # Enviar archivos del workspace (imágenes generadas, pendings, etc.)
    if files:
        await send_files(message.channel, files)

    # Enviar imagen por URL (fallback si no vino por _send_file)
    if image_url and not files:
        await send_generated_image(message.channel, image_url)


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    runner = None

    try:
        await agent.init()
        runner = await start_http()
        await client.start(config.discord_token)

    finally:
        print("Cerrando Subtom...")
        await connector.close()
        await agent.close()
        await client.close()
        if runner is not None:
            await runner.cleanup()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
