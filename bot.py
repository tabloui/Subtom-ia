from __future__ import annotations

import asyncio
import io
from pathlib import Path

import aiohttp
from aiohttp import web

import discord

from ai import agent
from config import config


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

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
            "model": config.ai_model,
        }
    )


async def root(_: web.Request) -> web.Response:
    return web.Response(
        text=f"{config.bot_name} online"
    )


async def start_http() -> web.AppRunner:

    app = web.Application()

    app.router.add_get(
        "/",
        root,
    )

    app.router.add_get(
        "/health",
        health,
    )

    app.router.add_get(
        "/api/health",
        health,
    )

    app.router.add_get(
        "/api/status",
        health,
    )

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
# DISCORD MESSAGE HELPERS
# ============================================================

async def send_long(
    chat: discord.Messageable,
    text: str,
) -> None:

    text = text or "Sin respuesta."

    for i in range(
        0,
        len(text),
        1900,
    ):
        await chat.send(
            text[i:i + 1900]
        )


async def should_reply(
    message: discord.Message,
) -> bool:

    # Ignorar bots
    if message.author.bot:
        return False

    # Solo nuestro usuario autorizado
    if message.author.id != config.allowed_user_id:
        return False

    # Mensajes privados
    if isinstance(
        message.channel,
        discord.DMChannel,
    ):
        return True

    # Mensajes en servidor:
    # responder solamente si mencionan al bot.
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
    )

    for prefix in prefixes:

        if lowered.startswith(prefix):

            return (
                content[len(prefix):].strip(),
                True,
            )

    if message.attachments:
        flag_image = True

    return (
        content.strip(),
        flag_image,
    )


# ============================================================
# ADJUNTOS / CONTEXTO DE USUARIO
# ============================================================

async def download_attachments(
    message: discord.Message,
) -> list[str]:
    """
    Descarga los archivos adjuntos al workspace
    y devuelve sus rutas locales.
    """

    if not message.attachments:
        return []

    uploads = Path(
        config.workspace
    ) / "uploads"

    uploads.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: list[str] = []

    async with aiohttp.ClientSession() as session:

        for att in message.attachments:

            safe_name = f"{message.id}_{att.filename}"

            target = uploads / safe_name

            try:

                async with session.get(
                    att.url
                ) as resp:

                    if resp.status >= 400:
                        continue

                    data = await resp.read()

                    target.write_bytes(data)

                    paths.append(str(target))

                    print(
                        f"Adjunto guardado: {target}"
                    )

            except Exception as exc:

                print(
                    f"Error descargando "
                    f"{att.filename}: {exc}"
                )

    return paths


def build_user_context(
    message: discord.Message,
    attachment_paths: list[str],
) -> str:

    user = message.author

    lines = [
        "[Contexto Discord]",
        f"- Usuario: {user.display_name}",
        f"- Nombre: {user.name}",
        f"- ID: {user.id}",
        f"- Avatar: {user.display_avatar.url}",
    ]

    if message.guild:

        lines.append(
            f"- Servidor: {message.guild.name}"
        )

    if isinstance(
        message.channel,
        discord.TextChannel,
    ):

        lines.append(
            f"- Canal: #{message.channel.name}"
        )

    elif isinstance(
        message.channel,
        discord.DMChannel,
    ):

        lines.append("- Canal: DM")

    if attachment_paths:

        lines.append("- Archivos adjuntos guardados:")

        for path in attachment_paths:

            lines.append(f"  * {path}")

    return "\n".join(lines)


async def send_files(
    chat: discord.Messageable,
    files: list,
) -> None:
    """
    Envía archivos al chat.

    Acepta rutas (str / Path) o tuplas (nombre, bytes).
    """

    for item in files:

        try:

            if isinstance(item, (str, Path)):

                path = Path(item)

                if path.exists():

                    await chat.send(
                        file=discord.File(path)
                    )

            elif isinstance(item, tuple) and len(item) == 2:

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
# DISCORD EVENTS
# ============================================================

@client.event
async def on_ready() -> None:

    print(
        f"{config.bot_name} | "
        f"modelo={config.ai_model}"
    )


@client.event
async def on_message(
    message: discord.Message,
) -> None:

    if not await should_reply(message):
        return

    prompt, flag_image = await extract_prompt(
        message
    )

    # --------------------------------------------------------
    # DESCARGAR ADJUNTOS
    # --------------------------------------------------------

    attachment_paths = await download_attachments(
        message
    )

    if not prompt and not attachment_paths:

        await message.channel.send(
            "No entendí tu mensaje."
        )

        return

    # --------------------------------------------------------
    # CONTEXTO + PROMPT
    # --------------------------------------------------------

    user_context = build_user_context(
        message,
        attachment_paths,
    )

    full_prompt = (
        f"{user_context}\n\n{prompt}"
    ).strip()

    # --------------------------------------------------------
    # REACCIÓN DE "PROCESANDO"
    # --------------------------------------------------------

    try:
        await message.add_reaction("⏳")

    except Exception:
        pass

    # --------------------------------------------------------
    # LLAMAR AL AGENTE
    # --------------------------------------------------------

    try:

        result = await agent.ask(
            message.author.id,
            message.channel.id,
            full_prompt,
            force_image=flag_image,
        )

    except Exception as exc:

        print(
            f"Error procesando mensaje: {exc}"
        )

        await message.channel.send(
            f"Error: {exc}"
        )

        return

    finally:

        try:
            await message.remove_reaction(
                "⏳",
                client.user,
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # SOPORTE 2-TUPLE Y 3-TUPLE
    # --------------------------------------------------------

    files = None

    if isinstance(result, tuple) and len(result) == 3:

        response, image_url, files = result

    else:

        response, image_url = result

    # --------------------------------------------------------
    # ENVIAR RESPUESTA
    # --------------------------------------------------------

    await send_long(
        message.channel,
        response,
    )

    # --------------------------------------------------------
    # ENVIAR IMAGEN GENERADA
    # --------------------------------------------------------

    if image_url:

        try:

            async with aiohttp.ClientSession() as session:

                async with session.get(
                    image_url
                ) as resp:

                    if resp.status < 400:

                        data = await resp.read()

                        await message.channel.send(
                            file=discord.File(
                                io.BytesIO(data),
                                filename="subtom.png",
                            )
                        )

                    else:

                        await message.channel.send(
                            f"Imagen generada: {image_url}"
                        )

        except Exception:

            await message.channel.send(
                f"Imagen generada: {image_url}"
            )

    # --------------------------------------------------------
    # ENVIAR ARCHIVOS DEL BOT
    # --------------------------------------------------------

    if files:

        await send_files(
            message.channel,
            files,
        )


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    runner = None

    try:

        # Inicializar PostgreSQL/memoria
        await agent.init()

        # Servidor HTTP para Railway
        runner = await start_http()

        # Conectar Discord
        await client.start(
            config.discord_token
        )

    finally:

        print("Cerrando Subtom...")

        await agent.close()

        await client.close()

        if runner is not None:
            await runner.cleanup()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
