from __future__ import annotations

import asyncio
import io

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

    if not prompt:
        await message.channel.send(
            "No entendí tu mensaje."
        )
        return

    try:

        response, image_url = await agent.ask(
            message.author.id,
            message.channel.id,
            prompt,
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
