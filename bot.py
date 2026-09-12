from __future__ import annotations

import asyncio
import io
from aiohttp import web, ClientSession
import discord
from discord.ext import tasks

from ai import agent
from config import config
from connector import connector


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def health(_: web.Request) -> web.Response:
    return web.json_response({
        "ok": True,
        "service": config.bot_name,
        "model": config.ai_model,
    })


async def root(_: web.Request) -> web.Response:
    return web.Response(text=f"{config.bot_name} online")


async def start_http() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_get("/api/healthz", health)
    app.router.add_get("/api/status", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", config.port)
    await site.start()
    return runner


async def send_long(channel: discord.abc.Messageable, text: str) -> None:
    text = text or "Sin respuesta."
    for i in range(0, len(text), 1900):
        await channel.send(text[i:i + 1900])


def should_answer(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    if message.author.id != config.allowed_user_id:
        return False
    if isinstance(message.channel, discord.DMChannel):
        return True
    return client.user is not None and client.user in message.mentions


def extract_prompt(message: discord.Message) -> tuple[str, bool]:
    content = message.content.strip()
    force_image = False

    lowered = content.lower()
    prefixes = ("!imagen ", "/imagen ", "imagen: ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return content[len(prefix):].strip(), True

    if client.user:
        content = content.replace(f"<@{client.user.id}>", "")
        content = content.replace(f"<@!{client.user.id}>", "")
    return content.strip(), force_image


@client.event
async def on_ready() -> None:
    print(f"Conectado como {client.user} | modelo={config.ai_model}")


@client.event
async def on_message(message: discord.Message) -> None:
    if not should_answer(message):
        return

    prompt, force_image = extract_prompt(message)
    if not prompt:
        await message.channel.send("Dime qué necesitas.")
        return

    async with message.channel.typing():
        try:
            answer, image_url = await agent.ask(
                message.author.id,
                message.channel.id,
                prompt,
                force_image=force_image,
            )
        except Exception as exc:
            print(f"AI error: {exc!r}")
            await message.channel.send(connector.public_error(exc))
            return

    await send_long(message.channel, answer)

    if image_url:
        try:
            async with ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status < 400:
                        data = await response.read()
                        await message.channel.send(
                            file=discord.File(io.BytesIO(data), filename="subtom.png")
                        )
                    else:
                        await message.channel.send(f"Imagen generada: {image_url}")
        except Exception:
            await message.channel.send(f"Imagen generada: {image_url}")


@tasks.loop(minutes=10)
async def memory_cleanup() -> None:
    try:
        await agent.cleanup_memory()
    except Exception as exc:
        print(f"Memory cleanup error: {exc!r}")


async def main() -> None:
    await agent.init()
    runner = await start_http()
    memory_cleanup.start()
    try:
        await client.start(config.discord_token)
    finally:
        memory_cleanup.cancel()
        await agent.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
