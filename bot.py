from __future__ import annotations
import asyncio
import io
import json
from typing import Optional

import aiohttp
import discord
from discord.ext import commands

from ai import agent
from config import config
from connector import connector


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def health(_: aiohttp.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({
        "ok": True,
        "service": config.bot_name,
        "model": config.ai_model,
    })


async def root(_: aiohttp.Request) -> aiohttp.web.Response:
    return aiohttp.web.Response(text=f"{config.bot_name} online")


async def start_http() -> aiohttp.web.Application:
    app = aiohttp.web.Application()
    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", health)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", config.port)
    await site.start()
    return runner


async def send_long(chat: discord.Messageable, text: str) -> None:
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
    return client.user is not None and message.mentions(client.user)


async def extract_prompt(message: discord.Message) -> tuple[str, bool]:
    content = message.content.strip()
    flag_image = False
    lowered = content.lower()
    prefixes = ("!image ", "/image ", "image: ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return content[len(prefix):].strip(), True
    if message.attachments:
        flag_image = True
    return content.strip(), flag_image


@client.event
async def on_ready() -> None:
    print(f"{config.bot_name} | modelo={config.ai_model}")


@client.event
async def on_message(message: discord.Message) -> None:
    if not await should_reply(message):
        return

    prompt, flag_image = await extract_prompt(message)
    if not prompt:
        await message.channel.send("No entendí tu mensaje.")
        return

    try:
        response, image_url = await agent.ask(
            message.author.id,
            message.channel.id,
            prompt,
            flag_image=flag_image,
        )
    except Exception as exc:
        await message.channel.send(f"Error: {exc}")
        return

    await send_long(message.channel, response)

    if image_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status < 400:
                        data = await resp.read()
                        await message.channel.send(
                            file=discord.File(io.BytesIO(data), filename="subtom.png")
                        )
                    else:
                        await message.channel.send(f"Imagen generada: {image_url}")
        except Exception:
            await message.channel.send(f"Imagen generada: {image_url}")


tasks.max_concurrency = 10
async def memory_cleaner() -> None:
    try:
        await agent.cleanup_memory()
    except Exception as exc:
        print(f"Error en limpieza de memoria: {exc}")


async def main() -> None:
    await agent.init()
    runner = await start_http()
    try:
        await client.start(config.discord_token)
    finally:
        memory_cleaner.cancel()
        await client.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
