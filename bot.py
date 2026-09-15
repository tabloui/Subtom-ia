from __future__ import annotations

import asyncio
import io
from datetime import timedelta
from pathlib import Path
from typing import Any

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
intents.members = True

client = discord.Client(intents=intents)


# ============================================================
# DISCORD TOOLS (integradas)
# ============================================================

async def _resolve_guild(guild_id: int | None) -> discord.Guild | None:
    if guild_id is None:
        if not client.guilds:
            return None
        return client.guilds[0]
    return client.get_guild(int(guild_id))


async def _resolve_channel(channel_id: int):
    ch = client.get_channel(int(channel_id))
    if ch is None:
        try:
            ch = await client.fetch_channel(int(channel_id))
        except Exception:
            return None
    return ch


async def _resolve_user(user_id: int):
    u = client.get_user(int(user_id))
    if u is None:
        try:
            u = await client.fetch_user(int(user_id))
        except Exception:
            return None
    return u


async def dtool_send_message(channel_id, content="", embed=None, reply_to=None):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        kwargs = {"content": content or None}
        if embed:
            e = discord.Embed(
                title=embed.get("title"),
                description=embed.get("description"),
                color=discord.Color(embed.get("color", 0x5865F2)),
            )
            for f in embed.get("fields", []):
                e.add_field(name=f.get("name", "—"), value=f.get("value", "—"), inline=f.get("inline", False))
            if embed.get("footer"):
                e.set_footer(text=embed["footer"])
            if embed.get("image"):
                e.set_image(url=embed["image"])
            if embed.get("thumbnail"):
                e.set_thumbnail(url=embed["thumbnail"])
            kwargs["embed"] = e
        if reply_to:
            try:
                ref = await ch.fetch_message(int(reply_to))
                kwargs["reference"] = ref
            except Exception:
                pass
        msg = await ch.send(**kwargs)
        return {"message_id": msg.id, "channel_id": msg.channel.id, "jump_url": msg.jump_url}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_create_poll(channel_id, question, options, duration_hours=24):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        poll = discord.Poll(question=question, duration=timedelta(hours=max(1, min(int(duration_hours), 168))))
        for opt in options[:10]:
            poll.add_answer(text=str(opt))
        msg = await ch.send(poll=poll)
        return {"message_id": msg.id, "jump_url": msg.jump_url}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_send_dm(user_id, content):
    try:
        user = await _resolve_user(user_id)
        if user is None:
            return {"error": f"Usuario {user_id} no encontrado"}
        msg = await user.send(content)
        return {"ok": True, "message_id": msg.id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_purge(channel_id, amount=10, user_id=None):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        def check(m):
            if user_id is None:
                return True
            return m.author.id == int(user_id)
        deleted = await ch.purge(limit=min(int(amount), 100), check=check)
        return {"deleted": len(deleted)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_ban(user_id, reason="", delete_days=0, guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        member = guild.get_member(int(user_id))
        if member is None:
            return {"error": f"Usuario {user_id} no está en el servidor"}
        await guild.ban(member, reason=reason or None, delete_message_days=max(0, min(int(delete_days), 7)))
        return {"ok": True, "banned": user_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_kick(user_id, reason="", guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        member = guild.get_member(int(user_id))
        if member is None:
            return {"error": f"Usuario {user_id} no está en el servidor"}
        await guild.kick(member, reason=reason or None)
        return {"ok": True, "kicked": user_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_timeout(user_id, minutes, reason="", guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        member = guild.get_member(int(user_id))
        if member is None:
            return {"error": f"Usuario {user_id} no está en el servidor"}
        until = discord.utils.utcnow() + timedelta(minutes=max(1, min(int(minutes), 40320)))
        await member.timeout(until, reason=reason or None)
        return {"ok": True, "timed_out": user_id, "until": until.isoformat()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_list_roles(guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        return {"roles": [{"id": r.id, "name": r.name, "color": str(r.color)} for r in guild.roles if not r.is_default()]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_add_role(user_id, role_id, guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        member = guild.get_member(int(user_id))
        role = guild.get_role(int(role_id))
        if member is None:
            return {"error": f"Usuario {user_id} no está"}
        if role is None:
            return {"error": f"Rol {role_id} no encontrado"}
        await member.add_roles(role)
        return {"ok": True, "role": role.name}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_remove_role(user_id, role_id, guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        member = guild.get_member(int(user_id))
        role = guild.get_role(int(role_id))
        if member is None:
            return {"error": f"Usuario {user_id} no está"}
        if role is None:
            return {"error": f"Rol {role_id} no encontrado"}
        await member.remove_roles(role)
        return {"ok": True, "role": role.name}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_list_channels(guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        return {"channels": [{"id": c.id, "name": c.name, "type": str(c.type)} for c in guild.channels]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_create_channel(name, channel_type="text", category_id=None, guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        cmap = {
            "text": discord.ChannelType.text,
            "voice": discord.ChannelType.voice,
            "category": discord.ChannelType.category,
            "forum": discord.ChannelType.forum,
        }
        ctype = cmap.get(channel_type.lower(), discord.ChannelType.text)
        category = guild.get_channel(int(category_id)) if category_id else None
        ch = await guild.create_channel(name=name, type=ctype, category=category)
        return {"ok": True, "id": ch.id, "name": ch.name}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_delete_channel(channel_id, reason=""):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        await ch.delete(reason=reason or None)
        return {"ok": True, "deleted": channel_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_edit_channel(channel_id, name=None, topic=None, slowmode=None):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        kwargs = {}
        if name is not None:
            kwargs["name"] = name
        if topic is not None:
            kwargs["topic"] = topic
        if slowmode is not None:
            kwargs["slowmode_delay"] = max(0, min(int(slowmode), 21600))
        await ch.edit(**kwargs)
        return {"ok": True, "channel_id": channel_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_create_invite(channel_id, max_age=86400, max_uses=0):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        invite = await ch.create_invite(max_age=max(0, min(int(max_age), 604800)), max_uses=max(0, min(int(max_uses), 100)))
        return {"ok": True, "url": invite.url, "code": invite.code}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_pin_message(channel_id, message_id):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        msg = await ch.fetch_message(int(message_id))
        await msg.pin()
        return {"ok": True, "pinned": message_id}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_list_pins(channel_id):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        pins = await ch.pins()
        return {"pins": [{"id": m.id, "content": m.content[:200], "author": str(m.author)} for m in pins]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_get_guild_info(guild_id=None):
    try:
        guild = await _resolve_guild(guild_id)
        if guild is None:
            return {"error": "Guild no encontrado"}
        return {
            "name": guild.name, "id": guild.id, "members": guild.member_count,
            "owner_id": guild.owner_id, "created_at": guild.created_at.isoformat(),
            "icon": guild.icon.url if guild.icon else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_get_user_info(user_id, guild_id=None):
    try:
        user = await _resolve_user(user_id)
        if user is None:
            return {"error": f"Usuario {user_id} no encontrado"}
        result = {"name": str(user), "id": user.id, "bot": user.bot,
                  "created_at": user.created_at.isoformat(), "avatar": user.display_avatar.url}
        guild = await _resolve_guild(guild_id)
        if guild:
            member = guild.get_member(int(user_id))
            if member:
                result["joined_at"] = member.joined_at.isoformat() if member.joined_at else None
                result["roles"] = [r.name for r in member.roles if not r.is_default()]
                result["nick"] = member.nick
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def dtool_add_reaction(channel_id, message_id, emoji):
    try:
        ch = await _resolve_channel(channel_id)
        if ch is None:
            return {"error": f"Canal {channel_id} no encontrado"}
        msg = await ch.fetch_message(int(message_id))
        await msg.add_reaction(emoji)
        return {"ok": True, "emoji": emoji}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ============================================================
# REGISTRO DE HERRAMIENTAS DISCORD
# (el agente las buscará aquí por nombre)
# ============================================================

DISCORD_TOOLS = {
    "discord_send_message": dtool_send_message,
    "discord_create_poll": dtool_create_poll,
    "discord_send_dm": dtool_send_dm,
    "discord_purge": dtool_purge,
    "discord_ban": dtool_ban,
    "discord_kick": dtool_kick,
    "discord_timeout": dtool_timeout,
    "discord_list_roles": dtool_list_roles,
    "discord_add_role": dtool_add_role,
    "discord_remove_role": dtool_remove_role,
    "discord_list_channels": dtool_list_channels,
    "discord_create_channel": dtool_create_channel,
    "discord_delete_channel": dtool_delete_channel,
    "discord_edit_channel": dtool_edit_channel,
    "discord_create_invite": dtool_create_invite,
    "discord_pin_message": dtool_pin_message,
    "discord_list_pins": dtool_list_pins,
    "discord_get_guild_info": dtool_get_guild_info,
    "discord_get_user_info": dtool_get_user_info,
    "discord_add_reaction": dtool_add_reaction,
}


# ============================================================
# HTTP HEALTH SERVER
# ============================================================

async def health(_: web.Request) -> web.Response:
    return web.json_response({
        "ok": True, "service": config.bot_name,
        "slots": len(config.ai_slots),
        "provider": connector.provider,
        "model": config.ai_model,
    })


async def root(_: web.Request) -> web.Response:
    return web.Response(text=f"{config.bot_name} online | {len(config.ai_slots)} slots")


async def metrics(_: web.Request) -> web.Response:
    try:
        data = motor.snapshot()
        data["rotador"] = connector.stats()
    except Exception as exc:
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
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
    site = web.TCPSite(runner, "0.0.0.0", config.port)
    await site.start()
    print(f"HTTP server escuchando en puerto {config.port}")
    return runner


# ============================================================
# HELPERS
# ============================================================

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
    return client.user is not None and client.user in message.mentions


async def extract_prompt(message: discord.Message) -> tuple[str, bool]:
    content = message.content.strip()
    flag_image = False
    lowered = content.lower()
    prefixes = ("!image ", "/image ", "image: ", "!imagen ", "/imagen ",
                "imagen: ", "!draw ", "/draw ", "genera: ", "dibuja: ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return (content[len(prefix):].strip(), True)
    if message.attachments:
        flag_image = True
    return (content.strip(), flag_image)


# ============================================================
# ADJUNTOS
# ============================================================

async def download_attachments(message: discord.Message) -> list[dict]:
    if not message.attachments:
        return []
    uploads = Path(config.workspace) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    items = []
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
                        "path": str(target), "filename": att.filename,
                        "content_type": att.content_type or "",
                        "url": att.url, "size": len(data),
                        "is_image": (att.content_type or "").startswith("image/"),
                    })
                    print(f"Adjunto guardado: {target}")
            except Exception as exc:
                print(f"Error descargando {att.filename}: {exc}")
    return items


# ============================================================
# CONTEXTO
# ============================================================

def build_user_context(message: discord.Message, attachments: list[dict]) -> str:
    user = message.author
    lines = [
        "[Contexto Discord]",
        f"- Usuario: {user.display_name}",
        f"- Nombre: {user.name}",
        f"- ID: {user.id}",
        f"- Canal ID actual: {message.channel.id}",
    ]
    if message.guild:
        lines.append(f"- Servidor: {message.guild.name}")
        lines.append(f"- Servidor ID: {message.guild.id}")
        lines.append(f"- Miembros: {message.guild.member_count}")
    if isinstance(message.channel, discord.TextChannel):
        lines.append(f"- Canal: #{message.channel.name}")
    elif isinstance(message.channel, discord.DMChannel):
        lines.append("- Canal: DM")
    if attachments:
        lines.append("- Adjuntos:")
        for att in attachments:
            kind = "imagen" if att["is_image"] else "archivo"
            lines.append(f"  * [{kind}] {att['filename']} ({att['content_type']}) -> ruta local: {att['path']}")
        has_image = any(a["is_image"] for a in attachments)
        if has_image:
            lines.append("")
            lines.append("[Instrucciones]")
            lines.append("- El usuario adjuntó una imagen. Analízala directamente.")
    return "\n".join(lines)


# ============================================================
# ENVÍO DE ARCHIVOS
# ============================================================

async def send_files(chat: discord.Messageable, files: list) -> None:
    valid = []
    for item in files:
        try:
            if isinstance(item, (str, Path)):
                p = Path(item)
                if p.exists():
                    valid.append(discord.File(p))
            elif isinstance(item, tuple) and len(item) == 2:
                name, data = item
                if isinstance(data, bytes):
                    valid.append(discord.File(io.BytesIO(data), filename=name))
        except Exception as exc:
            print(f"Error preparando archivo: {exc}")
    for i in range(0, len(valid), 10):
        batch = valid[i:i + 10]
        try:
            await chat.send(files=batch)
        except Exception as exc:
            print(f"Error enviando lote: {exc}")


async def send_generated_image(chat: discord.Messageable, image_url: str) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status < 400:
                    data = await resp.read()
                    await chat.send(file=discord.File(io.BytesIO(data), filename="subtom.png"))
                    return
                await chat.send(f"Imagen generada: {image_url}")
    except Exception as exc:
        print(f"Error descargando imagen generada: {exc}")
        try:
            await chat.send(f"Imagen generada: {image_url}")
        except Exception:
            pass


# ============================================================
# EVENTOS
# ============================================================

@client.event
async def on_ready() -> None:
    print(f"{config.bot_name} | {len(config.ai_slots)} slots | {connector.provider} | {config.ai_model}")
    # Registrar herramientas Discord en el agente
    agent.register_discord_tools(DISCORD_TOOLS)


@client.event
async def on_message(message: discord.Message) -> None:
    if not await should_reply(message):
        return
    prompt, flag_image = await extract_prompt(message)
    attachments = await download_attachments(message)
    if not prompt and not attachments:
        await message.channel.send("No entendí tu mensaje.")
        return
    user_context = build_user_context(message, attachments)
    full_prompt = f"{user_context}\n\n{prompt}".strip()
    try:
        await message.add_reaction("⏳")
    except Exception:
        pass
    try:
        result = await agent.ask(message.author.id, message.channel.id, full_prompt, force_image=flag_image)
    except Exception as exc:
        print(f"Error procesando: {exc}")
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
    await send_long(message.channel, response)
    if image_url:
        await send_generated_image(message.channel, image_url)
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
        await connector.close()
        await agent.close()
        await client.close()
        if runner is not None:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
