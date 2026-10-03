from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
import discord

from ai import agent
from api_motor import setup_api
from config import config
from connector import connector
from motor import motor


_START_TIME = time.monotonic()
_HTTP_RUNNER: web.AppRunner | None = None
_SHUTTING_DOWN = False
_MESSAGE_LOCKS: dict[tuple[int, int], asyncio.Lock] = {}


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


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)


def _error_payload(exc: Exception, operation: str) -> dict[str, Any]:
    return {
        "ok": False,
        "operation": operation,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc) or repr(exc),
        },
    }


async def health(_: web.Request) -> web.Response:
    try:
        connector_stats = connector.stats()

        return web.json_response({
            "ok": True,
            "service": config.bot_name,
            "provider": connector.provider,
            "model": config.ai_model,
            "uptime_seconds": round(_uptime_seconds(), 1),
            "uptime_human": _uptime_human(),
            "discord_ready": client.is_ready(),
            "shutting_down": _SHUTTING_DOWN,
            "connector": connector_stats,
        })
    except Exception as exc:
        return web.json_response(
            _error_payload(exc, "health"),
            status=500,
        )


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
        data["discord_ready"] = client.is_ready()

        return web.json_response(data)

    except Exception as exc:
        return web.json_response(
            _error_payload(exc, "metrics"),
            status=500,
        )


async def start_http() -> web.AppRunner:
    app = web.Application(
        client_max_size=100 * 1024 * 1024
    )

    setup_api(app)

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
        reuse_address=True,
    )

    await site.start()

    print(
        f"HTTP server escuchando en puerto {config.port}"
    )

    return runner


async def send_long(
    chat: discord.Messageable,
    text: str,
) -> None:
    if text is None:
        return

    text = str(text)

    if not text.strip():
        return

    chunks = []

    while text:
        if len(text) <= 1900:
            chunks.append(text)
            break

        split_at = text.rfind("\n", 0, 1900)

        if split_at < 1000:
            split_at = text.rfind(" ", 0, 1900)

        if split_at < 500:
            split_at = 1900

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    for chunk in chunks:
        await chat.send(
            chunk,
            allowed_mentions=discord.AllowedMentions.none(),
        )


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

    if not content:
        content = ""

    lowered = content.lower()

    image_prefixes = (
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

    for prefix in image_prefixes:
        if lowered.startswith(prefix):
            return (
                content[len(prefix):].strip(),
                True,
            )

    return (
        content,
        bool(message.attachments),
    )


def _safe_filename(name: str) -> str:
    name = Path(name).name.strip()

    if not name:
        name = "archivo"

    invalid = '<>:"/\\|?*'

    for char in invalid:
        name = name.replace(char, "_")

    return name[:180]


async def download_attachments(
    message: discord.Message,
) -> list[dict[str, Any]]:
    if not message.attachments:
        return []

    uploads = (
        Path(config.workspace)
        / "uploads"
    )

    uploads.mkdir(
        parents=True,
        exist_ok=True,
    )

    items: list[dict[str, Any]] = []

    timeout = aiohttp.ClientTimeout(
        total=120,
        connect=20,
        sock_read=120,
    )

    connector_http = aiohttp.TCPConnector(
        limit=5,
        ttl_dns_cache=300,
    )

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector_http,
    ) as session:

        for index, attachment in enumerate(
            message.attachments
        ):
            filename = _safe_filename(
                attachment.filename
            )

            target = uploads / (
                f"{message.id}_{index}_{filename}"
            )

            try:
                async with session.get(
                    attachment.url,
                    allow_redirects=True,
                ) as response:

                    if response.status >= 400:
                        print(
                            f"[ATTACHMENT] HTTP "
                            f"{response.status}: "
                            f"{attachment.filename}"
                        )
                        continue

                    content_length = response.headers.get(
                        "Content-Length"
                    )

                    if content_length:
                        try:
                            if int(content_length) > 50 * 1024 * 1024:
                                print(
                                    f"[ATTACHMENT] Archivo demasiado "
                                    f"grande: {attachment.filename}"
                                )
                                continue
                        except ValueError:
                            pass

                    data = await response.read()

                    if len(data) > 50 * 1024 * 1024:
                        print(
                            f"[ATTACHMENT] Archivo demasiado "
                            f"grande: {attachment.filename}"
                        )
                        continue

                    target.write_bytes(data)

                    content_type = (
                        attachment.content_type
                        or ""
                    )

                    items.append({
                        "path": str(target),
                        "filename": attachment.filename,
                        "content_type": content_type,
                        "url": attachment.url,
                        "size": len(data),
                        "is_image": content_type.startswith(
                            "image/"
                        ),
                        "is_pdf": (
                            content_type
                            == "application/pdf"
                        ),
                        "is_text": content_type.startswith(
                            "text/"
                        ),
                    })

                    print(
                        f"[ATTACHMENT] OK: {target}"
                    )

            except Exception as exc:
                print(
                    f"[ATTACHMENT] Error "
                    f"{attachment.filename}: "
                    f"{type(exc).__name__}: {exc}"
                )

    return items


def build_user_context(
    message: discord.Message,
    attachments: list[dict[str, Any]],
) -> str:
    user = message.author

    lines = [
        "[Contexto Discord]",
        f"- Usuario: {user.display_name}",
        f"- Nombre: {user.name}",
        f"- ID: {user.id}",
        f"- Avatar: {user.display_avatar.url}",
        f"- Canal ID: {message.channel.id}",
    ]

    if message.guild:
        lines.append(
            f"- Servidor: {message.guild.name}"
        )
        lines.append(
            f"- Servidor ID: {message.guild.id}"
        )
        lines.append(
            f"- Miembros: {message.guild.member_count}"
        )

        if message.guild.icon:
            lines.append(
                f"- Icono servidor: "
                f"{message.guild.icon.url}"
            )

    if isinstance(
        message.channel,
        discord.TextChannel,
    ):
        lines.append(
            f"- Canal: #{message.channel.name}"
        )

        if message.channel.topic:
            lines.append(
                f"- Tema del canal: "
                f"{message.channel.topic}"
            )

    elif isinstance(
        message.channel,
        discord.DMChannel,
    ):
        lines.append("- Canal: DM")

    if attachments:
        lines.append(
            "- Archivos adjuntos:"
        )

        for attachment in attachments:
            kind = (
                "imagen"
                if attachment["is_image"]
                else "archivo"
            )

            lines.append(
                f"  * [{kind}] "
                f"{attachment['filename']} "
                f"({attachment['content_type']}) "
                f"-> {attachment['path']}"
            )

    return "\n".join(lines)


def _workspace_candidates(
    raw: str | Path,
) -> list[Path]:
    if isinstance(raw, Path):
        value = str(raw)
    else:
        value = str(raw).strip()

    if not value:
        return []

    p = Path(value)

    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)

    workspace = Path(config.workspace)

    candidates.extend([
        workspace / p,
        workspace / p.name,
        workspace / "generated" / p.name,
        workspace / "uploads" / p.name,
    ])

    unique: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue

        key = str(resolved)

        if key not in seen:
            seen.add(key)
            unique.append(resolved)

    return unique


def _resolve_workspace_path(
    raw: str | Path,
) -> Path | None:
    for candidate in _workspace_candidates(raw):
        try:
            if (
                candidate.exists()
                and candidate.is_file()
            ):
                return candidate
        except Exception:
            continue

    return None


def _normalize_files(
    files: Any,
) -> list[Any]:
    if files is None:
        return []

    if isinstance(
        files,
        (str, Path, bytes),
    ):
        return [files]

    if isinstance(files, tuple):
        if (
            len(files) == 2
            and isinstance(files[0], str)
            and isinstance(files[1], bytes)
        ):
            return [files]

        return list(files)

    if isinstance(files, list):
        return files

    return [files]


def _prepare_discord_file(
    item: Any,
) -> tuple[discord.File | None, str | None]:
    try:
        if isinstance(item, discord.File):
            return item, None

        if isinstance(item, bytes):
            return (
                discord.File(
                    io.BytesIO(item),
                    filename="archivo.bin",
                ),
                None,
            )

        if isinstance(item, tuple):
            if len(item) != 2:
                return (
                    None,
                    "Formato de archivo inválido.",
                )

            name, data = item

            if not isinstance(name, str):
                name = "archivo.bin"

            if isinstance(data, bytes):
                return (
                    discord.File(
                        io.BytesIO(data),
                        filename=_safe_filename(name),
                    ),
                    None,
                )

            if isinstance(data, bytearray):
                return (
                    discord.File(
                        io.BytesIO(bytes(data)),
                        filename=_safe_filename(name),
                    ),
                    None,
                )

            if isinstance(data, (str, Path)):
                resolved = _resolve_workspace_path(data)

                if resolved is None:
                    return (
                        None,
                        f"Archivo no encontrado: {data}",
                    )

                return (
                    discord.File(
                        resolved,
                        filename=_safe_filename(name),
                    ),
                    None,
                )

            return (
                None,
                "Contenido de archivo no soportado.",
            )

        if isinstance(item, (str, Path)):
            resolved = _resolve_workspace_path(item)

            if resolved is None:
                return (
                    None,
                    f"Archivo no encontrado: {item}",
                )

            return (
                discord.File(resolved),
                None,
            )

        return (
            None,
            f"Tipo de archivo no soportado: "
            f"{type(item).__name__}",
        )

    except Exception as exc:
        return (
            None,
            f"{type(exc).__name__}: {exc}",
        )


async def send_files(
    chat: discord.Messageable,
    files: Any,
) -> dict[str, Any]:
    normalized = _normalize_files(files)

    if not normalized:
        return {
            "ok": True,
            "sent": 0,
            "failed": 0,
            "errors": [],
        }

    prepared: list[discord.File] = []
    errors: list[str] = []

    for item in normalized:
        discord_file, error = _prepare_discord_file(
            item
        )

        if discord_file is not None:
            prepared.append(discord_file)
        elif error:
            errors.append(error)
            print(
                f"[SEND_FILES] {error}"
            )

    sent = 0

    for index in range(
        0,
        len(prepared),
        10,
    ):
        batch = prepared[
            index:index + 10
        ]

        try:
            await chat.send(
                files=batch,
                allowed_mentions=discord.AllowedMentions.none(),
            )

            sent += len(batch)

        except Exception as exc:
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(
                f"Error enviando lote: {error_text}"
            )

            print(
                f"[SEND_FILES] {error_text}"
            )

            for single in batch:
                try:
                    await chat.send(
                        file=single,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                    sent += 1

                except Exception as single_exc:
                    errors.append(
                        "Error enviando archivo "
                        f"individual: "
                        f"{type(single_exc).__name__}: "
                        f"{single_exc}"
                    )

    return {
        "ok": not errors,
        "sent": sent,
        "failed": len(errors),
        "errors": errors,
    }


async def send_generated_image(
    chat: discord.Messageable,
    image_url: str,
) -> bool:
    if not image_url:
        return False

    timeout = aiohttp.ClientTimeout(
        total=120
    )

    try:
        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.get(
                image_url,
                allow_redirects=True,
            ) as response:

                if response.status >= 400:
                    await chat.send(
                        f"Imagen generada: {image_url}",
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                    return False

                data = await response.read()

                if not data:
                    raise RuntimeError(
                        "La imagen generada llegó vacía."
                    )

                await chat.send(
                    file=discord.File(
                        io.BytesIO(data),
                        filename="generated_image.png",
                    ),
                    allowed_mentions=discord.AllowedMentions.none(),
                )

                return True

    except Exception as exc:
        print(
            f"[IMAGE] Error: "
            f"{type(exc).__name__}: {exc}"
        )

        try:
            await chat.send(
                f"Imagen generada: {image_url}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            pass

        return False


def _normalize_agent_result(
    result: Any,
) -> tuple[Any, Any, Any]:
    response = None
    image_url = None
    files = None

    if result is None:
        return response, image_url, files

    if isinstance(result, tuple):
        if len(result) == 3:
            return (
                result[0],
                result[1],
                result[2],
            )

        if len(result) == 2:
            return (
                result[0],
                result[1],
                None,
            )

        if len(result) == 1:
            return (
                result[0],
                None,
                None,
            )

    if isinstance(result, dict):
        response = (
            result.get("response")
            or result.get("text")
            or result.get("message")
        )

        image_url = (
            result.get("image_url")
            or result.get("image")
        )

        files = result.get("files")

        return (
            response,
            image_url,
            files,
        )

    return result, None, None


def _get_message_lock(
    message: discord.Message,
) -> asyncio.Lock:
    key = (
        message.author.id,
        message.channel.id,
    )

    lock = _MESSAGE_LOCKS.get(key)

    if lock is None:
        lock = asyncio.Lock()
        _MESSAGE_LOCKS[key] = lock

    return lock


async def _cleanup_message_lock(
    message: discord.Message,
) -> None:
    key = (
        message.author.id,
        message.channel.id,
    )

    lock = _MESSAGE_LOCKS.get(key)

    if lock is not None and not lock.locked():
        _MESSAGE_LOCKS.pop(key, None)


async def _process_message(
    message: discord.Message,
) -> None:
    prompt, flag_image = await extract_prompt(
        message
    )

    attachments = await download_attachments(
        message
    )

    if not prompt and not attachments:
        await message.channel.send(
            "No entendí tu mensaje.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    user_context = build_user_context(
        message,
        attachments,
    )

    full_prompt = (
        f"{user_context}\n\n{prompt}"
    ).strip()

    reaction_added = False

    try:
        await message.add_reaction("⏳")
        reaction_added = True
    except Exception:
        pass

    try:
        result = await agent.ask(
            message.author.id,
            message.channel.id,
            full_prompt,
            force_image=flag_image,
        )

        response, image_url, files = (
            _normalize_agent_result(result)
        )

        if response:
            await send_long(
                message.channel,
                str(response),
            )

        if files:
            file_result = await send_files(
                message.channel,
                files,
            )

            if (
                not file_result["ok"]
                and file_result["sent"] == 0
            ):
                await message.channel.send(
                    "No pude enviar los archivos generados.",
                    allowed_mentions=discord.AllowedMentions.none(),
                )

        if image_url and not files:
            await send_generated_image(
                message.channel,
                str(image_url),
            )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc) or repr(exc)

        print(
            f"[MESSAGE ERROR] "
            f"{error_type}: {error_message}"
        )

        try:
            await message.channel.send(
                f"Error real: {error_type}: "
                f"{error_message}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            pass

    finally:
        if reaction_added:
            try:
                if client.user:
                    await message.remove_reaction(
                        "⏳",
                        client.user,
                    )
            except Exception:
                pass


@client.event
async def on_ready() -> None:
    print(
        f"{config.bot_name} | "
        f"{len(config.ai_slots)} slots | "
        f"principal: {connector.provider} | "
        f"modelo: {config.ai_model}"
    )


@client.event
async def on_message(
    message: discord.Message,
) -> None:
    if not await should_reply(message):
        return

    lock = _get_message_lock(
        message
    )

    if lock.locked():
        return

    async with lock:
        try:
            await _process_message(
                message
            )
        finally:
            await _cleanup_message_lock(
                message
            )


@client.event
async def on_error(
    event_method: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    import traceback

    print(
        f"[DISCORD ERROR] Evento: {event_method}"
    )
    traceback.print_exc()


async def main() -> None:
    global _HTTP_RUNNER
    global _SHUTTING_DOWN

    try:
        await agent.init()

        _HTTP_RUNNER = await start_http()

        await client.start(
            config.discord_token
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        print(
            f"[MAIN ERROR] "
            f"{type(exc).__name__}: "
            f"{str(exc) or repr(exc)}"
        )
        raise

    finally:
        _SHUTTING_DOWN = True

        print(
            "Cerrando bot..."
        )

        try:
            await connector.close()
        except Exception as exc:
            print(
                f"[CLOSE CONNECTOR] "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            await agent.close()
        except Exception as exc:
            print(
                f"[CLOSE AGENT] "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            from sandbox import sandbox

            await sandbox.close()

        except Exception as exc:
            print(
                f"[CLOSE SANDBOX] "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            if not client.is_closed():
                await client.close()
        except Exception as exc:
            print(
                f"[CLOSE DISCORD] "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            if _HTTP_RUNNER is not None:
                await _HTTP_RUNNER.cleanup()
        except Exception as exc:
            print(
                f"[CLOSE HTTP] "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    asyncio.run(main())
