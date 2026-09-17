from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg

from config import config
from connector import connector
from file_tools import FileTools
from motor import motor, TaskType, PromptLang


IMAGES_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}

class Agent:

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
        self.files = FileTools(config.workspace)

        from collections import deque
        self._flux_keys = deque(
            key for key in (config.flux_api_key_1, config.flux_api_key_2) if key
        )

    async def init(self) -> None:
        self.pool = await asyncpg.create_pool(
            config.database_url, min_size=1, max_size=3, command_timeout=60,
        )
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subtom_memory (
                    id BIGSERIAL PRIMARY KEY KEY,
                    user_id BIGINT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS subtom_memory_lookup
                ON subtom_memory(user_id, channel_id, created_at DESC)
                """
            )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def cleanup_memory(self) -> None:
        if not self.pool or config.memory_limit <= 0:
            return
        await self.pool.execute(
            """
            DELETE FROM subtom_memory
            WHERE created_at < NOW() - ($1 * INTERVAL '1 minute')
            """,
            config.memory_limit,
        )

    async def save(self, user_id: int, channel_id: int, role: str, content: str) -> None:
        if not self.pool:
            return
        await self.pool.execute(
            """
            INSERT INTO subtom_memory(user_id, channel_id, role, content)
            VALUES ($1, $2, $3, $4)
            """,
            user_id, channel_id, role, content[:30000],
        )

    async def history(self, user_id: int, channel_id: int) -> list[dict[str, Any]]:
        if not self.pool:
            return []
        rows = await self.pool.fetch(
            """
            SELECT role, content FROM subtom_memory
            WHERE user_id = $1 AND channel_id = $2
            ORDER BY created_at DESC LIMIT $3
            """,
            user_id, channel_id, config.memory_messages,
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def _extract_image_paths(self, prompt: str) -> list[str]:
        paths: list[str] = []
        for m in re.finditer(r"->\s*ruta local:\s*(\S+)", prompt):
            raw = m.group(1).strip().strip('"')
            try:
                p = Path(raw)
                if (p.suffix.lower() in IMAGES_EXTENSIONS
                        and p.exists() and p.is_file()):
                    paths.append(str(p))
            except Exception:
                continue
        seen, unique = set(), []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _build_multimodal_content(self, prompt: str, image_paths: list[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            try:
                info = self.files.read_image_base64(path)
                if info["size"] > 20 * 1024 * 1024:
                    content.append({"type": "text", "text": f"[Imagen >20MB omitida: {info['filename']}]"})
                    continue
                content.append({"type": "image_url", "image_url": {"url": info["data_url"]}})
            except Exception as exc:
                content.append({"type": "text", "text": f"[No se pudo cargar {path}: {exc}]"})
        return content

    def _select_model(self, prompt: str, task: TaskType, force_image: bool = False) -> list[str]:
        return [config.ai_model]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "sandbox_verify_change",
                    "description": "Verifica los cambios antes de subir un archivo Python al repositorio.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_content": {"type": "string"},
                            "new_content": {"type": "string"}
                        },
                        "required": ["file_path", "old_content", "new_content"]
                    },
                },
            },
            # Other schemas...
        ]

    async def github_write(self, repo: str, path: str, content: str, message: str):
        # Leer el contenido actual del archivo
        old_file = await self.github_read(repo, path)
        old_content = old_file.get("content", "")

        # Si el archivo tiene extensión .py, realizar la verificación avanzada
        if path.endswith(".py"):
            sandbox = Sandbox()
            verification_result = sandbox.verify_python_change(
                file_path=path,
                old_content=old_content,
                new_content=content
            )

            if not verification_result["ok"]:
                raise ValueError(f"Error en verificación: {verification_result['errors']}")

        # Proceder con la subida si todo está bien
        await self.github_write(repo=repo, path=path, content=content, message=message)

    # Otras funciones...
