from __future__ import annotations

import asyncio
import base64
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


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}


class Agent:

    # ==================================================================
    # RESPALDO ESTÁTICO DE VISIÓN
    # ==================================================================

    VISION_MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-thinking-exp:free",
        "qwen/qwen-2-vl-7b-instruct:free",
        "qwen/qwen-2-vl-72b-instruct:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "microsoft/phi-3.5-vision-instruct:free",
    ]

    # ==================================================================
    # MODELOS DE CÓDIGO
    # ==================================================================

    CODE_MODELS = [
        "qwen/qwen3-coder:free",
        "qwen/qwen3.6-plus:free",
        "minimax/minimax-m2.5:free",
        "minimax/minimax-m2.7:free",
        "cohere/north-mini-code:free",
        "z-ai/glm-4.5-air:free",
        "z-ai/glm-4.5:free",
        "stepfun/step-3.5-flash:free",
        "arcee-ai/trinity-large-preview:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "qwen/qwen3-32b:free",
        "qwen/qwen3-235b-a22b:free",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-14b:free",
        "qwen/qwen3-8b:free",
        "deepseek/deepseek-v4-flash:free",
        "deepseek/deepseek-v4-pro:free",
        "deepseek/deepseek-chat-v3.1:free",
        "deepseek/deepseek-r1:free",
        "mistralai/devstral-small:free",
        "nex-agi/nex-n2.5-pro:free",
        "poolside/laguna-xs-2.1:free",
        "poolside/laguna-s-2.1:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "qwen/qwen3.5-9b:free",
        "microsoft/phi-4:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ]

    # ==================================================================
    # MODELOS DE TEXTO
    # ==================================================================

    TEXT_MODELS = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "minimax/minimax-m2.7:free",
        "minimax/minimax-m3:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat-v3.1:free",
        "z-ai/glm-4.5-air:free",
        "z-ai/glm-4.5:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "stepfun/step-3.5-flash:free",
        "arcee-ai/trinity-large-preview:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "qwen/qwen3.6-plus:free",
        "deepseek/deepseek-r1-0528:free",
        "deepseek/deepseek-v4-flash:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "qwen/qwen3-32b:free",
        "qwen/qwen3-235b-a22b:free",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-14b:free",
        "qwen/qwen3-8b:free",
        "microsoft/phi-4:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "liquid/lfm-2.5-2.6b:free",
        "poolside/laguna-m.1:free",
        "tngtech/deepseek-r1t2-chimera:free",
        "tngtech/deepseek-r1t-chimera:free",
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "gryphe/mythomax-l2-13b:free",
    ]

    # ==================================================================
    # INIT
    # ==================================================================

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

        self.files = FileTools(config.workspace)

        self._vision_models: list[str] = []
        self._vision_loaded: bool = False

        from collections import deque
        self._flux_keys = deque(
            key
            for key in (
                config.flux_api_key_1,
                config.flux_api_key_2,
            )
            if key
        )

    # ==================================================================
    # BASE DE DATOS / MEMORIA
    # ==================================================================

    async def init(self) -> None:
        self.pool = await asyncpg.create_pool(
            config.database_url,
            min_size=1,
            max_size=3,
            command_timeout=60,
        )

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subtom_memory (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS subtom_memory_lookup
                ON subtom_memory(user_id, channel_id, created_at DESC)
                """
            )

        await self._refresh_vision_models()

        try:
            await motor.prewarm()
        except Exception as exc:
            print(f"[MOTOR] prewarm falló: {exc}")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def cleanup_memory(self) -> None:
        if not self.pool:
            return
        if config.memory_limit <= 0:
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
            VALUES($1, $2, $3, $4)
            """,
            user_id, channel_id, role, content[:30000],
        )

    async def history(self, user_id: int, channel_id: int) -> list[dict[str, Any]]:
        if not self.pool:
            return []
        rows = await self.pool.fetch(
            """
            SELECT role, content
            FROM subtom_memory
            WHERE user_id = $1 AND channel_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id, channel_id, config.memory_messages,
        )
        rows = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    # ==================================================================
    # DETECCIÓN DINÁMICA DE MODELOS CON VISIÓN
    # ==================================================================

    async def _refresh_vision_models(self) -> None:
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("https://openrouter.ai/api/v1/models") as resp:
                    data = await resp.json(content_type=None)
        except Exception as exc:
            print(f"[VISION] No se pudo cargar lista de modelos: {exc}. Usando lista estática.")
            self._vision_models = list(self.VISION_MODELS)
            self._vision_loaded = False
            return

        models = data.get("data") or []
        vision_free = []
        for m in models:
            model_id = m.get("id", "")
            if not model_id.endswith(":free"):
                continue
            arch = m.get("architecture") or {}
            input_mods = arch.get("input_modalities") or []
            if "image" in input_mods:
                vision_free.append(model_id)

        if not vision_free:
            print("[VISION] OpenRouter no devolvió modelos free con visión. Usando lista estática.")
            self._vision_models = list(self.VISION_MODELS)
            self._vision_loaded = False
            return

        self._vision_models = vision_free
        self._vision_loaded = True
        print(f"[VISION] {len(vision_free)} modelos :free con visión detectados:")
        for mid in vision_free[:10]:
            print(f"  - {mid}")
        if len(vision_free) > 10:
            print(f"  ... y {len(vision_free) - 10} más")

    # ==================================================================
    # VISIÓN
    # ==================================================================

    def _extract_image_paths(self, prompt: str) -> list[str]:
        paths = []
        for match in re.finditer(r"->\s*ruta local:\s*(\S+)", prompt):
            raw = match.group(1).strip().strip('",')
            try:
                p = Path(raw)
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists() and p.is_file():
                    paths.append(str(p))
            except Exception:
                continue
        seen = set()
        unique = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _build_multimodal_content(self, prompt: str, image_paths: list[str]) -> list[dict[str, Any]]:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            try:
                info = self.files.read_image_base64(path)
                if info["size"] > 20 * 1024 * 1024:
                    content.append({"type": "text", "text": f"[Imagen omitida por tamaño >20MB: {info['filename']}]"})
                    continue
                content.append({"type": "image_url", "image_url": {"url": info["data_url"]}})
            except Exception as exc:
                content.append({"type": "text", "text": f"[No se pudo cargar imagen {path}: {exc}]"})
        return content

    # ==================================================================
    # SELECCIÓN DE MODELO SEGÚN LA TAREA
    # ==================================================================

    def _select_model(self, prompt: str, task: TaskType, force_image: bool = False) -> list[str]:
        vision_list = self._vision_models or self.VISION_MODELS
        if force_image or task == TaskType.IMAGE_GEN:
            return vision_list
        if task == TaskType.IMAGE_READ:
            return vision_list
        if task == TaskType.CODE:
            return motor.pick_models(self.CODE_MODELS, task, top=5)
        return motor.pick_models(self.TEXT_MODELS, task, top=5)

    # ==================================================================
    # DEFINICIÓN DE HERRAMIENTAS
    # ==================================================================

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            # ------------------------- BÚSQUEDA WEB -------------------------
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Busca información actual en internet con DuckDuckGo. Devuelve título, URL y snippet. Usa web_fetch después para leer una URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Consulta en lenguaje natural."},
                            "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Descarga una página web y devuelve el texto legible de su contenido. Úsala DESPUÉS de web_search para leer una URL concreta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL completa (http/https)."},
                            "max_chars": {"type": "integer", "default": 8000, "minimum": 500, "maximum": 30000},
                        },
                        "required": ["url"],
                    },
                },
            },
            # ------------------------- ARCHIVOS -------------------------
            {
                "type": "function",
                "function": {
                    "name": "file_list",
                    "description": "Lista archivos y carpetas dentro del workspace seguro de Subtom.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Ruta relativa dentro del workspace.", "default": "."}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": "Lee un archivo de texto dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write",
                    "description": "Crea o reemplaza completamente un archivo de texto dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_append",
                    "description": "Añade contenido al final de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_search",
                    "description": "Busca archivos y carpetas mediante un patrón dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "description": "Obtiene información sobre un archivo o carpeta.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_mkdir",
                    "description": "Crea una carpeta dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_copy",
                    "description": "Copia un archivo o carpeta dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
                        "required": ["source", "destination"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_move",
                    "description": "Mueve o renombra un archivo o carpeta.",
                    "parameters": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
                        "required": ["source", "destination"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_delete",
                    "description": "Elimina un archivo o carpeta. Debe usarse con cuidado.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean", "default": False}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_zip",
                    "description": "Crea un archivo ZIP dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"output_zip": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}},
                        "required": ["output_zip", "sources"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_zip",
                    "description": "Lista el contenido de un archivo ZIP.",
                    "parameters": {
                        "type": "object",
                        "properties": {"zip_name": {"type": "string"}},
                        "required": ["zip_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "unzip_file",
                    "description": "Extrae un ZIP dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"zip_name": {"type": "string"}, "destination": {"type": "string", "default": "."}},
                        "required": ["zip_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_detect_kind",
                    "description": "Detecta el tipo real de un archivo: image, pdf, text, binary o directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_any",
                    "description": "Lee cualquier archivo y devuelve algo útil: imagen (info), pdf (texto), texto (contenido), binario (info).",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_pdf",
                    "description": "Extrae el texto de un archivo PDF.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_text",
                    "description": "Lee un archivo de texto detectando encoding automáticamente.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_image_info",
                    "description": "Devuelve información de una imagen: ancho, alto, formato, modo y tamaño.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_tree",
                    "description": "Devuelve un árbol recursivo del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "default": "."}, "max_depth": {"type": "integer", "default": 3}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_grep",
                    "description": "Busca texto dentro de archivos del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}, "extensions": {"type": "array", "items": {"type": "string"}}},
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_head",
                    "description": "Devuelve las primeras N líneas de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "lines": {"type": "integer", "default": 20}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_tail",
                    "description": "Devuelve las últimas N líneas de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "lines": {"type": "integer", "default": 20}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_count_lines",
                    "description": "Cuenta las líneas de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_replace",
                    "description": "Reemplaza texto dentro de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "count": {"type": "integer", "default": -1}},
                        "required": ["path", "old", "new"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_hash",
                    "description": "Devuelve el hash (sha256 por defecto) de un archivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "algorithm": {"type": "string", "default": "sha256"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_exists",
                    "description": "Comprueba si una ruta existe dentro del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_backup",
                    "description": "Crea una copia de seguridad con timestamp de un archivo o carpeta. Úsalo SIEMPRE antes de modificar tu propio código.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_image_resize",
                    "description": "Redimensiona una imagen.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "output": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "keep_aspect": {"type": "boolean", "default": True}},
                        "required": ["path", "output", "width"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_image_thumbnail",
                    "description": "Genera un thumbnail de una imagen.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "output": {"type": "string"}, "size": {"type": "integer", "default": 256}},
                        "required": ["path", "output"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_image_convert",
                    "description": "Convierte una imagen a otro formato.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "output": {"type": "string"}, "format": {"type": "string"}},
                        "required": ["path", "output"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_json",
                    "description": "Lee un archivo JSON.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write_json",
                    "description": "Escribe un archivo JSON a partir de un objeto.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "data": {}, "indent": {"type": "integer", "default": 2}},
                        "required": ["path", "data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_csv",
                    "description": "Lee un CSV y devuelve las filas como dicts.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "max_rows": {"type": "integer"}},
                        "required": ["path"],
                    },
                },
            },
            # ------------------------- GITHUB BÁSICO -------------------------
            {
                "type": "function",
                "function": {
                    "name": "github_list",
                    "description": "Lista los repositorios del usuario autenticado en GitHub.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_read",
                    "description": "Lee un archivo de un repositorio GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "path": {"type": "string"}},
                        "required": ["repo", "path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_write",
                    "description": "Crea o actualiza un archivo en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}, "message": {"type": "string"}},
                        "required": ["repo", "path", "content", "message"],
                    },
                },
            },
            # ========================================================
            # GITHUB AVANZADO
            # ========================================================
            {
                "type": "function",
                "function": {
                    "name": "github_upload_file",
                    "description": "Sube un archivo (texto o binario como PNG) al repositorio. Úsalo para subir el logo generado por FLUX usando su local_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo"},
                            "repo_path": {"type": "string", "description": "Ruta destino en el repo (ej: 'public/logo.png')"},
                            "local_path": {"type": "string", "description": "Ruta del archivo dentro del workspace (ej: 'generated/logo.png')"},
                            "message": {"type": "string", "description": "Mensaje del commit"},
                        },
                        "required": ["repo", "repo_path", "local_path", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_upload_project",
                    "description": "Sube VARIOS archivos del workspace a GitHub en UN SOLO commit. Úsalo para subir un proyecto completo de golpe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo"},
                            "files": {"type": "object", "description": "Diccionario {ruta_en_repo: ruta_en_workspace}. Ej: {'index.html': 'myapp/index.html', 'logo.png': 'generated/logo.png'}"},
                            "message": {"type": "string", "description": "Mensaje del commit"},
                            "branch": {"type": "string", "default": "main"},
                        },
                        "required": ["repo", "files", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_create_repo",
                    "description": "Crea un repositorio nuevo en GitHub para el usuario autenticado.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "private": {"type": "boolean", "default": False}},
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_create_issue",
                    "description": "Crea un issue en un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "title": {"type": "string"}, "body": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}}},
                        "required": ["repo", "title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_list_issues",
                    "description": "Lista los issues de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "state": {"type": "string", "default": "open", "enum": ["open", "closed", "all"]}},
                        "required": ["repo"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_create_pr",
                    "description": "Crea un pull request en un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "title": {"type": "string"}, "head": {"type": "string", "description": "Rama con los cambios"}, "base": {"type": "string", "default": "main", "description": "Rama destino"}, "body": {"type": "string"}},
                        "required": ["repo", "title", "head"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_list_prs",
                    "description": "Lista los pull requests de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "state": {"type": "string", "default": "open", "enum": ["open", "closed", "all"]}},
                        "required": ["repo"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_merge_pr",
                    "description": "Fusiona un pull request en un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "pr_number": {"type": "integer"}, "merge_method": {"type": "string", "default": "merge", "enum": ["merge", "squash", "rebase"]}},
                        "required": ["repo", "pr_number"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_list_branches",
                    "description": "Lista las ramas de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}},
                        "required": ["repo"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_delete_branch",
                    "description": "Elimina una rama de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "branch": {"type": "string"}},
                        "required": ["repo", "branch"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_get_commit",
                    "description": "Obtiene información detallada de un commit en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "sha": {"type": "string"}},
                        "required": ["repo", "sha"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_list_commits",
                    "description": "Lista los commits de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "branch": {"type": "string", "default": "main"}, "per_page": {"type": "integer", "default": 10}},
                        "required": ["repo"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_get_file",
                    "description": "Obtiene el contenido y metadatos de un archivo en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "path": {"type": "string"}, "ref": {"type": "string", "default": "main"}},
                        "required": ["repo", "path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_update_file",
                    "description": "Actualiza un archivo existente en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "path": {"type": "string"}, "content": {"type": "string"}, "message": {"type": "string"}, "sha": {"type": "string", "description": "SHA del archivo a actualizar"}},
                        "required": ["repo", "path", "content", "message", "sha"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_delete_file",
                    "description": "Elimina un archivo de un repositorio de GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string", "description": "owner/repo"}, "path": {"type": "string"}, "message": {"type": "string"}, "sha": {"type": "string", "description": "SHA del archivo a eliminar"}},
                        "required": ["repo", "path", "message", "sha"],
                    },
                },
            },
            # ========================================================
            # VERCEL BÁSICO
            # ========================================================
            {
                "type": "function",
                "function": {
                    "name": "vercel_projects",
                    "description": "Lista proyectos de Vercel.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_deployments",
                    "description": "Lista deployments de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            # ========================================================
            # VERCEL AVANZADO
            # ========================================================
            {
                "type": "function",
                "function": {
                    "name": "vercel_project_info",
                    "description": "Obtiene info de un proyecto Vercel: ID, framework, última deploy, URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_list_envs",
                    "description": "Lista las variables de entorno de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_set_env",
                    "description": "Crea o actualiza una variable de entorno en Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project": {"type": "string"},
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                            "target": {"type": "array", "items": {"type": "string"}, "description": "production, preview, development"},
                        },
                        "required": ["project", "key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_delete_env",
                    "description": "Elimina una variable de entorno de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "env_id": {"type": "string"}},
                        "required": ["project", "env_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_redeploy",
                    "description": "Fuerza un nuevo deployment de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "target": {"type": "string", "default": "production"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_build_logs",
                    "description": "Obtiene los logs del último deployment de un proyecto.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "limit": {"type": "integer", "default": 100}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_list_deployments",
                    "description": "Lista todos los deployments de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_get_deployment",
                    "description": "Obtiene los detalles de un deployment específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {"deployment_id": {"type": "string"}},
                        "required": ["deployment_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_cancel_deployment",
                    "description": "Cancela un deployment en curso.",
                    "parameters": {
                        "type": "object",
                        "properties": {"deployment_id": {"type": "string"}},
                        "required": ["deployment_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_list_domains",
                    "description": "Lista los dominios asociados a un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_add_domain",
                    "description": "Añade un dominio personalizado a un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "domain": {"type": "string"}},
                        "required": ["project", "domain"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_remove_domain",
                    "description": "Elimina un dominio de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}, "domain": {"type": "string"}},
                        "required": ["project", "domain"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_list_aliases",
                    "description": "Lista los alias de un proyecto Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_delete_project",
                    "description": "Elimina un proyecto de Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_create_project",
                    "description": "Crea un nuevo proyecto en Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "framework": {"type": "string", "default": "nextjs"}},
                        "required": ["name"],
                    },
                },
            },
        ]

    # ==================================================================
    # EJECUCIÓN DE HERRAMIENTAS
    # ==================================================================

    async def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            # --- BÚSQUEDA WEB ---
            if name == "web_search":
                return await self._cached_search(args)
            if name == "web_fetch":
                return await self._cached_fetch(args)

            # --- ARCHIVOS ---
            if name == "file_list":
                return {"files": self.files.list_files(args.get("path", "."))}
            if name == "file_read":
                return {"content": self.files.read_file(args["path"])}
            if name == "file_write":
                return {"path": self.files.write_file(args["path"], args["content"])}
            if name == "file_append":
                return {"path": self.files.append_file(args["path"], args["content"])}
            if name == "file_search":
                return {"matches": self.files.search(args["query"], args.get("path", "."))}
            if name == "file_info":
                return self.files.info(args["path"])
            if name == "file_mkdir":
                return {"path": self.files.mkdir(args["path"])}
            if name == "file_copy":
                return {"path": self.files.copy(args["source"], args["destination"])}
            if name == "file_move":
                return {"path": self.files.move(args["source"], args["destination"])}
            if name == "file_delete":
                return {"deleted": self.files.delete(args["path"], args.get("recursive", False))}
            if name == "create_zip":
                return {"path": self.files.zip_create(args["output_zip"], args["sources"])}
            if name == "list_zip":
                return {"files": self.files.zip_list(args["zip_name"])}
            if name == "unzip_file":
                return {"extracted": self.files.zip_extract(args["zip_name"], args.get("destination", "."))}
            if name == "file_detect_kind":
                return {"kind": self.files.detect_kind(args["path"])}
            if name == "file_read_any":
                data = self.files.read_any(args["path"])
                if isinstance(data, dict):
                    data.pop("base64", None)
                    data.pop("data_url", None)
                return data
            if name == "file_read_pdf":
                return {"text": self.files.read_pdf_text(args["path"])}
            if name == "file_read_text":
                return {"text": self.files.read_text_auto(args["path"])}
            if name == "file_image_info":
                return self.files.image_info(args["path"])
            if name == "file_tree":
                return self.files.tree(args.get("path", "."), int(args.get("max_depth", 3)))
            if name == "file_grep":
                return {"results": self.files.grep(args["pattern"], args.get("path", "."), args.get("extensions"))}
            if name == "file_head":
                return {"text": self.files.head_file(args["path"], int(args.get("lines", 20)))}
            if name == "file_tail":
                return {"text": self.files.tail_file(args["path"], int(args.get("lines", 20)))}
            if name == "file_count_lines":
                return {"lines": self.files.count_lines(args["path"])}
            if name == "file_replace":
                return self.files.replace_in_file(args["path"], args["old"], args["new"], int(args.get("count", -1)))
            if name == "file_hash":
                return self.files.file_hash(args["path"], args.get("algorithm", "sha256"))
            if name == "file_exists":
                return {"exists": self.files.exists(args["path"])}
            if name == "file_backup":
                return {"backup": self.files.backup(args["path"])}
            if name == "file_image_resize":
                return {"path": self.files.image_resize(args["path"], args["output"], int(args["width"]), int(args["height"]) if args.get("height") else None, bool(args.get("keep_aspect", True)))}
            if name == "file_image_thumbnail":
                return {"path": self.files.image_thumbnail(args["path"], args["output"], int(args.get("size", 256)))}
            if name == "file_image_convert":
                return {"path": self.files.image_convert(args["path"], args["output"], args.get("format"))}
            if name == "file_read_json":
                return {"data": self.files.read_json(args["path"])}
            if name == "file_write_json":
                return {"path": self.files.write_json(args["path"], args["data"], int(args.get("indent", 2)))}
            if name == "file_read_csv":
                max_rows = args.get("max_rows")
                return {"rows": self.files.read_csv(args["path"], int(max_rows) if max_rows else None)}

            # --- GITHUB BÁSICO ---
            if name == "github_list":
                return await self.github_request("GET", "/user/repos?per_page=100")
            if name == "github_read":
                repo = args["repo"].strip("/")
                path = args["path"].lstrip("/")
                return await self.github_request("GET", f"/repos/{repo}/contents/{path}")
            if name == "github_write":
                return await self.github_write(args)

            # --- GITHUB AVANZADO ---
            if name == "github_upload_file":
                return await self.github_upload_file(args)
            if name == "github_upload_project":
                return await self.github_upload_project(args)
            if name == "github_create_repo":
                return await self.github_create_repo(args)
            if name == "github_create_issue":
                return await self.github_create_issue(args)
            if name == "github_list_issues":
                return await self.github_list_issues(args)
            if name == "github_create_pr":
                return await self.github_create_pr(args)
            if name == "github_list_prs":
                return await self.github_list_prs(args)
            if name == "github_merge_pr":
                return await self.github_merge_pr(args)
            if name == "github_list_branches":
                return await self.github_list_branches(args)
            if name == "github_delete_branch":
                return await self.github_delete_branch(args)
            if name == "github_get_commit":
                return await self.github_get_commit(args)
            if name == "github_list_commits":
                return await self.github_list_commits(args)
            if name == "github_get_file":
                return await self.github_get_file(args)
            if name == "github_update_file":
                return await self.github_update_file(args)
            if name == "github_delete_file":
                return await self.github_delete_file(args)

            # --- VERCEL BÁSICO ---
            if name == "vercel_projects":
                return await self.vercel_request("GET", "/v9/projects?limit=100")
            if name == "vercel_deployments":
                project = aiohttp.helpers.quote(args["project"], safe="")
                return await self.vercel_request("GET", f"/v6/deployments?projectId={project}&limit=20")

            # --- VERCEL AVANZADO ---
            if name == "vercel_project_info":
                return await self.vercel_project_info(args)
            if name == "vercel_list_envs":
                return await self.vercel_list_envs(args)
            if name == "vercel_set_env":
                return await self.vercel_set_env(args)
            if name == "vercel_delete_env":
                return await self.vercel_delete_env(args)
            if name == "vercel_redeploy":
                return await self.vercel_redeploy(args)
            if name == "vercel_build_logs":
                return await self.vercel_build_logs(args)
            if name == "vercel_list_deployments":
                return await self.vercel_list_deployments(args)
            if name == "vercel_get_deployment":
                return await self.vercel_get_deployment(args)
            if name == "vercel_cancel_deployment":
                return await self.vercel_cancel_deployment(args)
            if name == "vercel_list_domains":
                return await self.vercel_list_domains(args)
            if name == "vercel_add_domain":
                return await self.vercel_add_domain(args)
            if name == "vercel_remove_domain":
                return await self.vercel_remove_domain(args)
            if name == "vercel_list_aliases":
                return await self.vercel_list_aliases(args)
            if name == "vercel_delete_project":
                return await self.vercel_delete_project(args)
            if name == "vercel_create_project":
                return await self.vercel_create_project(args)

            # --- GENERACIÓN DE IMAGEN ---
            if name == "generate_image":
                return await self.generate_image(args["prompt"])

            return {"error": f"Herramienta desconocida: {name}"}

        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}

    # ==================================================================
    # WRAPPERS CON CACHE
    # ==================================================================

    async def _cached_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        max_results = int(args.get("max_results", 5))
        cached = motor.cache_get_search(f"{query}::{max_results}")
        if cached is not None:
            cached = dict(cached)
            cached["_cached"] = True
            return cached
        result = await self.files.web_search(query, max_results)
        if isinstance(result, dict) and result.get("results"):
            motor.cache_set_search(f"{query}::{max_results}", result)
        return result

    async def _cached_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        url = args.get("url", "")
        max_chars = int(args.get("max_chars", 8000))
        cached = motor.cache_get_fetch(f"{url}::{max_chars}")
        if cached is not None:
            cached = dict(cached)
            cached["_cached"] = True
            return cached
        result = await self.files.web_fetch(url, max_chars)
        if isinstance(result, dict) and result.get("text") and not result.get("error"):
            motor.cache_set_fetch(f"{url}::{max_chars}", result)
        return result

    # ==================================================================
    # GITHUB BÁSICO (ya existía)
    # ==================================================================

    async def github_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, "https://api.github.com" + path, headers=headers, **kwargs) as response:
                text = await response.text()
                if response.status >= 400:
                    return {"error": f"GitHub HTTP {response.status}", "detail": text[:3000]}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"content": text}

    async def github_write(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        content = base64.b64encode(args["content"].encode("utf-8")).decode("ascii")
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            sha = None
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    existing = await response.json(content_type=None)
                    sha = existing.get("sha")
                elif response.status not in {404, 301, 302}:
                    detail = await response.text()
                    return {"error": f"GitHub HTTP {response.status}", "detail": detail[:3000]}
            payload = {"message": args["message"], "content": content}
            if sha:
                payload["sha"] = sha
            async with session.put(url, headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"GitHub HTTP {response.status}", "detail": data}
                return data

    # ==================================================================
    # GITHUB AVANZADO
    # ==================================================================

    async def github_upload_file(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        repo_path = args["repo_path"].lstrip("/")
        local_path = args["local_path"]
        message = args["message"]
        try:
            full = self.files._path(local_path)
            if not full.is_file():
                return {"error": f"No existe: {local_path}"}
            raw = full.read_bytes()
        except Exception as exc:
            return {"error": f"Error leyendo {local_path}: {exc}"}
        b64 = base64.b64encode(raw).decode("ascii")
        url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            sha = None
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    existing = await resp.json(content_type=None)
                    sha = existing.get("sha")
            payload = {"message": message, "content": b64}
            if sha:
                payload["sha"] = sha
            async with session.put(url, headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"GitHub HTTP {resp.status}", "detail": data}
                return {"path": repo_path, "size": len(raw), "url": data.get("content", {}).get("html_url")}

    async def github_upload_project(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        files_map = args.get("files") or {}
        message = args["message"]
        branch = args.get("branch", "main")
        if not files_map:
            return {"error": "No hay archivos para subir."}
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        base = f"https://api.github.com/repos/{repo}"
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base}/git/refs/heads/{branch}", headers=headers) as resp:
                if resp.status == 404:
                    return {"error": f"La rama '{branch}' no existe. Crea el repo con un README primero."}
                ref = await resp.json(content_type=None)
                parent_sha = ref["object"]["sha"]
            async with session.get(f"{base}/git/commits/{parent_sha}", headers=headers) as resp:
                commit = await resp.json(content_type=None)
                parent_tree = commit["tree"]["sha"]
            tree_items = []
            uploaded = []
            for repo_path, ws_path in files_map.items():
                try:
                    full = self.files._path(ws_path)
                    if not full.is_file():
                        continue
                    raw = full.read_bytes()
                    b64 = base64.b64encode(raw).decode("ascii")
                except Exception as exc:
                    print(f"[GITHUB] skip {ws_path}: {exc}")
                    continue
                async with session.post(f"{base}/git/blobs", headers=headers, json={"content": b64, "encoding": "base64"}) as resp:
                    blob = await resp.json(content_type=None)
                    if resp.status >= 400:
                        return {"error": f"Blob falló: {blob}"}
                    sha_blob = blob["sha"]
                tree_items.append({"path": repo_path.lstrip("/"), "mode": "100644", "type": "blob", "sha": sha_blob})
                uploaded.append({"repo_path": repo_path, "size": len(raw)})
            if not tree_items:
                return {"error": "Ningún archivo válido para subir."}
            async with session.post(f"{base}/git/trees", headers=headers, json={"base_tree": parent_tree, "tree": tree_items}) as resp:
                tree = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"Tree falló: {tree}"}
                new_tree = tree["sha"]
            async with session.post(f"{base}/git/commits", headers=headers, json={"message": message, "tree": new_tree, "parents": [parent_sha]}) as resp:
                new_commit = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"Commit falló: {new_commit}"}
                commit_sha = new_commit["sha"]
            async with session.patch(f"{base}/git/refs/heads/{branch}", headers=headers, json={"sha": commit_sha}) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    return {"error": f"Ref falló: {detail[:500]}"}
            return {"repo": repo, "branch": branch, "commit": commit_sha, "files_uploaded": len(uploaded), "files": uploaded}

    async def github_create_repo(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        payload = {"name": args["name"], "description": args.get("description", ""), "private": bool(args.get("private", False)), "auto_init": True}
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.github.com/user/repos", headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"name": data.get("full_name"), "url": data.get("html_url"), "clone_url": data.get("clone_url")}

    async def github_create_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        payload = {"title": args["title"], "body": args.get("body", "")}
        if args.get("labels"):
            payload["labels"] = args["labels"]
        return await self.github_request("POST", f"/repos/{repo}/issues", json=payload)

    async def github_list_issues(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        state = args.get("state", "open")
        return await self.github_request("GET", f"/repos/{repo}/issues?state={state}&per_page=100")

    async def github_create_pr(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        payload = {"title": args["title"], "head": args["head"], "base": args.get("base", "main"), "body": args.get("body", "")}
        return await self.github_request("POST", f"/repos/{repo}/pulls", json=payload)

    async def github_list_prs(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        state = args.get("state", "open")
        return await self.github_request("GET", f"/repos/{repo}/pulls?state={state}&per_page=100")

    async def github_merge_pr(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        pr_number = args["pr_number"]
        payload = {"merge_method": args.get("merge_method", "merge")}
        return await self.github_request("PUT", f"/repos/{repo}/pulls/{pr_number}/merge", json=payload)

    async def github_list_branches(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        return await self.github_request("GET", f"/repos/{repo}/branches?per_page=100")

    async def github_delete_branch(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        branch = args["branch"]
        return await self.github_request("DELETE", f"/repos/{repo}/git/refs/heads/{branch}")

    async def github_get_commit(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        sha = args["sha"]
        return await self.github_request("GET", f"/repos/{repo}/commits/{sha}")

    async def github_list_commits(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        branch = args.get("branch", "main")
        per_page = int(args.get("per_page", 10))
        return await self.github_request("GET", f"/repos/{repo}/commits?sha={branch}&per_page={per_page}")

    async def github_get_file(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        ref = args.get("ref", "main")
        return await self.github_request("GET", f"/repos/{repo}/contents/{path}?ref={ref}")

    async def github_update_file(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        content = base64.b64encode(args["content"].encode("utf-8")).decode("ascii")
        payload = {"message": args["message"], "content": content, "sha": args["sha"]}
        return await self.github_request("PUT", f"/repos/{repo}/contents/{path}", json=payload)

    async def github_delete_file(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        payload = {"message": args["message"], "sha": args["sha"]}
        return await self.github_request("DELETE", f"/repos/{repo}/contents/{path}", json=payload)

    # ==================================================================
    # VERCEL BÁSICO (ya existía)
    # ==================================================================

    async def vercel_request(self, method: str, path: str) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no está configurado."}
        headers = {"Authorization": f"Bearer {config.vercel_token}"}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, "https://api.vercel.com" + path, headers=headers) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"Vercel HTTP {response.status}", "detail": data}
                return data

    # ==================================================================
    # VERCEL AVANZADO
    # ==================================================================

    async def _vercel_project_id(self, project: str) -> str | None:
        if project.startswith("prj_"):
            return project
        data = await self.vercel_request("GET", f"/v9/projects/{project}")
        if isinstance(data, dict) and data.get("id"):
            return data["id"]
        return None

    async def vercel_project_info(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        data = await self.vercel_request("GET", f"/v9/projects/{project}")
        if not isinstance(data, dict) or data.get("error"):
            return data
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "framework": data.get("framework"),
            "url": f"https://{data.get('name')}.vercel.app" if data.get("name") else None,
            "created": data.get("createdAt"),
            "updated": data.get("updatedAt"),
        }

    async def vercel_list_envs(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        data = await self.vercel_request("GET", f"/v9/projects/{pid}/env")
        if not isinstance(data, dict):
            return data
        envs = data.get("envs", [])
        return {
            "count": len(envs),
            "envs": [{"key": e.get("key"), "target": e.get("target"), "type": e.get("type"), "id": e.get("id")} for e in envs],
        }

    async def vercel_set_env(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        project = args["project"]
        key = args["key"]
        value = args["value"]
        target = args.get("target") or ["production", "preview", "development"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        headers = {
            "Authorization": f"Bearer {config.vercel_token}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            existing_id = None
            async with session.get(f"https://api.vercel.com/v9/projects/{pid}/env", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    for e in data.get("envs", []):
                        if e.get("key") == key:
                            existing_id = e.get("id")
                            break
            payload = {"key": key, "value": value, "target": target, "type": "encrypted"}
            if existing_id:
                async with session.patch(f"https://api.vercel.com/v9/projects/{pid}/env/{existing_id}", headers=headers, json={"value": value, "target": target}) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status >= 400:
                        return {"error": f"HTTP {resp.status}", "detail": data}
                    return {"action": "updated", "key": key, "target": target}
            else:
                async with session.post(f"https://api.vercel.com/v10/projects/{pid}/env", headers=headers, json=payload) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status >= 400:
                        return {"error": f"HTTP {resp.status}", "detail": data}
                    return {"action": "created", "key": key, "target": target}

    async def vercel_delete_env(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        project = args["project"]
        env_id = args["env_id"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("DELETE", f"/v9/projects/{pid}/env/{env_id}")

    async def vercel_redeploy(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        project = args["project"]
        target = args.get("target", "production")
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        headers = {
            "Authorization": f"Bearer {config.vercel_token}",
            "Content-Type": "application/json",
        }
        payload = {"name": project, "project": pid, "target": target}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.vercel.com/v13/deployments", headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"id": data.get("id"), "url": data.get("url"), "status": data.get("status"), "target": target}

    async def vercel_build_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        limit = int(args.get("limit", 100))
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        data = await self.vercel_request("GET", f"/v6/deployments?projectId={pid}&limit=1")
        deployments = data.get("deployments", []) if isinstance(data, dict) else []
        if not deployments:
            return {"error": "Sin deployments."}
        deploy = deployments[0]
        deploy_id = deploy.get("uid") or deploy.get("id")
        data = await self.vercel_request("GET", f"/v2/deployments/{deploy_id}/events?limit={limit}")
        if not isinstance(data, list):
            return {"deployment": deploy_id, "events": [], "error": "No se pudieron obtener eventos."}
        lines = []
        for ev in data:
            if ev.get("type") in ("stdout", "stderr"):
                text = ev.get("payload", {}).get("text", "")
                if text:
                    lines.append(text)
        return {"deployment": deploy_id, "state": deploy.get("state"), "url": deploy.get("url"), "logs": lines[-limit:], "log_count": len(lines)}

    async def vercel_list_deployments(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        limit = int(args.get("limit", 20))
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        data = await self.vercel_request("GET", f"/v6/deployments?projectId={pid}&limit={limit}")
        deployments = data.get("deployments", []) if isinstance(data, dict) else []
        return {"deployments": [{"id": d.get("uid") or d.get("id"), "url": d.get("url"), "state": d.get("state"), "created": d.get("created")} for d in deployments]}

    async def vercel_get_deployment(self, args: dict[str, Any]) -> dict[str, Any]:
        deployment_id = args["deployment_id"]
        return await self.vercel_request("GET", f"/v13/deployments/{deployment_id}")

    async def vercel_cancel_deployment(self, args: dict[str, Any]) -> dict[str, Any]:
        deployment_id = args["deployment_id"]
        return await self.vercel_request("PATCH", f"/v12/deployments/{deployment_id}/cancel")

    async def vercel_list_domains(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("GET", f"/v9/projects/{pid}/domains")

    async def vercel_add_domain(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        domain = args["domain"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("POST", f"/v10/projects/{pid}/domains", json={"name": domain})

    async def vercel_remove_domain(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        domain = args["domain"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("DELETE", f"/v9/projects/{pid}/domains/{domain}")

    async def vercel_list_aliases(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("GET", f"/v4/aliases?projectId={pid}")

    async def vercel_delete_project(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args["project"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        return await self.vercel_request("DELETE", f"/v9/projects/{pid}")

    async def vercel_create_project(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        name = args["name"]
        framework = args.get("framework", "nextjs")
        headers = {
            "Authorization": f"Bearer {config.vercel_token}",
            "Content-Type": "application/json",
        }
        payload = {"name": name, "framework": framework}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.vercel.com/v11/projects", headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"id": data.get("id"), "name": data.get("name"), "url": f"https://{data.get('name')}.vercel.app"}

    # ==================================================================
    # GENERACIÓN DE IMAGEN (FLUX)
    # ==================================================================

    async def generate_image(self, prompt: str) -> dict[str, Any]:
        if not prompt or not prompt.strip():
            return {"error": "prompt vacío"}
        if not self._flux_keys:
            return {"error": "No hay claves FLUX configuradas."}
        prompt = prompt.strip()
        key = self._flux_keys[0]
        self._flux_keys.rotate(-1)
        base = config.flux_base_url.rstrip("/")
        model = config.flux_endpoint
        headers = {"accept": "application/json", "x-key": key, "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{base}/{model}", headers=headers, json={"prompt": prompt, "width": 1024, "height": 1024, "output_format": "png"}) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"FLUX HTTP {response.status}", "detail": data}
                polling_url = data.get("polling_url")
            if not polling_url:
                return {"error": "FLUX no devolvió polling_url", "detail": data}
            for _ in range(60):
                await asyncio.sleep(1)
                async with session.get(polling_url, headers={"x-key": key}) as response:
                    result = await response.json(content_type=None)
                    if response.status >= 400:
                        return {"error": f"FLUX polling HTTP {response.status}", "detail": result}
                    status = str(result.get("status", "")).lower()
                    if status == "ready":
                        sample = result.get("result", {}).get("sample")
                        if not sample:
                            return {"result": result}
                        local_path = await self._download_flux_image(sample, prompt)
                        return {"image_url": sample, "local_path": local_path, "note": "Imagen guardada localmente. Usa github_upload_file con este local_path para subirla."}
                    if status in {"error", "failed", "request moderated", "content moderated"}:
                        return {"error": f"FLUX falló: {status}", "detail": result}
            return {"error": "FLUX tardó demasiado"}

    async def _download_flux_image(self, url: str, prompt: str) -> str | None:
        try:
            from datetime import datetime
            folder = Path(config.workspace) / "generated"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:30]
            filename = f"{stamp}_{slug}.png"
            target = folder / filename
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return None
                    data = await resp.read()
                    target.write_bytes(data)
            return str(target.relative_to(config.workspace))
        except Exception as exc:
            print(f"[FLUX] Error descargando imagen: {exc}")
            return None

    # ==================================================================
    # CHAT / TOOL LOOP
    # ==================================================================

    async def ask(self, user_id: int, channel_id: int, prompt: str, force_image: bool = False) -> tuple[str, str | None]:
        task = motor.classify(prompt)
        lang = motor.lang(prompt)
        print(f"[MOTOR] tarea={task.value} lang={lang.value}")

        await self.cleanup_memory()
        await self.save(user_id, channel_id, "user", prompt)

        messages = await self.history(user_id, channel_id)

        image_paths = self._extract_image_paths(prompt)
        if image_paths and messages:
            multimodal = self._build_multimodal_content(prompt, image_paths)
            if messages[-1].get("role") == "user":
                messages[-1] = {"role": "user", "content": multimodal}

        if force_image:
            messages.append({"role": "user", "content": "Genera una imagen usando la herramienta generate_image para esta petición:\n\n" + prompt})

        image_url: str | None = None
        modelos = self._select_model(prompt, task, force_image)
        last_error: Exception | None = None

        for modelo_actual in modelos:
            t0 = asyncio.get_event_loop().time()
            try:
                for _ in range(config.ai_max_tool_rounds):
                    response = await connector.complete(messages, self.tool_schemas(), model=modelo_actual)
                    choices = response.get("choices") or []
                    if not choices:
                        raise RuntimeError("La IA no devolvió ninguna elección.")
                    message = choices[0].get("message") or {}
                    tool_calls = message.get("tool_calls") or []
                    assistant_message = {"role": "assistant", "content": message.get("content") or ""}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    if not tool_calls:
                        answer = (message.get("content") or "").strip() or "No he recibido una respuesta de texto del modelo."
                        await self.save(user_id, channel_id, "assistant", answer)
                        dt = asyncio.get_event_loop().time() - t0
                        motor.record(model=modelo_actual, task=task, lang=lang, latency=dt, error=False)
                        return answer, image_url

                    for call in tool_calls:
                        function = call.get("function") or {}
                        name = function.get("name") or ""
                        raw_args = function.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                args = {}
                        elif isinstance(raw_args, dict):
                            args = raw_args
                        else:
                            args = {}
                        if not isinstance(args, dict):
                            args = {}
                        result = await self.run_tool(name, args)
                        if isinstance(result, dict) and result.get("image_url"):
                            image_url = result["image_url"]
                        messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": connector.clean_tool_result(result)})
                raise RuntimeError("La IA agotó el número máximo de rondas de herramientas.")
            except Exception as exc:
                last_error = exc
                dt = asyncio.get_event_loop().time() - t0
                motor.record(model=modelo_actual, task=task, lang=lang, latency=dt, error=True, error_msg=str(exc)[:200])
                err_txt = str(exc).lower()
                if "image" in err_txt or "vision" in err_txt or "multimodal" in err_txt or "modality" in err_txt:
                    print(f"[VISION] Modelo '{modelo_actual}' rechazó la imagen. Probando siguiente.")
                else:
                    print(f"[FALLBACK] Modelo '{modelo_actual}' falló: {type(exc).__name__}: {str(exc)[:200]}")
                continue

        raise RuntimeError("Todos los modelos gratuitos fallaron. Último error: " + str(last_error))


# Instancia global.
agent = Agent()
