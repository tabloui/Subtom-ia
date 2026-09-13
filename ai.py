from __future__ import annotations

import asyncio
import base64
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg

from config import config
from connector import connector
from file_tools import FileTools


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}


class Agent:

    # ==================================================================
    # LISTAS DE MODELOS :free POR ESPECIALIDAD
    # ==================================================================

    VISION_MODELS = [
        "inclusionai/ling-3.0-flash-vl:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "minimax/minimax-m3:free",
        "thinkingmachines/inkling-small:free",
        "thinkingmachines/inkling:free",
        "dots-studio/dots-3-note-preview:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "qwen/qwen2.5-vl-7b-instruct:free",
        "qwen/qwen2.5-vl-3b-instruct:free",
        "qwen/qwen3.7-flash:free",
        "moonshotai/kimi-vl-a3b-thinking:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        "nvidia/nemotron-3.5-content-safety:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "qwen/qwen2-vl-72b-instruct:free",
        "qwen/qwen2-vl-7b-instruct:free",
        "qwen/qwen2-vl-2b-instruct:free",
        "microsoft/phi-3.5-vision-instruct:free",
        "llava-hf/llava-1.5-7b-hf:free",
        "OpenGVLab/InternVL2-8B:free",
        "OpenGVLab/InternVL2-26B:free",
        "google/paligemma-3b-pt-224:free",
    ]

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

    async def save(
        self,
        user_id: int,
        channel_id: int,
        role: str,
        content: str,
    ) -> None:
        if not self.pool:
            return

        await self.pool.execute(
            """
            INSERT INTO subtom_memory(
                user_id,
                channel_id,
                role,
                content
            )
            VALUES($1, $2, $3, $4)
            """,
            user_id,
            channel_id,
            role,
            content[:30000],
        )

    async def history(
        self,
        user_id: int,
        channel_id: int,
    ) -> list[dict[str, Any]]:

        if not self.pool:
            return []

        rows = await self.pool.fetch(
            """
            SELECT role, content
            FROM subtom_memory
            WHERE user_id = $1
              AND channel_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id,
            channel_id,
            config.memory_messages,
        )

        rows = list(reversed(rows))

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]

    # ==================================================================
    # VISIÓN
    # ==================================================================

    def _extract_image_paths(
        self,
        prompt: str,
    ) -> list[str]:

        paths: list[str] = []

        for match in re.finditer(
            r"->\s*ruta local:\s*(\S+)",
            prompt,
        ):
            raw = match.group(1).strip().strip('",')

            try:
                p = Path(raw)

                if (
                    p.suffix.lower() in IMAGE_EXTENSIONS
                    and p.exists()
                    and p.is_file()
                ):
                    paths.append(str(p))

            except Exception:
                continue

        seen: set[str] = set()
        unique: list[str] = []

        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique

    def _build_multimodal_content(
        self,
        prompt: str,
        image_paths: list[str],
    ) -> list[dict[str, Any]]:

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for path in image_paths:

            try:

                info = self.files.read_image_base64(path)

                if info["size"] > 20 * 1024 * 1024:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                f"[Imagen omitida por tamaño "
                                f"> 20MB: {info['filename']}]"
                            ),
                        }
                    )
                    continue

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": info["data_url"],
                        },
                    }
                )

            except Exception as exc:

                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[No se pudo cargar imagen "
                            f"{path}: {exc}]"
                        ),
                    }
                )

        return content

    # ==================================================================
    # SELECCIÓN DE MODELO SEGÚN LA TAREA
    # ==================================================================

    def _select_model(
        self,
        prompt: str,
        force_image: bool = False,
    ) -> list[str]:
        """
        Devuelve la lista ordenada de modelos a intentar.
        El primero es el preferido; los demás son fallback.
        """

        if force_image:
            return self.VISION_MODELS

        if self._extract_image_paths(prompt):
            return self.VISION_MODELS

        code_keywords = [
            "código", "code", "programa", "script",
            "función", "function", "deploy", "github",
            "error", "bug", "debug", "python",
            "javascript", "html", "css", "api",
            "servidor", "server", "base de datos",
            "database", "commit", "push", "repositorio",
            "repo", "vercel", "railway",
        ]

        if any(kw in prompt.lower() for kw in code_keywords):
            return self.CODE_MODELS

        return self.TEXT_MODELS

    # ==================================================================
    # DEFINICIÓN DE HERRAMIENTAS
    # ==================================================================

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [

            {
                "type": "function",
                "function": {
                    "name": "file_list",
                    "description": (
                        "Lista archivos y carpetas dentro del workspace "
                        "seguro de Subtom."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Ruta relativa dentro del workspace."
                                ),
                                "default": ".",
                            }
                        },
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": (
                        "Lee un archivo de texto dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_write",
                    "description": (
                        "Crea o reemplaza completamente un archivo "
                        "de texto dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_append",
                    "description": (
                        "Añade contenido al final de un archivo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_search",
                    "description": (
                        "Busca archivos y carpetas mediante un patrón "
                        "dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path": {
                                "type": "string",
                                "default": ".",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "description": (
                        "Obtiene información sobre un archivo o carpeta."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
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
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_copy",
                    "description": (
                        "Copia un archivo o carpeta dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "destination"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_move",
                    "description": (
                        "Mueve o renombra un archivo o carpeta."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "destination"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_delete",
                    "description": (
                        "Elimina un archivo o carpeta. "
                        "Debe usarse con cuidado."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {
                                "type": "boolean",
                                "default": False,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "create_zip",
                    "description": (
                        "Crea un archivo ZIP dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "output_zip": {"type": "string"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
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
                        "properties": {
                            "zip_name": {"type": "string"}
                        },
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
                        "properties": {
                            "zip_name": {"type": "string"},
                            "destination": {
                                "type": "string",
                                "default": ".",
                            },
                        },
                        "required": ["zip_name"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_detect_kind",
                    "description": (
                        "Detecta el tipo real de un archivo: "
                        "image, pdf, text, binary o directory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_read_any",
                    "description": (
                        "Lee cualquier archivo y devuelve algo útil: "
                        "imagen (info), pdf (texto), texto (contenido), "
                        "binario (info)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
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
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_read_text",
                    "description": (
                        "Lee un archivo de texto detectando encoding "
                        "automáticamente."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_image_info",
                    "description": (
                        "Devuelve información de una imagen: "
                        "ancho, alto, formato, modo y tamaño."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_tree",
                    "description": (
                        "Devuelve un árbol recursivo del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "default": ".",
                            },
                            "max_depth": {
                                "type": "integer",
                                "default": 3,
                            },
                        },
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_grep",
                    "description": (
                        "Busca texto dentro de archivos del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {
                                "type": "string",
                                "default": ".",
                            },
                            "extensions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_head",
                    "description": (
                        "Devuelve las primeras N líneas de un archivo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "lines": {
                                "type": "integer",
                                "default": 20,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_tail",
                    "description": (
                        "Devuelve las últimas N líneas de un archivo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "lines": {
                                "type": "integer",
                                "default": 20,
                            },
                        },
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
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_replace",
                    "description": (
                        "Reemplaza texto dentro de un archivo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "count": {
                                "type": "integer",
                                "default": -1,
                            },
                        },
                        "required": ["path", "old", "new"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_hash",
                    "description": (
                        "Devuelve el hash (sha256 por defecto) "
                        "de un archivo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "algorithm": {
                                "type": "string",
                                "default": "sha256",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_exists",
                    "description": (
                        "Comprueba si una ruta existe dentro "
                        "del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_backup",
                    "description": (
                        "Crea una copia de seguridad con timestamp "
                        "de un archivo o carpeta. Úsalo SIEMPRE antes "
                        "de modificar tu propio código."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
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
                        "properties": {
                            "path": {"type": "string"},
                            "output": {"type": "string"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                            "keep_aspect": {
                                "type": "boolean",
                                "default": True,
                            },
                        },
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
                        "properties": {
                            "path": {"type": "string"},
                            "output": {"type": "string"},
                            "size": {
                                "type": "integer",
                                "default": 256,
                            },
                        },
                        "required": ["path", "output"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_image_convert",
                    "description": (
                        "Convierte una imagen a otro formato."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "output": {"type": "string"},
                            "format": {"type": "string"},
                        },
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
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_write_json",
                    "description": (
                        "Escribe un archivo JSON a partir de un objeto."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "data": {},
                            "indent": {
                                "type": "integer",
                                "default": 2,
                            },
                        },
                        "required": ["path", "data"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "file_read_csv",
                    "description": (
                        "Lee un CSV y devuelve las filas como dicts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_rows": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "github_list",
                    "description": (
                        "Lista los repositorios del usuario autenticado "
                        "en GitHub."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "github_read",
                    "description": (
                        "Lee un archivo de un repositorio GitHub."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {
                                "type": "string",
                                "description": "owner/repo",
                            },
                            "path": {"type": "string"},
                        },
                        "required": ["repo", "path"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "github_write",
                    "description": (
                        "Crea o actualiza un archivo en GitHub."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": [
                            "repo",
                            "path",
                            "content",
                            "message",
                        ],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "vercel_projects",
                    "description": "Lista proyectos de Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "vercel_deployments",
                    "description": (
                        "Lista deployments de un proyecto Vercel."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project": {"type": "string"}
                        },
                        "required": ["project"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": (
                        "Genera una imagen mediante FLUX usando una "
                        "de las claves configuradas."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"}
                        },
                        "required": ["prompt"],
                    },
                },
            },
        ]

    # ==================================================================
    # EJECUCIÓN DE HERRAMIENTAS
    # ==================================================================

    async def run_tool(
        self,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:

        try:

            if name == "file_list":
                return {
                    "files": self.files.list_files(
                        args.get("path", ".")
                    )
                }

            if name == "file_read":
                return {
                    "content": self.files.read_file(args["path"])
                }

            if name == "file_write":
                return {
                    "path": self.files.write_file(
                        args["path"], args["content"]
                    )
                }

            if name == "file_append":
                return {
                    "path": self.files.append_file(
                        args["path"], args["content"]
                    )
                }

            if name == "file_search":
                return {
                    "matches": self.files.search(
                        args["query"],
                        args.get("path", "."),
                    )
                }

            if name == "file_info":
                return self.files.info(args["path"])

            if name == "file_mkdir":
                return {"path": self.files.mkdir(args["path"])}

            if name == "file_copy":
                return {
                    "path": self.files.copy(
                        args["source"], args["destination"]
                    )
                }

            if name == "file_move":
                return {
                    "path": self.files.move(
                        args["source"], args["destination"]
                    )
                }

            if name == "file_delete":
                return {
                    "deleted": self.files.delete(
                        args["path"],
                        args.get("recursive", False),
                    )
                }

            if name == "create_zip":
                return {
                    "path": self.files.zip_create(
                        args["output_zip"], args["sources"]
                    )
                }

            if name == "list_zip":
                return {
                    "files": self.files.zip_list(args["zip_name"])
                }

            if name == "unzip_file":
                return {
                    "extracted": self.files.zip_extract(
                        args["zip_name"],
                        args.get("destination", "."),
                    )
                }

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
                return self.files.tree(
                    args.get("path", "."),
                    int(args.get("max_depth", 3)),
                )

            if name == "file_grep":
                return {
                    "results": self.files.grep(
                        args["pattern"],
                        args.get("path", "."),
                        args.get("extensions"),
                    )
                }

            if name == "file_head":
                return {
                    "text": self.files.head_file(
                        args["path"],
                        int(args.get("lines", 20)),
                    )
                }

            if name == "file_tail":
                return {
                    "text": self.files.tail_file(
                        args["path"],
                        int(args.get("lines", 20)),
                    )
                }

            if name == "file_count_lines":
                return {"lines": self.files.count_lines(args["path"])}

            if name == "file_replace":
                return self.files.replace_in_file(
                    args["path"],
                    args["old"],
                    args["new"],
                    int(args.get("count", -1)),
                )

            if name == "file_hash":
                return self.files.file_hash(
                    args["path"],
                    args.get("algorithm", "sha256"),
                )

            if name == "file_exists":
                return {"exists": self.files.exists(args["path"])}

            if name == "file_backup":
                return {"backup": self.files.backup(args["path"])}

            if name == "file_image_resize":
                return {
                    "path": self.files.image_resize(
                        args["path"],
                        args["output"],
                        int(args["width"]),
                        (
                            int(args["height"])
                            if args.get("height")
                            else None
                        ),
                        bool(args.get("keep_aspect", True)),
                    )
                }

            if name == "file_image_thumbnail":
                return {
                    "path": self.files.image_thumbnail(
                        args["path"],
                        args["output"],
                        int(args.get("size", 256)),
                    )
                }

            if name == "file_image_convert":
                return {
                    "path": self.files.image_convert(
                        args["path"],
                        args["output"],
                        args.get("format"),
                    )
                }

            if name == "file_read_json":
                return {"data": self.files.read_json(args["path"])}

            if name == "file_write_json":
                return {
                    "path": self.files.write_json(
                        args["path"],
                        args["data"],
                        int(args.get("indent", 2)),
                    )
                }

            if name == "file_read_csv":
                max_rows = args.get("max_rows")
                return {
                    "rows": self.files.read_csv(
                        args["path"],
                        int(max_rows) if max_rows else None,
                    )
                }

            if name == "github_list":
                return await self.github_request(
                    "GET", "/user/repos?per_page=100"
                )

            if name == "github_read":
                repo = args["repo"].strip("/")
                path = args["path"].lstrip("/")
                return await self.github_request(
                    "GET", f"/repos/{repo}/contents/{path}"
                )

            if name == "github_write":
                return await self.github_write(args)

            if name == "vercel_projects":
                return await self.vercel_request(
                    "GET", "/v9/projects?limit=100"
                )

            if name == "vercel_deployments":
                project = aiohttp.helpers.quote(
                    args["project"], safe=""
                )
                return await self.vercel_request(
                    "GET",
                    f"/v6/deployments?projectId={project}&limit=20",
                )

            if name == "generate_image":
                return await self.generate_image(args["prompt"])

            return {"error": f"Herramienta desconocida: {name}"}

        except Exception as exc:
            return {
                "error": (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:2000]}"
                )
            }

    # ==================================================================
    # GITHUB
    # ==================================================================

    async def github_request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}

        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                "https://api.github.com" + path,
                headers=headers,
                **kwargs,
            ) as response:

                text = await response.text()

                if response.status >= 400:
                    return {
                        "error": f"GitHub HTTP {response.status}",
                        "detail": text[:3000],
                    }

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"content": text}

    async def github_write(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:

        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}

        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")

        content = base64.b64encode(
            args["content"].encode("utf-8")
        ).decode("ascii")

        url = (
            f"https://api.github.com/repos/"
            f"{repo}/contents/{path}"
        )

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
                    return {
                        "error": f"GitHub HTTP {response.status}",
                        "detail": detail[:3000],
                    }

            payload: dict[str, Any] = {
                "message": args["message"],
                "content": content,
            }

            if sha:
                payload["sha"] = sha

            async with session.put(
                url, headers=headers, json=payload
            ) as response:

                data = await response.json(content_type=None)

                if response.status >= 400:
                    return {
                        "error": f"GitHub HTTP {response.status}",
                        "detail": data,
                    }

                return data

    # ==================================================================
    # VERCEL
    # ==================================================================

    async def vercel_request(
        self,
        method: str,
        path: str,
    ) -> dict[str, Any]:

        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no está configurado."}

        headers = {
            "Authorization": f"Bearer {config.vercel_token}"
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                "https://api.vercel.com" + path,
                headers=headers,
            ) as response:

                data = await response.json(content_type=None)

                if response.status >= 400:
                    return {
                        "error": f"Vercel HTTP {response.status}",
                        "detail": data,
                    }

                return data

    # ==================================================================
    # FLUX
    # ==================================================================

    async def generate_image(
        self,
        prompt: str,
    ) -> dict[str, Any]:

        if not self._flux_keys:
            return {"error": "No hay ninguna clave FLUX configurada."}

        key = self._flux_keys[0]
        self._flux_keys.rotate(-1)

        base = config.flux_base_url.rstrip("/")

        headers = {
            "accept": "application/json",
            "x-key": key,
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:

            async with session.post(
                f"{base}/{config.flux_endpoint}",
                headers=headers,
                json={"prompt": prompt},
            ) as response:

                data = await response.json(content_type=None)

                if response.status >= 400:
                    return {
                        "error": f"FLUX HTTP {response.status}",
                        "detail": data,
                    }

                task_id = data.get("id") or data.get("task_id")

            if not task_id:
                return {"result": data}

            for _ in range(60):

                await asyncio.sleep(2)

                async with session.get(
                    f"{base}/get_result?id={task_id}",
                    headers=headers,
                ) as response:

                    result = await response.json(content_type=None)

                    if response.status >= 400:
                        return {
                            "error": (
                                "FLUX polling HTTP "
                                f"{response.status}"
                            ),
                            "detail": result,
                        }

                    status = str(result.get("status", "")).lower()

                    if status in {"ready", "succeeded", "completed"}:
                        nested = result.get("result")
                        sample = None

                        if isinstance(nested, dict):
                            sample = nested.get("sample")

                        if not sample:
                            sample = result.get("sample")

                        if sample:
                            return {"image_url": sample}

                        return {"result": result}

                    if status in {"failed", "error"}:
                        return {
                            "error": "La generación de imagen falló.",
                            "detail": result,
                        }

            return {"error": "La generación de imagen tardó demasiado."}

    # ==================================================================
    # CHAT / TOOL LOOP CON FALLBACK DE MODELOS
    # ==================================================================

    async def ask(
        self,
        user_id: int,
        channel_id: int,
        prompt: str,
        force_image: bool = False,
    ) -> tuple[str, str | None]:

        await self.cleanup_memory()

        await self.save(user_id, channel_id, "user", prompt)

        messages = await self.history(user_id, channel_id)

        # --- Imágenes ---
        image_paths = self._extract_image_paths(prompt)

        if image_paths and messages:
            multimodal = self._build_multimodal_content(
                prompt, image_paths
            )
            if messages[-1].get("role") == "user":
                messages[-1] = {
                    "role": "user",
                    "content": multimodal,
                }

        if force_image:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Genera una imagen usando la "
                        "herramienta generate_image "
                        "para esta petición:\n\n" + prompt
                    ),
                }
            )

        image_url: str | None = None

        # --- Modelos a intentar (fallback) ---
        modelos = self._select_model(prompt, force_image)

        last_error: Exception | None = None

        for modelo_actual in modelos:

            try:

                for _ in range(config.ai_max_tool_rounds):

                    response = await connector.complete(
                        messages,
                        self.tool_schemas(),
                        model=modelo_actual,
                    )

                    choices = response.get("choices") or []

                    if not choices:
                        raise RuntimeError(
                            "La IA no devolvió ninguna elección."
                        )

                    message = choices[0].get("message") or {}
                    tool_calls = message.get("tool_calls") or []

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": message.get("content") or "",
                    }

                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls

                    messages.append(assistant_message)

                    # --- Respuesta final ---
                    if not tool_calls:

                        answer = (
                            message.get("content") or ""
                        ).strip()

                        if not answer:
                            answer = (
                                "No he recibido una respuesta "
                                "de texto del modelo."
                            )

                        await self.save(
                            user_id,
                            channel_id,
                            "assistant",
                            answer,
                        )

                        return answer, image_url

                    # --- Tools ---
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

                        if (
                            isinstance(result, dict)
                            and result.get("image_url")
                        ):
                            image_url = result["image_url"]

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id") or "",
                                "content": (
                                    connector.clean_tool_result(result)
                                ),
                            }
                        )

                raise RuntimeError(
                    "La IA agotó el número máximo "
                    "de rondas de herramientas."
                )

            except Exception as exc:

                last_error = exc

                print(
                    f"[FALLBACK] Modelo '{modelo_actual}' "
                    f"falló: {type(exc).__name__}: "
                    f"{str(exc)[:200]}. Probando siguiente..."
                )

                continue

        raise RuntimeError(
            "Todos los modelos gratuitos fallaron. "
            f"Último error: {last_error}"
        )


# Instancia global.
agent = Agent()
