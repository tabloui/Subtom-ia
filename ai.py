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


# Extensiones de imagen (para detección de visión).
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}


class Agent:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

        # Workspace interno de la IA.
        self.files = FileTools(config.workspace)

        # Dos claves FLUX con rotación automática.
        self._flux_keys = deque(
            key
            for key in (
                config.flux_api_key_1,
                config.flux_api_key_2,
            )
            if key
        )

    # ------------------------------------------------------------------
    # BASE DE DATOS / MEMORIA
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """
        Inicializa la conexión con Neon y crea una tabla propia.

        IMPORTANTE:
        No utilizamos la tabla antigua 'subtom_messages' porque puede
        existir con un esquema incompatible. Usamos una tabla nueva
        e independiente para evitar el UndefinedColumnError.
        """

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

    # ------------------------------------------------------------------
    # VISIÓN: DETECCIÓN E INYECCIÓN DE IMÁGENES
    # ------------------------------------------------------------------

    def _extract_image_paths(
        self,
        prompt: str,
    ) -> list[str]:
        """
        Detecta rutas de imágenes dentro del prompt.
        bot.py inserta líneas tipo:
            -> ruta local: /app/workspace/uploads/123_foto.png
        """

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

        # Evitar duplicados conservando el orden.
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
        """
        Construye el contenido multimodal (texto + imágenes)
        para mandarlo al modelo con visión.
        """

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for path in image_paths:

            try:

                info = self.files.read_image_base64(path)

                # Limitar tamaño para no reventar la API.
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

    # ------------------------------------------------------------------
    # DEFINICIÓN DE HERRAMIENTAS
    # ------------------------------------------------------------------

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [

            # ==========================================================
            # ARCHIVOS (básicos)
            # ==========================================================

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
                            "path": {
                                "type": "string",
                                "description": "Ruta del archivo.",
                            }
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
                            "query": {
                                "type": "string",
                                "description": (
                                    "Patrón de búsqueda de pathlib."
                                ),
                            },
                            "path": {"type": "string", "default": "."},
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

            # ==========================================================
            # ARCHIVOS (nuevos: análisis / lectura avanzada)
            # ==========================================================

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
                        "binario (info). Ideal para analizar "
                        "adjuntos de Discord."
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
                    "description": (
                        "Extrae el texto de un archivo PDF."
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
                    "description": (
                        "Redimensiona una imagen."
                    ),
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

            # ==========================================================
            # GITHUB
            # ==========================================================

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

            # ==========================================================
            # VERCEL
            # ==========================================================

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

            # ==========================================================
            # FLUX
            # ==========================================================

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

    # ------------------------------------------------------------------
    # EJECUCIÓN DE HERRAMIENTAS
    # ------------------------------------------------------------------

    async def run_tool(
        self,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:

        try:

            # ==========================================================
            # ARCHIVOS (básicos)
            # ==========================================================

            if name == "file_list":
                return {
                    "files": self.files.list_files(
                        args.get("path", ".")
                    )
                }

            if name == "file_read":
                return {
                    "content": self.files.read_file(
                        args["path"]
                    )
                }

            if name == "file_write":
                return {
                    "path": self.files.write_file(
                        args["path"],
                        args["content"],
                    )
                }

            if name == "file_append":
                return {
                    "path": self.files.append_file(
                        args["path"],
                        args["content"],
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
                return self.files.info(
                    args["path"]
                )

            if name == "file_mkdir":
                return {
                    "path": self.files.mkdir(
                        args["path"]
                    )
                }

            if name == "file_copy":
                return {
                    "path": self.files.copy(
                        args["source"],
                        args["destination"],
                    )
                }

            if name == "file_move":
                return {
                    "path": self.files.move(
                        args["source"],
                        args["destination"],
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
                        args["output_zip"],
                        args["sources"],
                    )
                }

            if name == "list_zip":
                return {
                    "files": self.files.zip_list(
                        args["zip_name"]
                    )
                }

            if name == "unzip_file":
                return {
                    "extracted": self.files.zip_extract(
                        args["zip_name"],
                        args.get("destination", "."),
                    )
                }

            # ==========================================================
            # ARCHIVOS (nuevos: análisis / lectura avanzada)
            # ==========================================================

            if name == "file_detect_kind":
                return {
                    "kind": self.files.detect_kind(
                        args["path"]
                    )
                }

            if name == "file_read_any":
                data = self.files.read_any(args["path"])

                # Evitar devolver el base64 completo como texto.
                if isinstance(data, dict):
                    data.pop("base64", None)
                    data.pop("data_url", None)

                return data

            if name == "file_read_pdf":
                return {
                    "text": self.files.read_pdf_text(
                        args["path"]
                    )
                }

            if name == "file_read_text":
                return {
                    "text": self.files.read_text_auto(
                        args["path"]
                    )
                }

            if name == "file_image_info":
                return self.files.image_info(
                    args["path"]
                )

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
                return {
                    "lines": self.files.count_lines(
                        args["path"]
                    )
                }

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
                return {
                    "exists": self.files.exists(
                        args["path"]
                    )
                }

            if name == "file_backup":
                return {
                    "backup": self.files.backup(
                        args["path"]
                    )
                }

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
                return {
                    "data": self.files.read_json(
                        args["path"]
                    )
                }

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

            # ==========================================================
            # GITHUB
            # ==========================================================

            if name == "github_list":
                return await self.github_request(
                    "GET",
                    "/user/repos?per_page=100",
                )

            if name == "github_read":
                repo = args["repo"].strip("/")
                path = args["path"].lstrip("/")

                return await self.github_request(
                    "GET",
                    f"/repos/{repo}/contents/{path}",
                )

            if name == "github_write":
                return await self.github_write(args)

            # ==========================================================
            # VERCEL
            # ==========================================================

            if name == "vercel_projects":
                return await self.vercel_request(
                    "GET",
                    "/v9/projects?limit=100",
                )

            if name == "vercel_deployments":
                project = aiohttp.helpers.quote(
                    args["project"],
                    safe="",
                )

                return await self.vercel_request(
                    "GET",
                    f"/v6/deployments?projectId={project}&limit=20",
                )

            # ==========================================================
            # FLUX
            # ==========================================================

            if name == "generate_image":
                return await self.generate_image(
                    args["prompt"]
                )

            return {
                "error": f"Herramienta desconocida: {name}"
            }

        except Exception as exc:
            return {
                "error": (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:2000]}"
                )
            }

    # ------------------------------------------------------------------
    # GITHUB
    # ------------------------------------------------------------------

    async def github_request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        if not config.github_token:
            return {
                "error": (
                    "GITHUB_TOKEN no está configurado."
                )
            }

        headers = {
            "Authorization": (
                f"Bearer {config.github_token}"
            ),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.request(
                method,
                "https://api.github.com" + path,
                headers=headers,
                **kwargs,
            ) as response:

                text = await response.text()

                if response.status >= 400:
                    return {
                        "error": (
                            f"GitHub HTTP "
                            f"{response.status}"
                        ),
                        "detail": text[:3000],
                    }

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {
                        "content": text
                    }

    async def github_write(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:

        if not config.github_token:
            return {
                "error": (
                    "GITHUB_TOKEN no está configurado."
                )
            }

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
            "Authorization": (
                f"Bearer {config.github_token}"
            ),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            sha = None

            async with session.get(
                url,
                headers=headers,
            ) as response:

                if response.status == 200:
                    existing = await response.json(
                        content_type=None
                    )
                    sha = existing.get("sha")

                elif response.status not in {
                    404,
                    301,
                    302,
                }:
                    detail = await response.text()
                    return {
                        "error": (
                            f"GitHub HTTP "
                            f"{response.status}"
                        ),
                        "detail": detail[:3000],
                    }

            payload: dict[str, Any] = {
                "message": args["message"],
                "content": content,
            }

            if sha:
                payload["sha"] = sha

            async with session.put(
                url,
                headers=headers,
                json=payload,
            ) as response:

                data = await response.json(
                    content_type=None
                )

                if response.status >= 400:
                    return {
                        "error": (
                            f"GitHub HTTP "
                            f"{response.status}"
                        ),
                        "detail": data,
                    }

                return data

    # ------------------------------------------------------------------
    # VERCEL
    # ------------------------------------------------------------------

    async def vercel_request(
        self,
        method: str,
        path: str,
    ) -> dict[str, Any]:

        if not config.vercel_token:
            return {
                "error": (
                    "VERCEL_TOKEN no está configurado."
                )
            }

        headers = {
            "Authorization": (
                f"Bearer {config.vercel_token}"
            )
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.request(
                method,
                "https://api.vercel.com" + path,
                headers=headers,
            ) as response:

                data = await response.json(
                    content_type=None
                )

                if response.status >= 400:
                    return {
                        "error": (
                            f"Vercel HTTP "
                            f"{response.status}"
                        ),
                        "detail": data,
                    }

                return data

    # ------------------------------------------------------------------
    # FLUX
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
    ) -> dict[str, Any]:

        if not self._flux_keys:
            return {
                "error": (
                    "No hay ninguna clave FLUX configurada."
                )
            }

        key = self._flux_keys[0]

        self._flux_keys.rotate(-1)

        base = config.flux_base_url.rstrip("/")

        headers = {
            "accept": "application/json",
            "x-key": key,
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.post(
                f"{base}/{config.flux_endpoint}",
                headers=headers,
                json={"prompt": prompt},
            ) as response:

                data = await response.json(
                    content_type=None
                )

                if response.status >= 400:
                    return {
                        "error": (
                            f"FLUX HTTP "
                            f"{response.status}"
                        ),
                        "detail": data,
                    }

                task_id = (
                    data.get("id")
                    or data.get("task_id")
                )

            if not task_id:
                return {"result": data}

            for _ in range(60):

                await asyncio.sleep(2)

                async with session.get(
                    f"{base}/get_result?id={task_id}",
                    headers=headers,
                ) as response:

                    result = await response.json(
                        content_type=None
                    )

                    if response.status >= 400:
                        return {
                            "error": (
                                "FLUX polling HTTP "
                                f"{response.status}"
                            ),
                            "detail": result,
                        }

                    status = str(
                        result.get("status", "")
                    ).lower()

                    if status in {
                        "ready",
                        "succeeded",
                        "completed",
                    }:

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
                            "error": (
                                "La generación de "
                                "imagen falló."
                            ),
                            "detail": result,
                        }

            return {
                "error": (
                    "La generación de imagen "
                    "tardó demasiado."
                )
            }

    # ------------------------------------------------------------------
    # CHAT / TOOL LOOP
    # ------------------------------------------------------------------

    async def ask(
        self,
        user_id: int,
        channel_id: int,
        prompt: str,
        force_image: bool = False,
    ) -> tuple[str, str | None]:

        await self.cleanup_memory()

        await self.save(
            user_id,
            channel_id,
            "user",
            prompt,
        )

        messages = await self.history(
            user_id,
            channel_id,
        )

        # ----------------------------------------------------------
        # VISIÓN: si el prompt trae rutas de imágenes,
        # las inyectamos como contenido multimodal.
        # ----------------------------------------------------------

        image_paths = self._extract_image_paths(prompt)

        if image_paths and messages:

            multimodal = self._build_multimodal_content(
                prompt,
                image_paths,
            )

            # El último mensaje es el que acabamos de guardar.
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
                        "para esta petición:\n\n"
                        + prompt
                    ),
                }
            )

        image_url: str | None = None

        for _ in range(
            config.ai_max_tool_rounds
        ):

            response = await connector.complete(
                messages,
                self.tool_schemas(),
            )

            choices = (
                response.get("choices")
                or []
            )

            if not choices:
                raise RuntimeError(
                    "La IA no devolvió ninguna elección."
                )

            message = (
                choices[0].get("message")
                or {}
            )

            tool_calls = (
                message.get("tool_calls")
                or []
            )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": (
                    message.get("content")
                    or ""
                ),
            }

            if tool_calls:
                assistant_message[
                    "tool_calls"
                ] = tool_calls

            messages.append(
                assistant_message
            )

            # ----------------------------------------------------------
            # RESPUESTA FINAL
            # ----------------------------------------------------------

            if not tool_calls:

                answer = (
                    message.get("content")
                    or ""
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

            # ----------------------------------------------------------
            # EJECUTAR TOOLS
            # ----------------------------------------------------------

            for call in tool_calls:

                function = (
                    call.get("function")
                    or {}
                )

                name = (
                    function.get("name")
                    or ""
                )

                raw_args = function.get(
                    "arguments",
                    "{}",
                )

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

                result = await self.run_tool(
                    name,
                    args,
                )

                if (
                    isinstance(result, dict)
                    and result.get("image_url")
                ):
                    image_url = result["image_url"]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            call.get("id")
                            or ""
                        ),
                        "content": (
                            connector.clean_tool_result(
                                result
                            )
                        ),
                    }
                )

        raise RuntimeError(
            "La IA agotó el número máximo "
            "de rondas de herramientas."
        )


# Instancia global.
agent = Agent()
