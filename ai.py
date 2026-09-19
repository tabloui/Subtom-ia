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


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}

REPO_PRINCIPAL = "tabloui/subtom-ia"

MAX_LINES_PER_TOOL_CALL = 50

# Modelos de Pollinations en orden de preferencia (si uno falla, prueba el siguiente)
POLLINATIONS_MODELS = ["flux", "turbo", "flux-realism", "flux-anime"]


class Agent:

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
        self.files = FileTools(config.workspace)

    async def init(self) -> None:
        self.pool = await asyncpg.create_pool(
            config.database_url, min_size=1, max_size=3, command_timeout=60,
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
        try:
            await motor.prewarm()
        except Exception as exc:
            print(f"[MOTOR] prewarm falló: {exc}")

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
            VALUES($1, $2, $3, $4)
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
            raw = m.group(1).strip().strip('",')
            try:
                p = Path(raw)
                if (p.suffix.lower() in IMAGE_EXTENSIONS
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
        modelos = []
        for slot in config.ai_slots:
            if slot.model and slot.model not in modelos:
                modelos.append(slot.model)
        return modelos or [config.ai_model]

    def _extract_tool_calls(self, message: dict) -> list[dict]:
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return tool_calls
        parts = message.get("parts") or []
        converted = []
        for p in parts:
            fc = p.get("functionCall")
            if fc:
                converted.append({
                    "id": f"call_{hashlib.blake2b(json.dumps(fc, sort_keys=True).encode(), digest_size=8).hexdigest()}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                    },
                })
        return converted

    # ================================================================
    # EXTRACCIÓN DE BLOQUES DE CÓDIGO (OPCIÓN 3)
    # ================================================================

    # Regex: detecta "FILE: nombre.ext" seguido de un bloque ```lang ... ```
    _FILE_BLOCK_RE = re.compile(
        r"FILE:\s*([^\n`]+?)\s*\n"
        r"```[a-zA-Z0-9_+\-]*[ \t]*\n"
        r"(.*?)"
        r"\n```",
        re.DOTALL | re.IGNORECASE,
    )

    def _extract_file_blocks(self, text: str) -> list[dict[str, str]]:
        if not text:
            return []
        blocks = []
        for match in self._FILE_BLOCK_RE.finditer(text):
            filename = match.group(1).strip().strip("`").strip()
            content = match.group(2)
            if filename and content:
                blocks.append({
                    "filename": filename,
                    "content": content.rstrip() + "\n",
                })
        return blocks

    async def _process_file_blocks(self, text: str) -> list[str]:
        """
        Extrae bloques FILE: del texto, los guarda en el workspace,
        y los sube a GitHub con github_upload_project.
        Devuelve la lista de archivos subidos.
        """
        blocks = self._extract_file_blocks(text)
        if not blocks:
            return []

        uploaded_files: list[str] = []
        files_map: dict[str, str] = {}

        for block in blocks:
            filename = block["filename"]
            content = block["content"]

            safe_name = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
            pending_name = f"pending_{safe_name}"
            try:
                self.files.write_file(pending_name, content)
                print(f"[FILE_BLOCK] Guardado {pending_name} ({len(content)} chars)")
            except Exception as exc:
                print(f"[FILE_BLOCK] Error guardando {pending_name}: {exc}")
                continue

            files_map[filename] = pending_name
            uploaded_files.append(filename)

        if not files_map:
            return []

        try:
            result = await self.github_upload_project({
                "repo": REPO_PRINCIPAL,
                "files": files_map,
                "message": f"Update {', '.join(uploaded_files)}",
                "branch": "main",
            })
            if isinstance(result, dict) and result.get("error"):
                print(f"[FILE_BLOCK] Error subiendo: {result['error']}")
                return []
            print(f"[FILE_BLOCK] Subidos {len(uploaded_files)} archivos: {uploaded_files}")
        except Exception as exc:
            print(f"[FILE_BLOCK] Excepción subiendo: {exc}")
            return []

        return uploaded_files

    # ================================================================
    # TOOL SCHEMAS
    # ================================================================
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            # WEB
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Busca en internet con DuckDuckGo.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                }, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "web_fetch",
                "description": "Descarga una URL y devuelve el texto legible.",
                "parameters": {"type": "object", "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 8000},
                }, "required": ["url"]},
            }},

            # ARCHIVOS
            {"type": "function", "function": {
                "name": "file_list",
                "description": "Lista archivos y carpetas del workspace.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string", "default": "."},
                }},
            }},
            {"type": "function", "function": {
                "name": "file_read",
                "description": "Lee un archivo de texto.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_write",
                "description": (
                    "Crea o reemplaza un archivo de texto. LÍMITE DURO: máximo 50 líneas por llamada. "
                    "Para archivos grandes, escribe el código en tu RESPUESTA DE TEXTO con el formato:\n"
                    "FILE: nombre.ext\n```lenguaje\n...contenido...\n```\n"
                    "y el sistema lo extraerá y subirá automáticamente."
                ),
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            }},
            {"type": "function", "function": {
                "name": "file_append",
                "description": "Añade contenido al final de un archivo. LÍMITE DURO: 50 líneas por llamada.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            }},
            {"type": "function", "function": {
                "name": "file_search",
                "description": "Busca archivos por patrón.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "file_info",
                "description": "Info de un archivo o carpeta.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_mkdir",
                "description": "Crea una carpeta.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_delete",
                "description": "Elimina un archivo o carpeta.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean", "default": False}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_tree",
                "description": "Árbol recursivo del workspace.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "max_depth": {"type": "integer", "default": 3}}},
            }},
            {"type": "function", "function": {
                "name": "file_grep",
                "description": "Busca texto dentro de archivos.",
                "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["pattern"]},
            }},
            {"type": "function", "function": {
                "name": "file_backup",
                "description": "Copia de seguridad con timestamp.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_read_pdf",
                "description": "Extrae el texto de un PDF.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_count_words",
                "description": "Cuenta las palabras de un archivo.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_stats",
                "description": "Estadísticas de un archivo.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "file_replace_many",
                "description": "Reemplaza varias cadenas en un archivo.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "replacements": {"type": "object"}}, "required": ["path", "replacements"]},
            }},
            {"type": "function", "function": {
                "name": "file_search_and_replace_dir",
                "description": "Reemplaza texto en todos los archivos de una carpeta.",
                "parameters": {"type": "object", "properties": {"directory": {"type": "string", "default": "."}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]},
            }},

            # GITHUB
            {"type": "function", "function": {
                "name": "github_search_code",
                "description": "Busca archivos o código en GitHub antes de github_read.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "github_list",
                "description": "Lista tus repositorios de GitHub.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "github_tree",
                "description": "Lista recursivamente los archivos de un repo o carpeta.",
                "parameters": {"type": "object", "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string", "default": ""},
                    "ref": {"type": "string", "default": "main"},
                }, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_read",
                "description": "Lee un archivo de un repositorio.",
                "parameters": {"type": "object", "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "ref": {"type": "string", "default": "main"},
                }, "required": ["repo", "path"]},
            }},
            {"type": "function", "function": {
                "name": "github_create_branch",
                "description": "Crea una rama nueva en un repo.",
                "parameters": {"type": "object", "properties": {
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "from_branch": {"type": "string", "default": "main"},
                }, "required": ["repo", "branch"]},
            }},
            {"type": "function", "function": {
                "name": "github_write",
                "description": (
                    "Escribe un archivo en GitHub. LÍMITE DURO: máximo 50 líneas en 'content'. "
                    "Para archivos grandes, USA EL FORMATO 'FILE: nombre.ext' + bloque ``` en tu "
                    "respuesta de texto. El sistema lo extraerá y subirá automáticamente."
                ),
                "parameters": {"type": "object", "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "message": {"type": "string"},
                    "branch": {"type": "string", "default": "main"},
                }, "required": ["repo", "path", "content", "message"]},
            }},
            {"type": "function", "function": {
                "name": "github_create_repo",
                "description": "Crea un repositorio nuevo.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "private": {"type": "boolean", "default": False}}, "required": ["name"]},
            }},
            {"type": "function", "function": {
                "name": "github_upload_project",
                "description": (
                    "Sube uno o varios archivos a GitHub usando la Git Data API. "
                    "Parámetro 'files': diccionario {ruta_en_repo: ruta_local}."
                ),
                "parameters": {"type": "object", "properties": {
                    "repo": {"type": "string"},
                    "files": {"type": "object"},
                    "message": {"type": "string"},
                    "branch": {"type": "string", "default": "main"},
                }, "required": ["repo", "files", "message"]},
            }},
            {"type": "function", "function": {
                "name": "github_create_issue",
                "description": "Crea un issue.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["repo", "title"]},
            }},
            {"type": "function", "function": {
                "name": "github_list_issues",
                "description": "Lista los issues de un repositorio.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "state": {"type": "string", "default": "open"}}, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_create_pr",
                "description": "Crea un pull request.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "title": {"type": "string"}, "head": {"type": "string"}, "base": {"type": "string", "default": "main"}, "body": {"type": "string"}}, "required": ["repo", "title", "head"]},
            }},
            {"type": "function", "function": {
                "name": "github_list_prs",
                "description": "Lista los pull requests de un repositorio.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "state": {"type": "string", "default": "open"}}, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_merge_pr",
                "description": "Fusiona un pull request.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "pr_number": {"type": "integer"}}, "required": ["repo", "pr_number"]},
            }},
            {"type": "function", "function": {
                "name": "github_list_branches",
                "description": "Lista las ramas de un repositorio.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_delete_branch",
                "description": "Elimina una rama.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "branch": {"type": "string"}}, "required": ["repo", "branch"]},
            }},
            {"type": "function", "function": {
                "name": "github_get_commit",
                "description": "Info detallada de un commit.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "sha": {"type": "string"}}, "required": ["repo", "sha"]},
            }},
            {"type": "function", "function": {
                "name": "github_list_commits",
                "description": "Historial de commits.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "branch": {"type": "string", "default": "main"}}, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_delete_file",
                "description": "Elimina un archivo (necesita SHA).",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "message": {"type": "string"}, "sha": {"type": "string"}}, "required": ["repo", "path", "message", "sha"]},
            }},
            {"type": "function", "function": {
                "name": "github_star_repo",
                "description": "Marca un repo con estrella.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
            }},
            {"type": "function", "function": {
                "name": "github_fork_repo",
                "description": "Hace fork de un repositorio.",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
            }},

            # VERCEL
            {"type": "function", "function": {
                "name": "vercel_projects",
                "description": "Lista tus proyectos de Vercel.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "vercel_deployments",
                "description": "Lista deployments de un proyecto.",
                "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]},
            }},
            {"type": "function", "function": {
                "name": "vercel_set_env",
                "description": "Configura una variable de entorno.",
                "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "key": {"type": "string"}, "value": {"type": "string"}}, "required": ["project", "key", "value"]},
            }},
            {"type": "function", "function": {
                "name": "vercel_redeploy",
                "description": "Fuerza un nuevo deployment.",
                "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "target": {"type": "string", "default": "production"}}, "required": ["project"]},
            }},

            # IMAGEN
            {"type": "function", "function": {
                "name": "generate_image",
                "description": "Genera una imagen con Subtom IA Image (Pollinations.AI). La imagen se envía automáticamente al chat.",
                "parameters": {"type": "object", "properties": {
                    "prompt": {"type": "string"},
                }, "required": ["prompt"]},
            }},

            # SANDBOX
            {"type": "function", "function": {
                "name": "sandbox_run_python",
                "description": "Ejecuta código Python en sandbox.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_run_shell",
                "description": "Ejecuta un comando shell.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_run_node",
                "description": "Ejecuta código JavaScript.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_run_bash",
                "description": "Ejecuta un script bash.",
                "parameters": {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_pip_install",
                "description": "Instala paquetes de Python.",
                "parameters": {"type": "object", "properties": {"packages": {"type": "array", "items": {"type": "string"}}}, "required": ["packages"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_analyze_json",
                "description": "Analiza un JSON.",
                "parameters": {"type": "object", "properties": {"json_str": {"type": "string"}}, "required": ["json_str"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_regex_test",
                "description": "Prueba una regex.",
                "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "text": {"type": "string"}, "flags": {"type": "string", "default": ""}}, "required": ["pattern", "text"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_http_request",
                "description": "Hace una petición HTTP.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string", "default": "GET"}, "body": {"type": "string"}}, "required": ["url"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_hash_text",
                "description": "Calcula el hash de un texto.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "algorithm": {"type": "string", "default": "sha256"}}, "required": ["text"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_ocr_image",
                "description": "OCR de una imagen.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lang": {"type": "string", "default": "spa+eng"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_extract_pdf",
                "description": "Extrae texto de un PDF.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_git_clone",
                "description": "Clona un repositorio.",
                "parameters": {"type": "object", "properties": {"repo_url": {"type": "string"}, "dest": {"type": "string"}}, "required": ["repo_url"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_download",
                "description": "Descarga un archivo de una URL.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "output": {"type": "string"}}, "required": ["url", "output"]},
            }},
            {"type": "function", "function": {
                "name": "sandbox_verify_change",
                "description": "Verifica un cambio en un archivo Python antes de subirlo.",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_content": {"type": "string"}, "new_content": {"type": "string"}}, "required": ["file_path", "old_content", "new_content"]},
            }},

            # DISCORD
            {"type": "function", "function": {
                "name": "discord_send_message",
                "description": "Envía un mensaje a un canal.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "content": {"type": "string"}, "embed": {"type": "object"}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_create_poll",
                "description": "Crea una encuesta.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}, "duration_hours": {"type": "integer", "default": 24}}, "required": ["channel_id", "question", "options"]},
            }},
            {"type": "function", "function": {
                "name": "discord_send_dm",
                "description": "Envía un DM a un usuario.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "content": {"type": "string"}}, "required": ["user_id", "content"]},
            }},
            {"type": "function", "function": {
                "name": "discord_purge",
                "description": "Borra los últimos N mensajes.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "amount": {"type": "integer", "default": 10}, "user_id": {"type": "integer"}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_ban",
                "description": "Banea a un usuario.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "reason": {"type": "string"}, "delete_days": {"type": "integer", "default": 0}}, "required": ["user_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_kick",
                "description": "Expulsa a un usuario.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["user_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_timeout",
                "description": "Silencia a un usuario.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "minutes": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["user_id", "minutes"]},
            }},
            {"type": "function", "function": {
                "name": "discord_list_roles",
                "description": "Lista los roles del servidor.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "discord_add_role",
                "description": "Añade un rol a un miembro.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "role_id": {"type": "integer"}}, "required": ["user_id", "role_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_remove_role",
                "description": "Quita un rol a un miembro.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "role_id": {"type": "integer"}}, "required": ["user_id", "role_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_list_channels",
                "description": "Lista los canales del servidor.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "discord_create_channel",
                "description": "Crea un canal.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "channel_type": {"type": "string", "default": "text"}, "category_id": {"type": "integer"}}, "required": ["name"]},
            }},
            {"type": "function", "function": {
                "name": "discord_delete_channel",
                "description": "Borra un canal.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_edit_channel",
                "description": "Edita un canal.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "name": {"type": "string"}, "topic": {"type": "string"}, "slowmode": {"type": "integer"}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_create_invite",
                "description": "Crea una invitación.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "max_age": {"type": "integer", "default": 86400}, "max_uses": {"type": "integer", "default": 0}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_pin_message",
                "description": "Fija un mensaje.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "message_id": {"type": "integer"}}, "required": ["channel_id", "message_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_list_pins",
                "description": "Lista los pins de un canal.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}}, "required": ["channel_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_get_guild_info",
                "description": "Info del servidor.",
                "parameters": {"type": "object", "properties": {}},
            }},
            {"type": "function", "function": {
                "name": "discord_get_user_info",
                "description": "Info de un usuario.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}}, "required": ["user_id"]},
            }},
            {"type": "function", "function": {
                "name": "discord_add_reaction",
                "description": "Añade una reacción.",
                "parameters": {"type": "object", "properties": {"channel_id": {"type": "integer"}, "message_id": {"type": "integer"}, "emoji": {"type": "string"}}, "required": ["channel_id", "message_id", "emoji"]},
            }},
            {"type": "function", "function": {
                "name": "discord_send_file",
                "description": "Envía un archivo del workspace a Discord.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "caption": {"type": "string", "default": ""}}, "required": ["path"]},
            }},
        ]

    # ================================================================
    # RUN TOOL
    # ================================================================
    async def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            # GUARD
            if name in ("file_write", "file_append", "github_write"):
                content = args.get("content", "")
                if isinstance(content, str):
                    line_count = content.count("\n") + 1
                    if line_count > MAX_LINES_PER_TOOL_CALL:
                        print(f"[TOOL_GUARD] {name} rechazado: {line_count} líneas (>{MAX_LINES_PER_TOOL_CALL})")
                        return {
                            "error": f"CONTENIDO DEMASIADO GRANDE: {line_count} líneas (máximo {MAX_LINES_PER_TOOL_CALL})",
                            "instruction": (
                                f"El bloque tiene {line_count} líneas y el máximo es {MAX_LINES_PER_TOOL_CALL}. "
                                "USA EL FORMATO 'FILE: nombre.ext' + bloque ``` en tu respuesta de texto. "
                                "El sistema lo extraerá y lo subirá automáticamente sin límites."
                            ),
                        }

            # WEB
            if name == "web_search":
                return await self._cached_search(args)
            if name == "web_fetch":
                return await self._cached_fetch(args)

            # ARCHIVOS
            if name == "file_list":
                return {"files": self.files.list_files(args.get("path", "."))}
            if name == "file_read":
                return {"content": self.files.read_file(args["path"])}
            if name == "file_write":
                filepath = self.files.write_file(args["path"], args["content"])
                content_written = args.get("content", "")
                lines_written = content_written.count("\n") + 1
                return {
                    "path": filepath,
                    "lines_written": lines_written,
                    "hash": hashlib.blake2b(content_written.encode("utf-8", "ignore"), digest_size=4).hexdigest(),
                }
            if name == "file_append":
                filepath = self.files.append_file(args["path"], args["content"])
                content_added = args.get("content", "")
                lines_added = content_added.count("\n") + 1
                try:
                    total_lines = len(Path(filepath).read_text(encoding="utf-8").splitlines())
                except Exception:
                    total_lines = 0
                return {
                    "path": filepath,
                    "lines_added": lines_added,
                    "total_lines": total_lines,
                    "hash": hashlib.blake2b(content_added.encode("utf-8", "ignore"), digest_size=4).hexdigest(),
                }
            if name == "file_search":
                return {"matches": self.files.search(args["query"], args.get("path", "."))}
            if name == "file_info":
                return self.files.info(args["path"])
            if name == "file_mkdir":
                return {"path": self.files.mkdir(args["path"])}
            if name == "file_delete":
                return {"deleted": self.files.delete(args["path"], args.get("recursive", False))}
            if name == "file_tree":
                return self.files.tree(args.get("path", "."), int(args.get("max_depth", 3)))
            if name == "file_grep":
                return {"results": self.files.grep(args["pattern"], args.get("path", "."), None)}
            if name == "file_backup":
                return {"backup": self.files.backup(args["path"])}
            if name == "file_read_pdf":
                return {"text": self.files.read_pdf_text(args["path"])}
            if name == "file_count_words":
                return {"words": self.files.count_words(args["path"])}
            if name == "file_stats":
                return self.files.file_stats(args["path"])
            if name == "file_replace_many":
                return self.files.replace_many(args["path"], args["replacements"])
            if name == "file_search_and_replace_dir":
                return self.files.search_and_replace_dir(args.get("directory", "."), args["old"], args["new"])

            # GITHUB
            if name == "github_search_code":
                return await self.github_request("GET", f"/search/code?q={args['query']}")
            if name == "github_list":
                return await self.github_request("GET", "/user/repos?per_page=100")
            if name == "github_tree":
                repo = args["repo"].strip("/")
                path = args.get("path", "").strip("/")
                ref = args.get("ref", "main")
                endpoint = f"/repos/{repo}/git/trees/{ref}"
                if path:
                    endpoint = f"/repos/{repo}/contents/{path}?ref={ref}"
                result = await self.github_request("GET", endpoint)
                if isinstance(result, dict) and result.get("tree"):
                    result["tree"] = [
                        {"path": t["path"], "type": t["type"], "size": t.get("size", 0)}
                        for t in result["tree"] if t.get("type") in ("blob", "tree")
                    ]
                return result
            if name == "github_read":
                return await self.github_read_with_fallback(args)
            if name == "github_create_branch":
                return await self.github_create_branch(args)
            if name == "github_write":
                return await self.github_write(args)
            if name == "github_create_repo":
                return await self.github_create_repo(args)
            if name == "github_upload_project":
                return await self.github_upload_project(args)
            if name == "github_create_issue":
                return await self.github_request("POST", f"/repos/{args['repo'].strip('/')}/issues", json={"title": args["title"], "body": args.get("body", "")})
            if name == "github_list_issues":
                return await self.github_request("GET", f"/repos/{args['repo'].strip('/')}/issues?state={args.get('state', 'open')}")
            if name == "github_create_pr":
                return await self.github_request("POST", f"/repos/{args['repo'].strip('/')}/pulls", json={"title": args["title"], "head": args["head"], "base": args.get("base", "main"), "body": args.get("body", "")})
            if name == "github_list_prs":
                return await self.github_request("GET", f"/repos/{args['repo'].strip('/')}/pulls?state={args.get('state', 'open')}")
            if name == "github_merge_pr":
                return await self.github_request("PUT", f"/repos/{args['repo'].strip('/')}/pulls/{args['pr_number']}/merge")
            if name == "github_list_branches":
                return await self.github_request("GET", f"/repos/{args['repo'].strip('/')}/branches")
            if name == "github_delete_branch":
                return await self.github_request("DELETE", f"/repos/{args['repo'].strip('/')}/git/refs/heads/{args['branch']}")
            if name == "github_get_commit":
                return await self.github_request("GET", f"/repos/{args['repo'].strip('/')}/commits/{args['sha']}")
            if name == "github_list_commits":
                return await self.github_request("GET", f"/repos/{args['repo'].strip('/')}/commits?sha={args.get('branch', 'main')}")
            if name == "github_delete_file":
                return await self.github_request("DELETE", f"/repos/{args['repo'].strip('/')}/contents/{args['path'].lstrip('/')}", json={"message": args["message"], "sha": args["sha"]})
            if name == "github_star_repo":
                return await self.github_request("PUT", f"/user/starred/{args['repo'].strip('/')}")
            if name == "github_fork_repo":
                return await self.github_request("POST", f"/repos/{args['repo'].strip('/')}/forks")

            # VERCEL
            if name == "vercel_projects":
                return await self.vercel_request("GET", "/v9/projects?limit=100")
            if name == "vercel_deployments":
                project = aiohttp.helpers.quote(args["project"], safe="")
                return await self.vercel_request("GET", f"/v6/deployments?projectId={project}&limit=20")
            if name == "vercel_set_env":
                return await self.vercel_set_env(args)
            if name == "vercel_redeploy":
                return await self.vercel_redeploy(args)

            # IMAGEN
            if name == "generate_image":
                return await self.generate_image(args["prompt"])

            # SANDBOX
            if name == "sandbox_run_python":
                from sandbox import sandbox
                return await sandbox.run_python(args["code"])
            if name == "sandbox_run_shell":
                from sandbox import sandbox
                return await sandbox.run_shell(args["command"])
            if name == "sandbox_run_node":
                from sandbox import sandbox
                return await sandbox.run_node(args["code"])
            if name == "sandbox_run_bash":
                from sandbox import sandbox
                return await sandbox.run_bash(args["script"])
            if name == "sandbox_pip_install":
                from sandbox import sandbox
                return await sandbox.pip_install(args["packages"])
            if name == "sandbox_analyze_json":
                from sandbox import sandbox
                return await sandbox.analyze_json(args["json_str"])
            if name == "sandbox_regex_test":
                from sandbox import sandbox
                return await sandbox.regex_test(args["pattern"], args["text"], args.get("flags", ""))
            if name == "sandbox_http_request":
                from sandbox import sandbox
                return await sandbox.http_request(args["url"], args.get("method", "GET"), body=args.get("body"))
            if name == "sandbox_hash_text":
                from sandbox import sandbox
                return await sandbox.hash_text(args["text"], args.get("algorithm", "sha256"))
            if name == "sandbox_ocr_image":
                from sandbox import sandbox
                return await sandbox.ocr_image(args["path"], args.get("lang", "spa+eng"))
            if name == "sandbox_extract_pdf":
                from sandbox import sandbox
                return await sandbox.extract_text_pdf(args["path"])
            if name == "sandbox_git_clone":
                from sandbox import sandbox
                return await sandbox.git_clone(args["repo_url"], args.get("dest"))
            if name == "sandbox_download":
                from sandbox import sandbox
                return await sandbox.download(args["url"], args["output"])
            if name == "sandbox_verify_change":
                from sandbox import sandbox
                return await sandbox.verify_python_change(args["file_path"], args["old_content"], args["new_content"])

            # DISCORD
            if name == "discord_send_message":
                import discord_tools
                return await discord_tools.send_message(args["channel_id"], args.get("content", ""), args.get("embed"))
            if name == "discord_create_poll":
                import discord_tools
                return await discord_tools.create_poll(args["channel_id"], args["question"], args["options"], int(args.get("duration_hours", 24)))
            if name == "discord_send_dm":
                import discord_tools
                return await discord_tools.send_dm(args["user_id"], args["content"])
            if name == "discord_purge":
                import discord_tools
                return await discord_tools.purge_messages(args["channel_id"], int(args.get("amount", 10)), args.get("user_id"))
            if name == "discord_ban":
                import discord_tools
                return await discord_tools.ban_member(args["user_id"], args.get("reason", ""), int(args.get("delete_days", 0)))
            if name == "discord_kick":
                import discord_tools
                return await discord_tools.kick_member(args["user_id"], args.get("reason", ""))
            if name == "discord_timeout":
                import discord_tools
                return await discord_tools.timeout_member(args["user_id"], int(args["minutes"]), args.get("reason", ""))
            if name == "discord_list_roles":
                import discord_tools
                return await discord_tools.list_roles()
            if name == "discord_add_role":
                import discord_tools
                return await discord_tools.add_role(args["user_id"], args["role_id"])
            if name == "discord_remove_role":
                import discord_tools
                return await discord_tools.remove_role(args["user_id"], args["role_id"])
            if name == "discord_list_channels":
                import discord_tools
                return await discord_tools.list_channels()
            if name == "discord_create_channel":
                import discord_tools
                return await discord_tools.create_channel(args["name"], args.get("channel_type", "text"), args.get("category_id"))
            if name == "discord_delete_channel":
                import discord_tools
                return await discord_tools.delete_channel(args["channel_id"], args.get("reason", ""))
            if name == "discord_edit_channel":
                import discord_tools
                return await discord_tools.edit_channel(args["channel_id"], args.get("name"), args.get("topic"), args.get("slowmode"))
            if name == "discord_create_invite":
                import discord_tools
                return await discord_tools.create_invite(args["channel_id"], int(args.get("max_age", 86400)), int(args.get("max_uses", 0)))
            if name == "discord_pin_message":
                import discord_tools
                return await discord_tools.pin_message(args["channel_id"], args["message_id"])
            if name == "discord_list_pins":
                import discord_tools
                return await discord_tools.list_pins(args["channel_id"])
            if name == "discord_get_guild_info":
                import discord_tools
                return await discord_tools.get_guild_info()
            if name == "discord_get_user_info":
                import discord_tools
                return await discord_tools.get_user_info(args["user_id"])
            if name == "discord_add_reaction":
                import discord_tools
                return await discord_tools.add_reaction(args["channel_id"], args["message_id"], args["emoji"])
            if name == "discord_send_file":
                raw_path = args.get("path", "").strip()
                if not raw_path:
                    return {"error": "path vacío"}
                p = Path(raw_path)
                resolved: Path | None = None
                if p.is_absolute():
                    if p.exists() and p.is_file():
                        resolved = p
                else:
                    if p.exists() and p.is_file():
                        resolved = p
                    else:
                        candidate = Path(config.workspace) / raw_path
                        if candidate.exists() and candidate.is_file():
                            resolved = candidate
                        else:
                            candidate2 = Path(config.workspace) / p.name
                            if candidate2.exists() and candidate2.is_file():
                                resolved = candidate2
                if resolved is None:
                    return {
                        "error": f"Archivo no encontrado: {raw_path}",
                        "note": f"Buscado en CWD y en {config.workspace}",
                        "_no_retry": True,
                    }
                print(f"[DISCORD_SEND_FILE] {raw_path} -> {resolved}")
                return {"_send_file": str(resolved), "caption": args.get("caption", "")}

            return {"error": f"Herramienta desconocida: {name}"}

        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}

    # ================================================================
    # GITHUB HELPERS
    # ================================================================

    async def github_read_with_fallback(self, args: dict[str, Any]) -> dict[str, Any]:
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        ref = args.get("ref", "main")

        result = await self.github_request("GET", f"/repos/{repo}/contents/{path}?ref={ref}")
        if not (isinstance(result, dict) and result.get("error")):
            return result

        if "/" in path:
            alt_path = path.split("/", 1)[1]
            print(f"[GITHUB] {path} no existe, probando {alt_path}...")
            alt_result = await self.github_request("GET", f"/repos/{repo}/contents/{alt_path}?ref={ref}")
            if not (isinstance(alt_result, dict) and alt_result.get("error")):
                alt_result["_note"] = f"Ruta original '{path}' no existía, se usó '{alt_path}'."
                return alt_result

        base_name = path.split("/")[-1]
        if base_name != path:
            root_result = await self.github_request("GET", f"/repos/{repo}/contents/{base_name}?ref={ref}")
            if not (isinstance(root_result, dict) and root_result.get("error")):
                root_result["_note"] = f"Ruta original '{path}' no existía, se usó '{base_name}' en la raíz."
                return root_result

        return result

    async def github_create_branch(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        branch = args["branch"]
        from_branch = args.get("from_branch", "main")
        ref = await self.github_request("GET", f"/repos/{repo}/git/refs/heads/{from_branch}")
        if isinstance(ref, dict) and ref.get("error"):
            return ref
        sha = ref["object"]["sha"]
        return await self.github_request("POST", f"/repos/{repo}/git/refs", json={
            "ref": f"refs/heads/{branch}",
            "sha": sha,
        })

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
        if isinstance(result, dict):
            if result.get("text") and not result.get("error"):
                motor.cache_set_fetch(f"{url}::{max_chars}", result)
            else:
                result["_no_retry"] = True
                result["_instruction"] = "NO reintentes esta URL. Informa del error."
        return result

    async def github_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
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
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        path = args["path"].lstrip("/")
        new_content = args["content"]
        branch = args.get("branch", "main")

        if path.endswith(".py"):
            current = await self.github_request("GET", f"/repos/{repo}/contents/{path}?ref={branch}")
            if isinstance(current, dict) and not current.get("error"):
                try:
                    old_content = base64.b64decode(current.get("content", "")).decode("utf-8", "replace")
                except Exception:
                    old_content = ""
                if old_content:
                    from sandbox import sandbox
                    verify = await sandbox.verify_python_change(path, old_content, new_content)
                    if not verify.get("ok"):
                        return {
                            "error": "VERIFICACIÓN FALLIDA - No se subió",
                            "errors": verify.get("errors"),
                            "warnings": verify.get("warnings"),
                            "instruction": "El contenido nuevo tiene errores. CORRÍGELOS y vuelve a intentar.",
                            "_no_retry": True,
                        }
                    if verify.get("warnings"):
                        print(f"[GITHUB] Warnings: {verify['warnings']}")

        if path.endswith(".py"):
            from sandbox import sandbox
            test_result = await sandbox.run_shell("python test_tools.py 2>&1 || true")
            stdout = (test_result.get("stdout") or "") if isinstance(test_result, dict) else ""
            if "TESTS FALLIDOS" in stdout:
                return {
                    "error": "TESTS FALLIDOS - No se subió",
                    "detail": stdout[:2000],
                    "instruction": "Los tests fallaron. CORRIGE el código antes de reintentar.",
                    "_no_retry": True,
                }

        content = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {
            "Authorization": f"Bearer {config.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            sha = None
            async with session.get(url + f"?ref={branch}", headers=headers) as response:
                if response.status == 200:
                    existing = await response.json(content_type=None)
                    sha = existing.get("sha")
                elif response.status not in {404, 301, 302}:
                    detail = await response.text()
                    return {"error": f"GitHub HTTP {response.status}", "detail": detail[:3000]}
            payload = {"message": args["message"], "content": content, "branch": branch}
            if sha:
                payload["sha"] = sha
            async with session.put(url, headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"GitHub HTTP {response.status}", "detail": data}
                return {"ok": True, "verified": True, "sha": data.get("content", {}).get("sha")}

    async def github_create_repo(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        payload = {"name": args["name"], "description": args.get("description", ""), "private": bool(args.get("private", False)), "auto_init": True}
        headers = {"Authorization": f"Bearer {config.github_token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.github.com/user/repos", headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"name": data.get("full_name"), "url": data.get("html_url")}

    async def github_upload_project(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        repo = args["repo"].strip("/")
        files_map = args.get("files") or {}
        message = args["message"]
        branch = args.get("branch", "main")
        if not files_map:
            return {"error": "No hay archivos para subir."}
        headers = {"Authorization": f"Bearer {config.github_token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        base = f"https://api.github.com/repos/{repo}"
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base}/git/refs/heads/{branch}", headers=headers) as resp:
                if resp.status == 404:
                    return {"error": f"La rama '{branch}' no existe."}
                ref = await resp.json(content_type=None)
                parent_sha = ref["object"]["sha"]
            async with session.get(f"{base}/git/commits/{parent_sha}", headers=headers) as resp:
                commit = await resp.json(content_type=None)
                parent_tree = commit["tree"]["sha"]
            tree_items = []
            uploaded = []
            for repo_path, ws_path in files_map.items():
                try:
                    full = Path(ws_path)
                    if not full.is_absolute() and not full.exists():
                        candidate = Path(config.workspace) / ws_path
                        if candidate.exists():
                            full = candidate
                    if not full.is_file():
                        print(f"[GITHUB_UPLOAD] No encontrado: {ws_path}")
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
                return {"error": "Ningún archivo válido."}
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
            return {"repo": repo, "branch": branch, "commit": commit_sha, "files_uploaded": len(uploaded)}

    async def vercel_request(self, method: str, path: str) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        headers = {"Authorization": f"Bearer {config.vercel_token}"}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, "https://api.vercel.com" + path, headers=headers) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"Vercel HTTP {response.status}", "detail": data}
                return data

    async def _vercel_project_id(self, project: str) -> str | None:
        if project.startswith("prj_"):
            return project
        data = await self.vercel_request("GET", f"/v9/projects/{project}")
        if isinstance(data, dict) and data.get("id"):
            return data["id"]
        return None

    async def vercel_set_env(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        project = args["project"]
        key = args["key"]
        value = args["value"]
        target = ["production", "preview", "development"]
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        headers = {"Authorization": f"Bearer {config.vercel_token}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"https://api.vercel.com/v10/projects/{pid}/env", headers=headers, json={"key": key, "value": value, "target": target, "type": "encrypted"}) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"action": "created", "key": key}

    async def vercel_redeploy(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no configurado."}
        project = args["project"]
        target = args.get("target", "production")
        pid = await self._vercel_project_id(project)
        if not pid:
            return {"error": f"Proyecto '{project}' no encontrado."}
        headers = {"Authorization": f"Bearer {config.vercel_token}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.vercel.com/v13/deployments", headers=headers, json={"name": project, "project": pid, "target": target}) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"id": data.get("id"), "url": data.get("url"), "status": data.get("status")}

    # ================================================================
    # GENERACIÓN DE IMÁGENES — POLLINATIONS.AI con reintentos
    # ================================================================

    async def generate_image(self, prompt: str, _retry: int = 0) -> dict[str, Any]:
        """
        Genera una imagen con Pollinations.AI.
        - Reintenta hasta 3 veces.
        - Prueba modelos alternativos si el principal falla (500, 502, timeout).
        - Devuelve _send_file para que el bot la envíe al chat.
        """
        if not prompt or not prompt.strip():
            return {"error": "prompt vacío"}
        prompt = prompt.strip()

        prompt_encoded = aiohttp.helpers.quote(prompt, safe="")

        # Elegir modelo según el intento (rota si falla)
        model = POLLINATIONS_MODELS[_retry % len(POLLINATIONS_MODELS)]

        url = (
            f"https://image.pollinations.ai/prompt/{prompt_encoded}"
            f"?model={model}&width=1024&height=1024&nologo=true&enhance=true"
        )

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 500 or resp.status == 502 or resp.status == 503:
                        print(f"[POLLINATIONS] HTTP {resp.status} con model={model} (intento {_retry + 1}/3)")
                        if _retry < 2:
                            await asyncio.sleep(2)
                            return await self.generate_image(prompt, _retry + 1)
                        return {"error": f"Pollinations HTTP {resp.status} tras 3 intentos con modelos distintos"}
                    if resp.status != 200:
                        return {"error": f"Pollinations HTTP {resp.status}"}
                    data = await resp.read()

            if not data or len(data) < 1000:
                print(f"[POLLINATIONS] Respuesta demasiado pequeña ({len(data)} bytes). Reintentando...")
                if _retry < 2:
                    await asyncio.sleep(2)
                    return await self.generate_image(prompt, _retry + 1)
                return {"error": "Pollinations devolvió una respuesta vacía o inválida"}

            from datetime import datetime
            folder = Path(config.workspace) / "generated"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:30]
            filename = f"{stamp}_{slug}.png"
            target = folder / filename
            target.write_bytes(data)

            abs_path = str(target.resolve())
            print(f"[POLLINATIONS] Imagen generada con {model}: {abs_path} ({len(data)} bytes)")

            return {
                "local_path": abs_path,
                "_send_file": abs_path,
                "caption": f"🎨 Imagen creada con Subtom IA Image\n\n_Prompt:_ {prompt[:200]}",
            }

        except asyncio.TimeoutError:
            print(f"[POLLINATIONS] Timeout con model={model} (intento {_retry + 1}/3)")
            if _retry < 2:
                return await self.generate_image(prompt, _retry + 1)
            return {"error": "Pollinations tardó demasiado tras 3 intentos"}
        except Exception as exc:
            print(f"[POLLINATIONS] Excepción con model={model}: {type(exc).__name__}: {exc}")
            if _retry < 2:
                await asyncio.sleep(2)
                return await self.generate_image(prompt, _retry + 1)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ================================================================
    # ASK
    # ================================================================
    async def ask(
        self,
        user_id: int,
        channel_id: int,
        prompt: str,
        force_image: bool = False,
    ) -> tuple[str, str | None, list[str] | None]:

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
        files_to_send: list[str] = []
        modelos = self._select_model(prompt, task, force_image)
        last_error: Exception | None = None

        MAX_SAFETY_ROUNDS = 100
        MAX_TOTAL_SECONDS = 300.0
        MAX_REPEAT_SAME_CALL = 5

        for modelo_actual in modelos:
            t0 = asyncio.get_event_loop().time()
            try:
                call_history: list[str] = []
                result_history: list[str] = []
                rounds_used = 0
                malformed_retries = 0

                while True:
                    rounds_used += 1
                    if rounds_used > MAX_SAFETY_ROUNDS:
                        raise RuntimeError(f"Límite de seguridad ({MAX_SAFETY_ROUNDS} rondas).")
                    elapsed = asyncio.get_event_loop().time() - t0
                    if elapsed > MAX_TOTAL_SECONDS:
                        raise RuntimeError(f"Tiempo total excedido ({int(elapsed)}s).")

                    response = await connector.complete(messages, self.tool_schemas(), model=modelo_actual)
                    choices = response.get("choices") or []
                    if not choices:
                        raise RuntimeError("La IA no devolvió ninguna elección.")

                    message = choices[0].get("message") or {}
                    tool_calls = self._extract_tool_calls(message)
                    finish_reason = choices[0].get("finish_reason", "unknown")

                    assistant_message: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    if not tool_calls and finish_reason == "malformed_function_call":
                        malformed_retries += 1
                        if malformed_retries <= 3:
                            print(f"[MOTOR] malformed_function_call. Reintentando ({malformed_retries}/3)...")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "⚠️ TU ÚLTIMO INTENTO FALLÓ con 'malformed_function_call'. "
                                    "OBLIGATORIO: escribe el código en un bloque 'FILE: nombre.ext' + ``` en tu respuesta de texto. "
                                    "NO uses github_write con contenido grande. Solo texto + bloque markdown. "
                                    "El sistema lo extraerá y subirá automáticamente."
                                )
                            })
                            continue
                        else:
                            answer = "⚠️ Gemini falló 3 veces con 'malformed_function_call'. Prueba de nuevo."
                            await self.save(user_id, channel_id, "assistant", answer)
                            return answer, image_url, files_to_send or None

                    if not tool_calls:
                        answer = (message.get("content") or "").strip()
                        if not answer:
                            if finish_reason in ("safety", "recitation", "blocked", "prohibited_content"):
                                answer = f"⚠️ Gemini bloqueó mi respuesta por sus filtros de seguridad ({finish_reason})."
                            elif finish_reason == "max_tokens":
                                answer = "⚠️ La respuesta se cortó por límite de tokens. Sube AI_MAX_TOKENS."
                            elif finish_reason == "other":
                                answer = "⚠️ Gemini devolvió un error genérico. Prueba otra vez."
                            else:
                                answer = f"⚠️ No he recibido respuesta del modelo (finish_reason: {finish_reason})."

                        # OPCIÓN 3: extraer bloques FILE: y subirlos a GitHub
                        if answer:
                            subidos = await self._process_file_blocks(answer)
                            if subidos:
                                answer += f"\n\n📦 Subí {len(subidos)} archivo(s) a GitHub: {', '.join(subidos)}"

                        await self.save(user_id, channel_id, "assistant", answer)
                        dt = asyncio.get_event_loop().time() - t0
                        motor.record(model=modelo_actual, task=task, lang=lang, latency=dt, error=False)
                        print(f"[MOTOR] Respuesta final tras {rounds_used} rondas ({int(elapsed)}s)")
                        return answer, image_url, files_to_send or None

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

                        call_sig = f"{name}::{json.dumps(args, sort_keys=True, default=str)}"
                        call_count = call_history.count(call_sig)
                        if call_count >= MAX_REPEAT_SAME_CALL:
                            print(f"[MOTOR] Bucle en '{name}' ({call_count + 1} veces). Cortando.")
                            raise RuntimeError(f"Bucle infinito en '{name}'.")
                        call_history.append(call_sig)

                        result = await self.run_tool(name, args)
                        if isinstance(result, dict):
                            if result.get("image_url"):
                                image_url = result["image_url"]
                            if result.get("_send_file"):
                                files_to_send.append(result["_send_file"])

                        if isinstance(result, dict) and result.get("_no_retry"):
                            err_msg = result.get("error", "error desconocido")
                            extra = result.get("instruction", "")
                            answer = f"⚠️ No pude completar la operación: {err_msg}"
                            if extra:
                                answer += f"\n\n{extra}"
                            await self.save(user_id, channel_id, "assistant", answer)
                            return answer, image_url, files_to_send or None

                        result_sig = connector.clean_tool_result(result)
                        result_hash = hashlib.blake2b(result_sig.encode("utf-8", "ignore"), digest_size=8).hexdigest()

                        if name == "sandbox_run_python" and isinstance(result, dict) and result.get("error"):
                            if result_history.count(result_hash) >= 1:
                                print(f"[MOTOR] sandbox_run_python falla repetidamente. Cortando.")
                                answer = f"No pude ejecutar el código en el sandbox. Error: {result.get('error')}"
                                await self.save(user_id, channel_id, "assistant", answer)
                                return answer, image_url, files_to_send or None

                        if result_history.count(result_hash) >= 5:
                            print(f"[MOTOR] Estancamiento en '{name}' ({result_history.count(result_hash) + 1} veces). Cortando.")
                            raise RuntimeError(f"Estancamiento en '{name}'.")
                        result_history.append(result_hash)

                        messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "name": name, "content": result_sig})
                        print(f"[MOTOR] Ronda {rounds_used}: tool '{name}' ({int(elapsed)}s)")

            except Exception as exc:
                last_error = exc
                dt = asyncio.get_event_loop().time() - t0
                motor.record(model=modelo_actual, task=task, lang=lang, latency=dt, error=True, error_msg=str(exc)[:200])
                err_txt = str(exc).lower()
                if "bucle" in err_txt or "estancamiento" in err_txt:
                    print(f"[MOTOR] Cortado: {exc}")
                    break
                if "image" in err_txt or "vision" in err_txt or "multimodal" in err_txt or "modality" in err_txt:
                    print(f"[VISION] '{modelo_actual}' rechazó imagen. Siguiente.")
                elif "404" in err_txt or "unavailable" in err_txt:
                    print(f"[MODEL-GONE] '{modelo_actual}' ya no existe.")
                else:
                    print(f"[FALLBACK] '{modelo_actual}' falló: {type(exc).__name__}: {str(exc)[:200]}")
                continue

        raise RuntimeError(f"Todos los modelos fallaron. Último error: {last_error}")


agent = Agent()
