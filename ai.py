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

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
        self.files = FileTools(config.workspace)

        from collections import deque
        self._flux_keys = deque(
            key
            for key in (
                config.flux_api_key_1,
                config.flux_api_key_2,
            )
            if key
        )

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
            INSERT INTO subtom_memory(user_id, channel_id, role, content)
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
            WHERE user_id = $1 AND channel_id = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id,
            channel_id,
            config.memory_messages,
        )
        rows = list(reversed(rows))
        return [
            {"role": row["role"], "content": row["content"]}
            for row in rows
        ]

    def _extract_image_paths(self, prompt: str) -> list[str]:
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
            {"type": "text", "text": prompt}
        ]
        for path in image_paths:
            try:
                info = self.files.read_image_base64(path)
                if info["size"] > 20 * 1024 * 1024:
                    content.append({
                        "type": "text",
                        "text": f"[Imagen omitida >20MB: {info['filename']}]",
                    })
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": info["data_url"]},
                })
            except Exception as exc:
                content.append({
                    "type": "text",
                    "text": f"[No se pudo cargar {path}: {exc}]",
                })
        return content

    def _select_model(
        self,
        prompt: str,
        task: TaskType,
        force_image: bool = False,
    ) -> list[str]:
        return [config.ai_model]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Busca información actual en internet con "
                        "DuckDuckGo. Devuelve título, URL y snippet. "
                        "Usa web_fetch después para leer una URL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {
                                "type": "integer",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": (
                        "Descarga una página web y devuelve el texto "
                        "legible. Úsala después de web_search."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "max_chars": {
                                "type": "integer",
                                "default": 8000,
                                "minimum": 500,
                                "maximum": 30000,
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_list",
                    "description": "Lista archivos del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "default": "."},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": "Lee un archivo de texto.",
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
                    "description": "Crea o reemplaza un archivo.",
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
                    "description": "Añade al final de un archivo.",
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
                    "description": "Busca archivos por patrón.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
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
                    "description": "Info de un archivo.",
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
                    "description": "Crea una carpeta.",
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
                    "name": "file_delete",
                    "description": "Borra un archivo o carpeta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean", "default": False},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_tree",
                    "description": "Árbol del workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "default": "."},
                            "max_depth": {"type": "integer", "default": 3},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_grep",
                    "description": "Busca texto en archivos.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string", "default": "."},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_backup",
                    "description": "Backup con timestamp.",
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
                    "description": "Extrae texto de un PDF.",
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
                    "name": "github_list",
                    "description": "Lista repos del usuario.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_read",
                    "description": "Lee archivo de un repo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
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
                    "description": "Crea/actualiza archivo en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["repo", "path", "content", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_create_repo",
                    "description": "Crea un repo en GitHub.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "private": {"type": "boolean", "default": False},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "github_upload_project",
                    "description": "Sube varios archivos en un commit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "files": {"type": "object"},
                            "message": {"type": "string"},
                            "branch": {"type": "string", "default": "main"},
                        },
                        "required": ["repo", "files", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_projects",
                    "description": "Lista proyectos Vercel.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_deployments",
                    "description": "Deployments de un proyecto.",
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
                    "description": "Crea/actualiza env en Vercel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project": {"type": "string"},
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["project", "key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vercel_redeploy",
                    "description": "Redeploy de un proyecto.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project": {"type": "string"},
                            "target": {"type": "string", "default": "production"},
                        },
                        "required": ["project"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Genera una imagen con FLUX desde un prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
            },
        ]

    async def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "web_search":
                return await self._cached_search(args)
            if name == "web_fetch":
                return await self._cached_fetch(args)

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

            if name == "github_list":
                return await self.github_request("GET", "/user/repos?per_page=100")
            if name == "github_read":
                repo = args["repo"].strip("/")
                path = args["path"].lstrip("/")
                return await self.github_request("GET", f"/repos/{repo}/contents/{path}")
            if name == "github_write":
                return await self.github_write(args)
            if name == "github_create_repo":
                return await self.github_create_repo(args)
            if name == "github_upload_project":
                return await self.github_upload_project(args)

            if name == "vercel_projects":
                return await self.vercel_request("GET", "/v9/projects?limit=100")
            if name == "vercel_deployments":
                project = aiohttp.helpers.quote(args["project"], safe="")
                return await self.vercel_request("GET", f"/v6/deployments?projectId={project}&limit=20")
            if name == "vercel_set_env":
                return await self.vercel_set_env(args)
            if name == "vercel_redeploy":
                return await self.vercel_redeploy(args)

            if name == "generate_image":
                return await self.generate_image(args["prompt"])

            return {"error": f"Herramienta desconocida: {name}"}

        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}

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
            async with session.request(
                method, "https://api.github.com" + path, headers=headers, **kwargs
            ) as response:
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

    async def github_create_repo(self, args: dict[str, Any]) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no configurado."}
        payload = {
            "name": args["name"],
            "description": args.get("description", ""),
            "private": bool(args.get("private", False)),
            "auto_init": True,
        }
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
        headers = {
            "Authorization": f"Bearer {config.vercel_token}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"https://api.vercel.com/v10/projects/{pid}/env",
                headers=headers,
                json={"key": key, "value": value, "target": target, "type": "encrypted"},
            ) as resp:
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
        headers = {
            "Authorization": f"Bearer {config.vercel_token}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://api.vercel.com/v13/deployments",
                headers=headers,
                json={"name": project, "project": pid, "target": target},
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return {"error": f"HTTP {resp.status}", "detail": data}
                return {"id": data.get("id"), "url": data.get("url"), "status": data.get("status")}

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
        headers = {
            "accept": "application/json",
            "x-key": key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{base}/{model}",
                headers=headers,
                json={"prompt": prompt, "width": 1024, "height": 1024, "output_format": "png"},
            ) as response:
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
                        return {
                            "image_url": sample,
                            "local_path": local_path,
                            "note": "Imagen guardada. Usa github_upload_file con este local_path.",
                        }
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

    async def ask(
        self,
        user_id: int,
        channel_id: int,
        prompt: str,
        force_image: bool = False,
    ) -> tuple[str, str | None]:

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
            messages.append({
                "role": "user",
                "content": (
                    "Genera una imagen usando la herramienta "
                    "generate_image para esta petición:\n\n" + prompt
                ),
            })

        image_url: str | None = None
        modelos = self._select_model(prompt, task, force_image)
        last_error: Exception | None = None

        for modelo_actual in modelos:

            t0 = asyncio.get_event_loop().time()

            try:
                for _ in range(config.ai_max_tool_rounds):

                    response = await connector.complete(
                        messages,
                        self.tool_schemas(),
                        model=modelo_actual,
                    )

                    choices = response.get("choices") or []
                    if not choices:
                        raise RuntimeError("La IA no devolvió ninguna elección.")

                    message = choices[0].get("message") or {}
                    tool_calls = message.get("tool_calls") or []

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": message.get("content") or "",
                    }
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    if not tool_calls:
                        answer = (message.get("content") or "").strip()
                        if not answer:
                            answer = "No he recibido una respuesta de texto del modelo."
                        await self.save(user_id, channel_id, "assistant", answer)
                        dt = asyncio.get_event_loop().time() - t0
                        motor.record(
                            model=modelo_actual,
                            task=task,
                            lang=lang,
                            latency=dt,
                            error=False,
                        )
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

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.get("id") or "",
                            "content": connector.clean_tool_result(result),
                        })

                raise RuntimeError("La IA agotó el número máximo de rondas de herramientas.")

            except Exception as exc:
                last_error = exc
                dt = asyncio.get_event_loop().time() - t0
                motor.record(
                    model=modelo_actual,
                    task=task,
                    lang=lang,
                    latency=dt,
                    error=True,
                    error_msg=str(exc)[:200],
                )

                err_txt = str(exc).lower()

                if (
                    "image" in err_txt
                    or "vision" in err_txt
                    or "multimodal" in err_txt
                    or "modality" in err_txt
                ):
                    print(f"[VISION] '{modelo_actual}' rechazó imagen. Siguiente.")
                elif "404" in err_txt or "unavailable" in err_txt:
                    print(f"[MODEL-GONE] '{modelo_actual}' ya no existe. Siguiente.")
                else:
                    print(f"[FALLBACK] '{modelo_actual}' falló: {type(exc).__name__}: {str(exc)[:200]}")

                continue

        raise RuntimeError(
            "Todos los modelos gratuitos fallaron. "
            f"Último error: {last_error}"
        )


agent = Agent()
