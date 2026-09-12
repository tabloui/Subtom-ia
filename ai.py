from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg

from config import config
from connector import connector
from file_tools import FileTools


class Agent:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
        self.files = FileTools(config.workspace)
        self._flux_keys = deque(
            key for key in (config.flux_api_key_1, config.flux_api_key_2) if key
        )

    async def init(self) -> None:
        self.pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=3)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subtom_messages (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS subtom_messages_lookup
                ON subtom_messages(user_id, channel_id, created_at DESC)
            """)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def cleanup_memory(self) -> None:
        if not self.pool:
            return
        if config.memory_minutes <= 0:
            return
        await self.pool.execute(
            "DELETE FROM subtom_messages WHERE created_at < NOW() - ($1 * INTERVAL '1 minute')",
            config.memory_minutes,
        )

    async def save(self, user_id: int, channel_id: int, role: str, content: str) -> None:
        if not self.pool:
            return
        await self.pool.execute(
            "INSERT INTO subtom_messages(user_id, channel_id, role, content) VALUES($1,$2,$3,$4)",
            user_id, channel_id, role, content[:30000],
        )

    async def history(self, user_id: int, channel_id: int) -> list[dict[str, str]]:
        if not self.pool:
            return []
        rows = await self.pool.fetch(
            """
            SELECT role, content
            FROM subtom_messages
            WHERE user_id=$1 AND channel_id=$2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id, channel_id, config.memory_messages,
        )
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {"type":"function","function":{"name":"file_list","description":"Lista archivos y carpetas del workspace seguro.","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."}}}}},
            {"type":"function","function":{"name":"file_read","description":"Lee un archivo de texto del workspace seguro.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
            {"type":"function","function":{"name":"file_write","description":"Escribe/reemplaza un archivo del workspace seguro.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
            {"type":"function","function":{"name":"file_append","description":"Añade contenido al final de un archivo del workspace seguro.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
            {"type":"function","function":{"name":"file_search","description":"Busca texto en archivos del workspace.","parameters":{"type":"object","properties":{"query":{"type":"string"},"path":{"type":"string","default":"."}},"required":["query"]}}},
            {"type":"function","function":{"name":"file_info","description":"Obtiene información de un archivo o carpeta.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
            {"type":"function","function":{"name":"file_mkdir","description":"Crea una carpeta dentro del workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
            {"type":"function","function":{"name":"github_list","description":"Lista repositorios del usuario autenticado. Requiere GITHUB_TOKEN.","parameters":{"type":"object","properties":{}}}},
            {"type":"function","function":{"name":"github_read","description":"Lee un archivo de un repositorio GitHub. Requiere GITHUB_TOKEN.","parameters":{"type":"object","properties":{"repo":{"type":"string","description":"owner/repo"},"path":{"type":"string"}},"required":["repo","path"]}}},
            {"type":"function","function":{"name":"github_write","description":"Crea o actualiza un archivo en GitHub. Requiere GITHUB_TOKEN.","parameters":{"type":"object","properties":{"repo":{"type":"string"},"path":{"type":"string"},"content":{"type":"string"},"message":{"type":"string"}},"required":["repo","path","content","message"]}}},
            {"type":"function","function":{"name":"vercel_projects","description":"Lista proyectos de Vercel. Requiere VERCEL_TOKEN.","parameters":{"type":"object","properties":{}}}},
            {"type":"function","function":{"name":"vercel_deployments","description":"Lista deployments de Vercel. Requiere VERCEL_TOKEN.","parameters":{"type":"object","properties":{"project":{"type":"string"}},"required":["project"]}}},
            {"type":"function","function":{"name":"generate_image","description":"Genera una imagen con FLUX si hay una clave configurada.","parameters":{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"]}}},
        ]

    async def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "file_list":
            return self.files.list(args.get("path", "."))
        if name == "file_read":
            return {"content": self.files.read(args["path"])}
        if name == "file_write":
            return self.files.write(args["path"], args["content"])
        if name == "file_append":
            return self.files.append(args["path"], args["content"])
        if name == "file_search":
            return self.files.search(args["query"], args.get("path", "."))
        if name == "file_info":
            return self.files.info(args["path"])
        if name == "file_mkdir":
            return self.files.mkdir(args["path"])
        if name == "github_list":
            return await self.github_request("GET", "/user/repos?per_page=100")
        if name == "github_read":
            return await self.github_request(
                "GET", f"/repos/{args['repo']}/contents/{args['path']}"
            )
        if name == "github_write":
            return await self.github_write(args)
        if name == "vercel_projects":
            return await self.vercel_request("GET", "/v9/projects?limit=100")
        if name == "vercel_deployments":
            project = aiohttp.helpers.quote(args["project"], safe="")
            return await self.vercel_request("GET", f"/v6/deployments?projectId={project}&limit=20")
        if name == "generate_image":
            return await self.generate_image(args["prompt"])
        return {"error": f"Herramienta desconocida: {name}"}

    async def github_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}
        headers = {"Authorization": f"Bearer {config.github_token}", "Accept": "application/vnd.github+json"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.request(method, "https://api.github.com" + path, headers=headers, **kwargs) as r:
                text = await r.text()
                if r.status >= 400:
                    return {"error": f"GitHub HTTP {r.status}", "detail": text[:2000]}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"content": text}

    async def github_write(self, args: dict[str, Any]) -> dict[str, Any]:
        content = base64.b64encode(args["content"].encode()).decode()
        path = args["path"].lstrip("/")
        url = f"https://api.github.com/repos/{args['repo']}/contents/{path}"
        headers = {"Authorization": f"Bearer {config.github_token}", "Accept": "application/vnd.github+json"}
        if not config.github_token:
            return {"error": "GITHUB_TOKEN no está configurado."}

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            sha = None
            async with session.get(url, headers=headers) as r:
                if r.status == 200:
                    data = await r.json()
                    sha = data.get("sha")
            payload = {"message": args["message"], "content": content}
            if sha:
                payload["sha"] = sha
            async with session.put(url, headers=headers, json=payload) as r:
                data = await r.json(content_type=None)
                if r.status >= 400:
                    return {"error": f"GitHub HTTP {r.status}", "detail": data}
                return data

    async def vercel_request(self, method: str, path: str) -> dict[str, Any]:
        if not config.vercel_token:
            return {"error": "VERCEL_TOKEN no está configurado."}
        headers = {"Authorization": f"Bearer {config.vercel_token}"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.request(method, "https://api.vercel.com" + path, headers=headers) as r:
                data = await r.json(content_type=None)
                if r.status >= 400:
                    return {"error": f"Vercel HTTP {r.status}", "detail": data}
                return data

    async def generate_image(self, prompt: str) -> dict[str, Any]:
        if not self._flux_keys:
            return {"error": "No hay FLUX_API_KEY_1 configurada."}
        key = self._flux_keys[0]
        self._flux_keys.rotate(-1)
        base = config.flux_base_url.rstrip("/")
        headers = {"accept": "application/json", "x-key": key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post(
                f"{base}/{config.flux_endpoint}",
                headers=headers,
                json={"prompt": prompt},
            ) as r:
                data = await r.json(content_type=None)
                if r.status >= 400:
                    return {"error": f"FLUX HTTP {r.status}", "detail": data}
                task_id = data.get("id") or data.get("task_id")
            if not task_id:
                return {"result": data}

            for _ in range(60):
                await asyncio.sleep(2)
                async with session.get(f"{base}/get_result?id={task_id}", headers=headers) as r:
                    result = await r.json(content_type=None)
                    if r.status >= 400:
                        return {"error": f"FLUX polling HTTP {r.status}", "detail": result}
                    status = str(result.get("status", "")).lower()
                    if status in {"ready", "succeeded", "completed"}:
                        sample = result.get("result", {}).get("sample") or result.get("sample")
                        return {"image_url": sample} if sample else result
                    if status in {"failed", "error"}:
                        return {"error": "La generación de imagen falló.", "detail": result}
            return {"error": "La generación de imagen tardó demasiado."}

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
        if force_image:
            messages.append({
                "role": "user",
                "content": "Genera una imagen usando la herramienta generate_image para esta petición: " + prompt,
            })

        image_url: str | None = None
        for _ in range(config.ai_max_tool_rounds):
            response = await connector.complete(messages, self.tool_schemas())
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("La IA no devolvió ninguna elección.")

            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []

            assistant_msg = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                answer = (message.get("content") or "").strip()
                if not answer:
                    answer = "No he recibido una respuesta de texto del modelo."
                await self.save(user_id, channel_id, "assistant", answer)
                return answer, image_url

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await self.run_tool(name, args)
                except Exception as exc:
                    result = {"error": str(exc)[:2000]}

                if isinstance(result, dict) and result.get("image_url"):
                    image_url = result["image_url"]

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": connector.clean_tool_result(result),
                })

        raise RuntimeError("La IA agotó el número máximo de rondas de herramientas.")


agent = Agent()
