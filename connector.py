from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from typing import Any

import aiohttp

from config import config, AISlot


def detect_provider(key: str, forced: str | None = None) -> tuple[str, str]:
    if forced:
        forced_map = {
            "ollama": ("ollama", "https://ollama.com/v1"),
            "aimlapi": ("aimlapi", "https://api.aimlapi.com/v1"),
            "together": ("together", "https://api.together.xyz/v1"),
            "deepinfra": ("deepinfra", "https://api.deepinfra.com/v1/openai"),
            "local": ("local", os.getenv("AI_LOCAL_URL", "http://127.0.0.1:8080/v1")),
            "termux": ("termux", os.getenv("AI_API_BASE_URL_1") or os.getenv("AI_LOCAL_URL", "http://127.0.0.1:8080/v1")),
            "freegpt4": ("freegpt4", os.getenv("AI_API_BASE_URL_1", "http://127.0.0.1:5500")),
            "puter": ("puter", os.getenv("AI_API_BASE_URL_1", "http://127.0.0.1:8741/v1")),
            "keylessai": ("keylessai", os.getenv("AI_API_BASE_URL_1", "https://keylessai.thryx.workers.dev/v1")),
            "omniroute": ("omniroute", os.getenv("AI_API_BASE_URL_1", "http://cloud.omniroute.online/v1")),
            "danyapi": ("danyapi", "https://danyapi.cloudpub.ru/v1/"),
            "gemini": ("gemini", os.getenv("AI_API_BASE_URL_1") or "https://generativelanguage.googleapis.com/v1beta"),
            "groq": ("groq", "https://api.groq.com/openai/v1"),
            "openrouter": ("openrouter", "https://openrouter.ai/api/v1"),
            "openai": ("openai", "https://api.openai.com/v1"),
            "nvidia": ("nvidia", "https://integrate.api.nvidia.com/v1"),
            "cerebras": ("cerebras", "https://api.cerebras.ai/v1"),
            "sambanova": ("sambanova", "https://api.sambanova.ai/v1"),
            "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
            "xai": ("xai", "https://api.x.ai/v1"),
            "bytez": ("bytez", "https://api.bytez.com/models/v2/openai/v1"),
        }
        if forced.lower() in forced_map:
            return forced_map[forced.lower()]

    key = (key or "").strip()
    url_env = os.getenv("AI_API_BASE_URL_1") or ""

    # === Gemini (clave AQ. o URL de Google) ===
    if key.startswith("AQ.") or key.startswith("AIza") or "generativelanguage.googleapis.com" in url_env:
        url = url_env or "https://generativelanguage.googleapis.com/v1beta"
        if url.rstrip("/").endswith("/openai"):
            url = url.rstrip("/")[:-7]
        return "gemini", url

    # === OmniRoute / KeylessAI / Puter / FreeGPT4 ===
    if "omniroute" in url_env or "omnirouter" in url_env:
        return "omniroute", url_env
    if key == "not-needed" or "keylessai" in url_env:
        return "keylessai", url_env or "https://keylessai.thryx.workers.dev/v1"
    if key == "subtom123" or ":8741" in url_env:
        return "puter", url_env or "http://127.0.0.1:8741/v1"
    if key == "dummy" or ("trycloudflare.com" in url_env and ":5500" in url_env):
        return "freegpt4", url_env or "http://127.0.0.1:5500"
    if "cloudpub.ru" in url_env:
        return "danyapi", "https://danyapi.cloudpub.ru/v1/"
    if key.startswith("sk-subtom-"):
        return "termux", url_env or "http://127.0.0.1:8080/v1"

    # === Groq ===
    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"

    # === Otros por prefijo ===
    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"
    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"
    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"
    if key.startswith("csk-"):
        return "cerebras", "https://api.cerebras.ai/v1"
    if key.startswith("nvapi-"):
        return "nvidia", "https://integrate.api.nvidia.com/v1"
    if key.startswith("tgp_v1_"):
        return "together", "https://api.together.xyz/v1"
    if key.startswith("di_"):
        return "deepinfra", "https://api.deepinfra.com/v1/openai"
    if key.startswith("sk-proj-") or key.startswith("sk-"):
        return "openai", "https://api.openai.com/v1"

    if len(key) == 36 and key.count("-") == 4:
        return "sambanova", "https://api.sambanova.ai/v1"

    if len(key) == 32 and all(c in "0123456789abcdef" for c in key.lower()):
        return "bytez", "https://api.bytez.com/models/v2/openai/v1"

    if url_env:
        return "openai", url_env

    return "openrouter", "https://openrouter.ai/api/v1"


class FastCache:
    def __init__(self, max_size: int = 150, ttl: float = 60.0):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(messages: list[dict], model: str) -> str:
        raw = json.dumps([messages, model], sort_keys=True, default=str)
        return hashlib.blake2b(raw.encode("utf-8", "ignore"), digest_size=16).hexdigest()

    def get(self, messages: list[dict], model: str) -> dict | None:
        k = self._key(messages, model)
        entry = self._data.get(k)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if expires < time.time():
            self._data.pop(k, None)
            self.misses += 1
            return None
        self._data.move_to_end(k)
        self.hits += 1
        return value

    def set(self, messages: list[dict], model: str, value: dict) -> None:
        k = self._key(messages, model)
        self._data[k] = (time.time() + self.ttl, value)
        self._data.move_to_end(k)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}


class AIConnector:

    REQUEST_TIMEOUT = 600.0

    def __init__(self) -> None:
        self._cooldowns: dict[int, float] = {}
        self._fail_count: dict[int, int] = {}
        self._cursor: int = 0
        self._last_used: int = -1
        self._cache = FastCache(max_size=150, ttl=60.0)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

        print(f"[CONNECTOR] {len(config.ai_slots)} slots configurados:")
        for slot in config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, url = detect_provider(slot.key, forced)
            print(f"  #{slot.index}: {provider} | {slot.model} | {url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                tcp = aiohttp.TCPConnector(
                    limit=20, limit_per_host=10,
                    ttl_dns_cache=300, enable_cleanup_closed=True, force_close=False,
                )
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                    connector=tcp,
                )
            return self._session

    async def close(self) -> None:
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    def system_prompt(self) -> str:
        return (
            "Eres Subtom IA, el asistente personal de Amin. Hablas siempre en español y eres "
            "súper amable, cálido y cercano, como un buen amigo que sabe programar. Te gusta "
            "conversar: das contexto, explicas con detalle, y tus respuestas son largas y "
            "completas, nunca de una línea seca. Usas un tono natural, con humor seco cuando "
            "encaja, sin exagerar con emojis. Eres técnico cuando hace falta pero sin ser "
            "pedante.\n\n"
            "FORMATO DE RESPUESTA: Separa tus ideas en párrafos cortos con líneas en blanco "
            "entre ellos. Usa listas con guiones cuando enumeres cosas. Pon el código en "
            "bloques con ```. No metas todo en un solo bloque de texto.\n\n"
            "Tienes herramientas reales. Úsalas cuando toca. Si no estás seguro de la ruta "
            "de un archivo en GitHub, usa github_search_code o github_tree ANTES de "
            "github_read. Nunca inventes contenido: si una herramienta falla, dilo.\n\n"
            "Eres Subtom, no finjas ser ChatGPT, Claude ni Gemini."
        )

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return time.monotonic()

    def _available_slots(self) -> list[AISlot]:
        now = self._now()
        total = len(config.ai_slots)
        if total == 0:
            return []
        available: list[AISlot] = []
        for offset in range(total):
            i = (self._cursor + offset) % total
            slot = config.ai_slots[i]
            if now >= self._cooldowns.get(slot.index, 0):
                available.append(slot)
        if not available:
            print("[CONNECTOR] Todos en cooldown. Reseteando.")
            self._cooldowns.clear()
            available = list(config.ai_slots)
        available.sort(key=lambda s: self._fail_count.get(s.index, 0))
        return available

    def _mark_failure(self, slot: AISlot, cooldown: float, reason: str) -> None:
        self._cooldowns[slot.index] = self._now() + cooldown
        self._fail_count[slot.index] = self._fail_count.get(slot.index, 0) + 1
        print(f"[CONNECTOR] #{slot.index} → {reason} (cd {cooldown:.0f}s)")

    def _mark_success(self, slot: AISlot) -> None:
        self._cursor = slot.index
        self._fail_count[slot.index] = 0
        if self._last_used != slot.index:
            print(f"[CONNECTOR] Usando slot #{slot.index}")
            self._last_used = slot.index

    # ============================================================
    # GEMINI NATIVO (con FIX error 400 / thought signatures)
    # ============================================================

    @staticmethod
    def _tools_to_gemini(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        declarations = []
        for t in tools:
            fn = t.get("function", {})
            declarations.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": declarations}]

    @staticmethod
    def _messages_to_gemini(messages: list[dict]) -> tuple[dict, list[dict]]:
        system_parts: list[dict] = []
        contents: list[dict] = []

        for m in messages:
            role = m.get("role")
            content = m.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append({"text": content})
                continue

            if role == "tool":
                tool_name = m.get("name") or m.get("tool_call_id") or "tool"
                try:
                    payload = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    payload = {"result": str(content)}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": payload if isinstance(payload, dict) else {"result": payload},
                        }
                    }],
                })
                continue

            if role == "assistant" and m.get("tool_calls"):
                parts = []
                if content:
                    parts.append({"text": content})
                for idx, call in enumerate(m["tool_calls"]):
                    fn = call.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_dict = {}
                    part = {
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": args_dict,
                        }
                    }
                    # FIX ERROR 400: thought_signature obligatoria en la primera tool_call
                    if idx == 0:
                        part["thoughtSignature"] = "skip_thought_signature_validator"
                    parts.append(part)
                contents.append({"role": "model", "parts": parts})
                continue

            g_role = "model" if role == "assistant" else "user"
            if isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64 = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                contents.append({"role": g_role, "parts": parts})
            else:
                contents.append({"role": g_role, "parts": [{"text": content or ""}]})

        return {"parts": system_parts}, contents

    @staticmethod
    def _gemini_response_to_openai(data: dict, model: str) -> dict:
        text = ""
        tool_calls = []
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    text += p["text"]
                if "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                        },
                    })

        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _call_gemini(
        self,
        slot: AISlot,
        messages: list[dict],
        tools: list[dict] | None,
        session: aiohttp.ClientSession,
    ) -> dict:
        base_url = (os.getenv("AI_API_BASE_URL_1") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        if base_url.endswith("/openai"):
            base_url = base_url[:-7]
        model = slot.model or "gemini-flash-lite-latest"
        endpoint = f"{base_url}/models/{model}:generateContent?key={slot.key}"

        system_instruction, contents = self._messages_to_gemini(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction["parts"]:
            payload["systemInstruction"] = system_instruction
        payload["generationConfig"] = {
            "temperature": config.ai_temperature,
            "maxOutputTokens": config.ai_max_tokens,
        }
        gemini_tools = self._tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        async with session.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
        ) as response:
            body = await response.text()

            if response.status == 200:
                self._mark_success(slot)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    raise RuntimeError(f"gemini respuesta no es JSON: {body[:200]}")
                return self._gemini_response_to_openai(data, model)

            if response.status == 429:
                self._mark_failure(slot, 60, "429 quota")
                raise RuntimeError("gemini 429")
            if response.status in (401, 403):
                self._mark_failure(slot, 300, f"{response.status} auth")
                raise RuntimeError(f"gemini {response.status}")
            if response.status == 400:
                self._mark_failure(slot, 30, "400 payload/thought_signature")
                raise RuntimeError(f"gemini 400: {body[:300]}")
            if response.status == 404:
                self._mark_failure(slot, 120, "404 modelo")
                raise RuntimeError("gemini 404")
            if response.status >= 500:
                self._mark_failure(slot, 60, f"{response.status} server")
                raise RuntimeError(f"gemini {response.status}")

            self._mark_failure(slot, 60, f"{response.status}")
            raise RuntimeError(f"gemini {response.status}: {body[:300]}")

    # ============================================================
    # OPENAI-COMPATIBLE
    # ============================================================

    async def _call_openai_compat(
        self,
        slot: AISlot,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        session: aiohttp.ClientSession,
    ) -> dict:
        forced = (
            os.getenv(f"AI_PROVIDER_{slot.index}")
            if slot.index > 0
            else os.getenv("AI_PROVIDER")
        )
        provider, base_url = detect_provider(slot.key, forced)

        payload = {
            "model": slot.model or model or "",
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        async with session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {slot.key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            body = await response.text()

            if response.status == 200:
                self._mark_success(slot)
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    raise RuntimeError(f"{provider} respuesta no es JSON: {body[:200]}")

            if response.status == 429:
                self._mark_failure(slot, 30, "429 rate limit")
                raise RuntimeError(f"{provider} 429")
            if response.status in (401, 403):
                self._mark_failure(slot, 300, f"{response.status} auth")
                raise RuntimeError(f"{provider} {response.status}")
            if response.status == 413:
                self._mark_failure(slot, 60, "413 payload too large")
                raise RuntimeError(f"{provider} 413")
            if response.status == 404:
                self._mark_failure(slot, 120, "404 modelo")
                raise RuntimeError(f"{provider} 404")
            if response.status >= 500:
                self._mark_failure(slot, 60, f"{response.status} server")
                raise RuntimeError(f"{provider} {response.status}")

            self._mark_failure(slot, 60, f"{response.status}")
            raise RuntimeError(f"{provider} {response.status}: {body[:200]}")

    # ============================================================
    # COMPLETE
    # ============================================================

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.system_prompt():
            messages = [{"role": "system", "content": self.system_prompt()}, *messages]

        cache_key_model = model or (config.ai_slots[0].model if config.ai_slots else "")
        if not tools:
            cached = self._cache.get(messages, cache_key_model)
            if cached is not None:
                print("[CONNECTOR] Cache hit")
                return cached

        slots = self._available_slots()
        if not slots:
            raise RuntimeError("No hay slots de IA configurados.")

        session = await self._get_session()
        last_error: Exception | None = None

        for slot in slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, _ = detect_provider(slot.key, forced)
            try:
                if provider == "gemini":
                    result = await self._call_gemini(slot, messages, tools, session)
                else:
                    result = await self._call_openai_compat(slot, messages, tools, model, session)
                if not tools:
                    self._cache.set(messages, cache_key_model, result)
                return result
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                last_error = RuntimeError("timeout")
                continue
            except aiohttp.ClientError as exc:
                self._mark_failure(slot, 30, f"red: {type(exc).__name__}")
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        raise last_error or RuntimeError("Todos los slots fallaron.")

    def clean_tool_result(self, result: Any) -> str:
        try:
            if isinstance(result, str):
                return result[:20000]
            return json.dumps(result, ensure_ascii=False, default=str)[:20000]
        except Exception:
            return str(result)[:20000]

    @property
    def provider(self) -> str:
        if config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{config.ai_slots[0].index}")
                if config.ai_slots[0].index > 0
                else os.getenv("AI_PROVIDER")
            )
            p, _ = detect_provider(config.ai_slots[0].key, forced)
            return p
        return "none"

    @property
    def base_url(self) -> str:
        if config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{config.ai_slots[0].index}")
                if config.ai_slots[0].index > 0
                else os.getenv("AI_PROVIDER")
            )
            _, u = detect_provider(config.ai_slots[0].key, forced)
            return u
        return ""

    def stats(self) -> dict[str, Any]:
        now = self._now()
        result = []
        for slot in config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, _ = detect_provider(slot.key, forced)
            cd = self._cooldowns.get(slot.index, 0)
            result.append({
                "slot": slot.index,
                "provider": provider,
                "model": slot.model,
                "fallos": self._fail_count.get(slot.index, 0),
                "cooldown_restante_s": max(0, round(cd - now, 1)),
                "activo": now >= cd,
            })
        return {"cursor_actual": self._cursor, "cache": self._cache.stats(), "slots": result}


connector = AIConnector()
