from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
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
            "sambanova": ("sambanova", "https://api.sambanova.ai/v1"),
            "groq": ("groq", "https://api.groq.com/openai/v1"),
            "openrouter": ("openrouter", "https://openrouter.ai/api/v1"),
            "gemini": ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
            "openai": ("openai", "https://api.openai.com/v1"),
            "nvidia": ("nvidia", "https://integrate.api.nvidia.com/v1"),
            "cerebras": ("cerebras", "https://api.cerebras.ai/v1"),
            "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
            "xai": ("xai", "https://api.x.ai/v1"),
            "puter": ("puter", "https://api.puter.com/puterai/openai/v1"),
            "bytez": ("bytez", "https://api.bytez.com/models/v2/openai/v1"),
        }
        if forced.lower() in forced_map:
            return forced_map[forced.lower()]

    key = (key or "").strip()

    # === TERMUX (tu servidor local vía Cloudflare Tunnel) ===
    if key.startswith("sk-subtom-"):
        url = os.getenv("AI_API_BASE_URL_1") or os.getenv("AI_LOCAL_URL", "http://127.0.0.1:8080/v1")
        return "termux", url

    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"
    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"
    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"
    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"
    if key.startswith("AQ.") or key.startswith("AIza"):
        return "gemini", "https://generativelanguage.googleapis.com/v1beta/openai"
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

    # SambaNova: UUID
    if len(key) == 36 and key.count("-") == 4:
        return "sambanova", "https://api.sambanova.ai/v1"

    # Puter: JWT
    if key.startswith("eyJ") and key.count(".") == 2:
        return "puter", "https://api.puter.com/puterai/openai/v1"

    # Bytez: clave hex de 32 caracteres
    if len(key) == 32 and all(c in "0123456789abcdef" for c in key.lower()):
        return "bytez", "https://api.bytez.com/models/v2/openai/v1"

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
        return hashlib.blake2b(
            raw.encode("utf-8", "ignore"), digest_size=16
        ).hexdigest()

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
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._data),
        }


class AIConnector:

    REQUEST_TIMEOUT = 90.0

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
                    limit=20,
                    limit_per_host=10,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                    force_close=False,
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
            "pedante. Cuando el usuario te pide algo, lo haces bien y con ganas.\n\n"
            "Tienes herramientas reales y las usas cuando toca:\n"
            "- web_search y web_fetch para buscar y leer internet\n"
            "- file_read, file_write, file_tree, file_grep y demás para archivos\n"
            "- file_read_pdf para PDFs, file_count_words, file_stats\n"
            "- github_* para gestionar repos, issues, PRs, branches, commits, archivos\n"
            "- vercel_* para proyectos, deploys, envs\n"
            "- sandbox_run_python, sandbox_run_shell, sandbox_run_node para ejecutar código\n"
            "- generate_image para generar imágenes con FLUX\n"
            "- discord_* para mensajes, encuestas, DMs, moderación, roles, canales\n"
            "- discord_send_file para enviar archivos al chat\n\n"
            "Cuando escribas código, que sea completo y funcional, no fragmentos. "
            "Si ves un problema, dilo con claridad. Si algo es buena idea, reconócelo. "
            "Nunca inventes información. Si no sabes algo, lo buscas o lo dices.\n\n"
            "REGLA CRÍTICA DE AUTO-MODIFICACIÓN: cuando modifiques tu propio código "
            "(connector.py, bot.py, ai.py, motor.py, config.py, file_tools.py, sandbox.py), "
            "lee el archivo completo con github_read, escríbelo entero en local con file_write, "
            "verifica con sandbox_run_python usando py_compile que compila sin errores, "
            "y compara el número de líneas con el original. Solo si TODO está OK, sube con "
            "github_write. Si tienes dudas, para y pregunta.\n\n"
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
        forced = (
            os.getenv(f"AI_PROVIDER_{slot.index}")
            if slot.index > 0
            else os.getenv("AI_PROVIDER")
        )
        provider, _ = detect_provider(slot.key, forced)
        print(
            f"[CONNECTOR] #{slot.index} {provider} → {reason} "
            f"(cd {cooldown:.0f}s, fallos: {self._fail_count[slot.index]})"
        )

    def _mark_success(self, slot: AISlot) -> None:
        self._cursor = slot.index
        self._fail_count[slot.index] = 0
        if self._last_used != slot.index:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, _ = detect_provider(slot.key, forced)
            print(f"[CONNECTOR] Usando slot #{slot.index} ({provider})")
            self._last_used = slot.index

    async def _call_slot(
        self,
        slot: AISlot,
        payload: dict,
        session: aiohttp.ClientSession,
    ) -> dict:
        forced = (
            os.getenv(f"AI_PROVIDER_{slot.index}")
            if slot.index > 0
            else os.getenv("AI_PROVIDER")
        )
        provider, base_url = detect_provider(slot.key, forced)

        if provider == "gemini":
            payload = dict(payload)
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        async with session.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {slot.key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            body = await response.text()

            if response.status == 200:
                self._mark_success(slot)
                return json.loads(body)

            if response.status == 429:
                self._mark_failure(slot, 30, "429 rate limit")
                raise RuntimeError(f"{provider} 429")
            if response.status in (401, 403):
                self._mark_failure(slot, 300, f"{response.status} auth")
                raise RuntimeError(f"{provider} {response.status}")
            if response.status == 402:
                self._mark_failure(slot, 3600, "402 pago")
                raise RuntimeError(f"{provider} 402")
            if response.status == 404:
                self._mark_failure(slot, 120, "404 modelo")
                raise RuntimeError(f"{provider} 404")
            if response.status == 413:
                self._mark_failure(slot, 60, "413 payload")
                raise RuntimeError(f"{provider} 413")
            if response.status >= 500:
                self._mark_failure(slot, 60, f"{response.status} server")
                raise RuntimeError(f"{provider} {response.status}")

            self._mark_failure(slot, 60, f"{response.status}")
            raise RuntimeError(f"{provider} {response.status}: {body[:200]}")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [
                {"role": "system", "content": self.system_prompt()},
                *messages,
            ]

        cache_key_model = model or (
            config.ai_slots[0].model if config.ai_slots else ""
        )
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
            payload = self._build_payload(slot, messages, tools, model)
            try:
                result = await self._call_slot(slot, payload, session)
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

    def _build_payload(
        self,
        slot: AISlot,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
    ) -> dict:
        model_use = slot.model or model or ""
        payload: dict[str, Any] = {
            "model": model_use,
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def clean_tool_result(self, result: Any) -> str:
        try:
            if isinstance(result, str):
                return result[:20000]
            return json.dumps(
                result, ensure_ascii=False, default=str
            )[:20000]
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
        return {
            "cursor_actual": self._cursor,
            "cache": self._cache.stats(),
            "slots": result,
        }


connector = AIConnector()
