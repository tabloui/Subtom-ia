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
        forced = forced.lower().strip()
        mapping = {
            "ollama": ("ollama", "https://ollama.com/v1"),
            "together": ("together", "https://api.together.xyz/v1"),
            "deepinfra": ("deepinfra", "https://api.deepinfra.com/v1/openai"),
            "cerebras": ("cerebras", "https://api.cerebras.ai/v1"),
            "nvidia": ("nvidia", "https://integrate.api.nvidia.com/v1"),
            "gemini": ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
            "groq": ("groq", "https://api.groq.com/openai/v1"),
            "openrouter": ("openrouter", "https://openrouter.ai/api/v1"),
            "openai": ("openai", "https://api.openai.com/v1"),
            "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
            "xai": ("xai", "https://api.x.ai/v1"),
        }
        if forced in mapping:
            return mapping[forced]

    key = (key or "").strip()

    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"
    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"
    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"
    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"
    if key.startswith("AI.") or key.startswith("AIza"):
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

    REQUEST_TIMEOUT = 25.0

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
            forced = os.getenv(f"AI_PROVIDER_{slot.index}") if slot.index > 0 else os.getenv("AI_PROVIDER")
            provider, _ = detect_provider(slot.key, forced)
            print(f"  #{slot.index}: {provider} | {slot.model}")

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
            "Eres Subtom IA, el asistente personal de Amin. "
            "Hablas siempre en español y eres súper amable, cálido y cercano, "
            "como un buen amigo que sabe programar. "
            "Te gusta conversar: das contexto, explicas con detalle, "
            "y tus respuestas son largas y completas, nunca de una línea seca. "
            "Usas un tono natural, con humor seco cuando encaja, sin exagerar con emojis. "
            "Eres técnico cuando hace falta pero sin ser pedante. "
            "Cuando el usuario te pide algo, lo haces bien y con ganas. "
            "Tienes herramientas reales (web, archivos, GitHub, Vercel, imágenes, sandbox, Discord) "
            "y las usas cuando toca. "
            "No inventas información. "
            "Si no sabes algo, lo dices o lo buscas. "
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
        forced = os.getenv(f"AI_PROVIDER_{slot.index}") if slot.index > 0 else os.getenv("AI_PROVIDER")
        provider, _ = detect_provider(slot.key, forced)
        print(f"[CONNECTOR] #{slot.index} {provider} ⚡ {reason} "
              f"(cd {cooldown:.0f}s, fallos: {self._fail_count[slot.index]})")

    def _mark_success(self, slot: AISlot) -> None:
        self._cursor = slot.index
        self._fail_count[slot.index] = 0
        if self._last_used != slot.index:
            forced = os.getenv(f"AI_PROVIDER_{slot.index}") if slot.index > 0 else os.getenv("AI_PROVIDER")
            provider, _ = detect_provider(slot.key, forced)
            print(f"[CONNECTOR] Usando slot #{slot.index} ({provider})")
            self._last_used = slot.index

    async def _call_slot(
        self,
        slot: AISlot,
        payload: dict,
        session: aiohttp.ClientSession,
    ) -> dict:
        forced = os.getenv(f"AI_PROVIDER_{slot.index}") if slot.index > 0 else os.getenv("AI_PROVIDER")
        provider, base_url = detect_provider(slot.key, forced)

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

        # Fase 1: race entre los 2 primeros
        if len(slots) >= 2:
            first_two = slots[:2]
            tasks = []
            for slot in first_two:
                payload = self._build_payload(slot, messages, tools, model)
                task = asyncio.create_task(self._call_slot(slot, payload, session))
                tasks.append((slot, task))

            try:
                done, pending = await asyncio.wait(
                    [t for _, t in tasks],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.REQUEST_TIMEOUT,
                )

                for _, t in tasks:
                    if not t.done():
                        t.cancel()

                for slot, t in tasks:
                    if t.done() and not t.cancelled() and t.exception() is None:
                        result = t.result()
                        if not tools:
                            self._cache.set(messages, cache_key_model, result)
                        return result
                    elif t.done() and t.exception() is not None:
                        last_error = t.exception()

                for _, t in tasks:
                    try:
                        t.cancel()
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                for _, t in tasks:
                    if not t.done():
                        t.cancel()
                last_error = RuntimeError("Race timeout")

        # Fase 2: resto en secuencia
        start_index = 2 if len(slots) >= 2 else 0
        for slot in slots[start_index:]:
            payload = self._build_payload(slot, messages, tools, model)
            try:
                result = await asyncio.call_slot(slot, payload, session) if hasattr(asyncio, 'call_slot') else await self._call_slot(slot, payload, session)
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
