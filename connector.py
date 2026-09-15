from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from config import config, AISlot


def detect_provider(key: str) -> tuple[str, str]:
    key = (key or "").strip()

    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"

    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"

    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"

    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"

    if key.startswith("AQ."):
        return "gemini", "https://generativelanguage.googleapis.com/v1beta/openai"

    if key.startswith("AIza"):
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


class AIConnector:

    def __init__(self) -> None:
        self._cooldowns: dict[int, float] = {}
        self._fail_count: dict[int, int] = {}
        self._cursor: int = 0
        self._last_used: int = -1

        print(f"[CONNECTOR] {len(config.ai_slots)} slots configurados:")
        for slot in config.ai_slots:
            provider, _ = detect_provider(slot.key)
            print(f"  #{slot.index}: {provider} | {slot.model}")

    def system_prompt(self) -> str:
        return (
            "Subtom IA de Amin. Amable, gracioso, cercano. "
            "Sin exceso de emojis. Español. Directo y conciso. "
            "Tienes herramientas (web, archivos, GitHub, Vercel, "
            "imágenes); úsalas cuando hagan falta. No inventes."
        )

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return asyncio.get_event_loop().time()

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
            print(
                "[CONNECTOR] Todos los slots en cooldown. Reseteando."
            )
            self._cooldowns.clear()
            available = list(config.ai_slots)

        available.sort(key=lambda s: self._fail_count.get(s.index, 0))
        return available

    def _mark_failure(self, slot: AISlot, cooldown: float, reason: str) -> None:
        self._cooldowns[slot.index] = self._now() + cooldown
        self._fail_count[slot.index] = self._fail_count.get(slot.index, 0) + 1
        provider, _ = detect_provider(slot.key)
        print(
            f"[CONNECTOR] #{slot.index} {provider} → {reason} "
            f"(cooldown {cooldown:.0f}s, "
            f"fallos: {self._fail_count[slot.index]})"
        )

    def _mark_success(self, slot: AISlot) -> None:
        self._cursor = slot.index
        self._fail_count[slot.index] = 0
        if self._last_used != slot.index:
            provider, _ = detect_provider(slot.key)
            print(
                f"[CONNECTOR] Usando slot #{slot.index} ({provider})"
            )
            self._last_used = slot.index

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

        slots = self._available_slots()
        if not slots:
            raise RuntimeError("No hay slots de IA configurados.")

        last_error: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=config.ai_timeouts_seconds)

        for slot in slots:
            provider, base_url = detect_provider(slot.key)
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

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
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
                            last_error = RuntimeError(
                                f"{provider} 429: {body[:300]}"
                            )
                            continue

                        if response.status in (401, 403):
                            self._mark_failure(
                                slot, 300, f"{response.status} auth"
                            )
                            last_error = RuntimeError(
                                f"{provider} {response.status}: {body[:300]}"
                            )
                            continue

                        if response.status == 402:
                            self._mark_failure(
                                slot, 3600, "402 pago requerido"
                            )
                            last_error = RuntimeError(
                                f"{provider} 402: {body[:300]}"
                            )
                            continue

                        if response.status == 404:
                            self._mark_failure(
                                slot, 120, "404 modelo no existe"
                            )
                            last_error = RuntimeError(
                                f"{provider} 404: {body[:300]}"
                            )
                            continue

                        if response.status == 413:
                            self._mark_failure(
                                slot, 60, "413 payload grande"
                            )
                            last_error = RuntimeError(
                                f"{provider} 413: {body[:300]}"
                            )
                            continue

                        if response.status >= 500:
                            self._mark_failure(
                                slot, 60, f"{response.status} server error"
                            )
                            last_error = RuntimeError(
                                f"{provider} {response.status}: {body[:300]}"
                            )
                            continue

                        self._mark_failure(slot, 60, f"{response.status}")
                        last_error = RuntimeError(
                            f"{provider} {response.status}: {body[:300]}"
                        )
                        continue

            except aiohttp.ClientError as exc:
                self._mark_failure(
                    slot, 30, f"error red: {type(exc).__name__}"
                )
                last_error = exc
                continue

            except asyncio.TimeoutError as exc:
                self._mark_failure(slot, 60, "timeout")
                last_error = exc
                continue

        raise last_error or RuntimeError("Todos los slots fallaron.")

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
            p, _ = detect_provider(config.ai_slots[0].key)
            return p
        return "none"

    @property
    def base_url(self) -> str:
        if config.ai_slots:
            _, u = detect_provider(config.ai_slots[0].key)
            return u
        return ""

    def stats(self) -> dict[str, Any]:
        now = self._now()
        result = []
        for slot in config.ai_slots:
            provider, _ = detect_provider(slot.key)
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
            "slots": result,
        }


connector = AIConnector()
