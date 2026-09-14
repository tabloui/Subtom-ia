from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from config import config


class AIConnector:

    def __init__(self) -> None:
        self.base_url = "https://api.cerebras.ai/v1"
        self._cooldown_until: float = 0.0

    def system_prompt(self) -> str:
        return (
            "Subtom IA de Amin. Amable, gracioso, cercano. "
            "Sin exceso de emojis. Español. Directo y conciso. "
            "Tienes herramientas (web, archivos, GitHub, Vercel, "
            "imágenes); úsalas cuando hagan falta. No inventes."
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {config.cerebras_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return asyncio.get_event_loop().time()

    def _in_cooldown(self) -> bool:
        return self._now() < self._cooldown_until

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

        modelo_usar = model or config.ai_model

        payload: dict[str, Any] = {
            "model": modelo_usar,
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if self._in_cooldown():
            print("[CONNECTOR] Cooldown activo, reintentando igual.")
            self._cooldown_until = 0.0

        timeout = aiohttp.ClientTimeout(total=config.ai_timeouts_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:

                    body = await response.text()

                    if response.status == 429:
                        self._cooldown_until = self._now() + 10
                        raise RuntimeError(
                            f"Cerebras HTTP 429 (modelo {modelo_usar}): "
                            f"{body[:500]}"
                        )

                    if response.status in (401, 403):
                        self._cooldown_until = self._now() + 30
                        raise RuntimeError(
                            f"Cerebras HTTP {response.status}: {body[:500]}"
                        )

                    if response.status == 404:
                        raise RuntimeError(
                            f"Cerebras HTTP 404 (modelo {modelo_usar}): "
                            f"{body[:500]}"
                        )

                    if response.status == 413:
                        raise RuntimeError(
                            f"Cerebras HTTP 413 (petición demasiado grande): "
                            f"{body[:500]}"
                        )

                    if response.status >= 500:
                        raise RuntimeError(
                            f"Cerebras HTTP {response.status} "
                            f"(modelo {modelo_usar}): {body[:500]}"
                        )

                    if 200 <= response.status < 300:
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"Respuesta inválida de Cerebras: {body[:300]}"
                            ) from exc

                    raise RuntimeError(
                        f"Cerebras HTTP {response.status} "
                        f"(modelo {modelo_usar}): {body[:500]}"
                    )

            except aiohttp.ClientError as exc:
                raise RuntimeError(
                    f"Error de conexión con Cerebras: {exc}"
                ) from exc

    def clean_tool_result(self, result: Any) -> str:
        try:
            if isinstance(result, str):
                return result[:20000]
            return json.dumps(
                result, ensure_ascii=False, default=str
            )[:20000]
        except Exception:
            return str(result)[:20000]


connector = AIConnector()
