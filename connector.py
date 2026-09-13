from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from config import config


class AIConnector:
    """Conector para Groq (API compatible con OpenAI)."""

    def __init__(self) -> None:
        self.base_url = config.groq_base_url.rstrip("/")
        self._cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        parts: list[str] = []

        if config.bot_identity:
            parts.append(f"# Identidad\n{config.bot_identity}")

        parts.append(f"# Nombre\n{config.bot_name}")

        if config.bot_language:
            parts.append(
                f"# Idioma\nResponde siempre en {config.bot_language}."
            )

        if config.bot_personality:
            parts.append(f"# Personalidad\n{config.bot_personality}")

        if config.bot_style:
            parts.append(f"# Estilo\n{config.bot_style}")

        if config.bot_creator:
            parts.append(f"# Creador\n{config.bot_creator}")

        if config.bot_capabilities:
            parts.append(f"# Capacidades\n{config.bot_capabilities}")

        if config.user_name or config.user_description:
            user_block = "# Usuario"
            if config.user_name:
                user_block += f"\nNombre: {config.user_name}"
            if config.user_description:
                user_block += f"\nDescripción: {config.user_description}"
            parts.append(user_block)

        if config.bot_system_text:
            parts.append(f"# Instrucciones\n{config.bot_system_text}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # HEADERS
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # UTILIDAD DE TIEMPO
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return asyncio.get_event_loop().time()

    def _in_cooldown(self) -> bool:
        return self._now() < self._cooldown_until

    # ------------------------------------------------------------------
    # LLAMADA PRINCIPAL
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        # Inyectar system prompt si no hay uno
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

        # Si está en cooldown, resetear (con Groq el cooldown es corto)
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

                    # Rate limit
                    if response.status == 429:
                        self._cooldown_until = self._now() + 10
                        raise RuntimeError(
                            f"Groq HTTP 429 (modelo {modelo_usar}): "
                            f"{body[:500]}"
                        )

                    # Clave inválida
                    if response.status in (401, 403):
                        self._cooldown_until = self._now() + 30
                        raise RuntimeError(
                            f"Groq HTTP {response.status}: {body[:500]}"
                        )

                    # Modelo no existe o fue deprecado
                    if response.status == 404:
                        raise RuntimeError(
                            f"Groq HTTP 404 (modelo {modelo_usar}): "
                            f"{body[:500]}"
                        )

                    # Error del servidor
                    if response.status >= 500:
                        raise RuntimeError(
                            f"Groq HTTP {response.status} "
                            f"(modelo {modelo_usar}): {body[:500]}"
                        )

                    if 200 <= response.status < 300:
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"Respuesta inválida de Groq: {body[:300]}"
                            ) from exc

                    raise RuntimeError(
                        f"Groq HTTP {response.status} "
                        f"(modelo {modelo_usar}): {body[:500]}"
                    )

            except aiohttp.ClientError as exc:
                raise RuntimeError(
                    f"Error de conexión con Groq: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # UTILIDAD
    # ------------------------------------------------------------------

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
