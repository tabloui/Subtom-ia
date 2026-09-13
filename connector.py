from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from config import config


class AIConnector:
    """Conector para el agente vía OpenRouter, gestionando las claves."""

    def __init__(self) -> None:
        self.base_url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )
        self._key_index = 0
        self._cooldowns: dict[int, float] = {}

    # ------------------------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        """
        Construye el system prompt usando TODO lo que hay en config:
        identidad, personalidad, estilo, creador, capacidades,
        usuario e instrucciones libres.
        """

        parts: list[str] = []

        if config.bot_identity:
            parts.append(
                f"# Identidad\n{config.bot_identity}"
            )

        parts.append(f"# Nombre\n{config.bot_name}")

        if config.bot_language:
            parts.append(
                f"# Idioma\nResponde siempre en "
                f"{config.bot_language}."
            )

        if config.bot_personality:
            parts.append(
                f"# Personalidad\n"
                f"{config.bot_personality}"
            )

        if config.bot_style:
            parts.append(
                f"# Estilo\n{config.bot_style}"
            )

        if config.bot_creator:
            parts.append(
                f"# Creador\n{config.bot_creator}"
            )

        if config.bot_capabilities:
            parts.append(
                f"# Capacidades\n"
                f"{config.bot_capabilities}"
            )

        if config.user_name or config.user_description:
            user_block = "# Usuario"

            if config.user_name:
                user_block += f"\nNombre: {config.user_name}"

            if config.user_description:
                user_block += (
                    f"\nDescripción: {config.user_description}"
                )

            parts.append(user_block)

        if config.bot_system_text:
            parts.append(
                f"# Instrucciones\n"
                f"{config.bot_system_text}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # HEADERS
    # ------------------------------------------------------------------

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://railway.app/",
            "X-Title": config.bot_name,
        }

    # ------------------------------------------------------------------
    # ROTACIÓN DE CLAVES
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        """Tiempo monotónico (evita deprecación de get_event_loop)."""
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return asyncio.get_event_loop().time()

    def _available_keys(self) -> list[tuple[int, str]]:
        """
        Devuelve lista de (índice_real, clave) que NO están en cooldown,
        empezando por la última usada con éxito.
        """

        now = self._now()
        total = len(config.openrouter_api_keys)

        result: list[tuple[int, str]] = []

        for offset in range(total):

            idx = (self._key_index + offset) % total

            cooldown = self._cooldowns.get(idx, 0)

            if now >= cooldown:
                result.append(
                    (idx, config.openrouter_api_keys[idx])
                )

        return result

    # ------------------------------------------------------------------
    # LLAMADA PRINCIPAL
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        # ----------------------------------------------------------
        # Inyectar system prompt si no hay uno.
        # ----------------------------------------------------------

        has_system = any(
            m.get("role") == "system"
            for m in messages
        )

        if not has_system:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt(),
                },
                *messages,
            ]

        # ----------------------------------------------------------
        # Modelo: el que pase ai.py, o el de config por defecto.
        # ----------------------------------------------------------

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

        keys = self._available_keys()

        if not keys:
            raise RuntimeError(
                "No hay claves de API disponibles para OpenRouter."
            )

        timeout = aiohttp.ClientTimeout(
            total=config.ai_timeouts_seconds
        )

        last_error: Exception | None = None

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            for real_idx, api_key in keys:

                try:

                    async with session.post(
                        self.base_url,
                        headers=self._headers(api_key),
                        json=payload,
                    ) as response:

                        body = await response.text()

                        # --- Rate limit ---
                        if response.status == 429:
                            self._cooldowns[real_idx] = (
                                self._now() + 60
                            )
                            last_error = RuntimeError(
                                f"OpenRouter HTTP 429 "
                                f"(clave #{real_idx + 1}, "
                                f"modelo {modelo_usar}): "
                                f"{body[:500]}"
                            )
                            continue

                        # --- Clave inválida / sin permiso ---
                        if response.status in (401, 403):
                            self._cooldowns[real_idx] = (
                                self._now() + 300
                            )
                            last_error = RuntimeError(
                                f"OpenRouter HTTP "
                                f"{response.status} "
                                f"(clave #{real_idx + 1}): "
                                f"{body[:500]}"
                            )
                            continue

                        # --- Modelo no disponible / error 404 ---
                        # No penalizamos la clave: es problema del modelo.
                        if response.status == 404:
                            raise RuntimeError(
                                f"OpenRouter HTTP 404 "
                                f"(modelo {modelo_usar}): "
                                f"{body[:500]}"
                            )

                        # --- Éxito ---
                        if 200 <= response.status < 300:
                            self._key_index = real_idx
                            return json.loads(body)

                        # --- Otro error: no penalizamos clave ---
                        last_error = RuntimeError(
                            f"OpenRouter HTTP "
                            f"{response.status} "
                            f"(modelo {modelo_usar}): "
                            f"{body[:500]}"
                        )
                        continue

                except aiohttp.ClientError as exc:
                    last_error = exc
                    continue

        raise last_error or RuntimeError(
            "No se pudo conectar con OpenRouter."
        )

    # ------------------------------------------------------------------
    # UTILIDAD
    # ------------------------------------------------------------------

    def clean_tool_result(self, result: Any) -> str:
        """Convierte un resultado de tool a string seguro para el modelo."""
        try:
            if isinstance(result, str):
                return result[:20000]

            return json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )[:20000]

        except Exception:
            return str(result)[:20000]


# Instancia global.
connector = AIConnector()
