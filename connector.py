from __future__ import annotations
import asyncio
import json
from typing import Any

import aiohttp

from config import config


class AIConnector:
    """Conector para el agente vía OpenRouter, gestionando las claves."""

    def __init__(self) -> None:
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self._key_index = 0
        # Memoria temporal (timestamp) de las peticiones para rate-limit.
        self._cooldowns: dict[int, float] = {}

    def system_prompt(self, api_key: str) -> str:
        text = config.bot_system_text if config.bot_system_text else ""
        return f"""SISTEMA DE INSTRUCCIONES:
{text}

RESPUESTA:
{config.bot_name}, un asistente IA."""

    async def _request(self, api_key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def _call_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        payload = {
            "model": config.ai_model,
            "messages": [
                {"role": "system", "content": self.system_prompt("")},
                *messages,
            ],
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

        timeout = aiohttp.ClientTimeout(total=config.ai_timeouts_seconds)
        last_error: Exception | None = None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for idx, api_key in enumerate(keys):
                try:
                    async with session.post(
                        self.base_url,
                        headers=self._headers(api_key),
                        json=payload,
                    ) as response:
                        body = await response.text()

                        # 429: Rate limit, guardar cooldown.
                        if response.status == 429:
                            self._cooldowns[idx] = asyncio.get_event_loop().time() + 60
                            last_error = RuntimeError(
                                f"OpenRouter HTTP 429 (clave #{idx + 1}): {body[:500]}"
                            )
                            continue
                        # 401/403: Clave inválida, quitar del pool.
                        if response.status in (401, 403):
                            self._cooldowns[idx] = asyncio.get_event_loop().time() + 60
                            last_error = RuntimeError(
                                f"OpenRouter HTTP {response.status} (clave #{idx + 1}): {body[:500]}"
                            )
                            continue
                        # 200: Éxito.
                        if response.status >= 200 and response.status < 300:
                            self._key_index = idx
                            return json.loads(body)

                        last_error = RuntimeError(
                            f"OpenRouter HTTP {response.status}: {body[:500]}"
                        )
                except aiohttp.ClientError as exc:
                    last_error = exc
                    continue

        raise last_error or RuntimeError(
            "No se pudo conectar con OpenRouter."
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://rallywr.app/",
            "X-Title": config.bot_name,
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        payload = {
            "model": config.ai_model,
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

        timeout = aiohttp.ClientTimeout(total=config.ai_timeouts_seconds)
        last_error: Exception | None = None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for idx, api_key in enumerate(keys):
                try:
                    async with session.post(
                        self.base_url,
                        headers=self._headers(api_key),
                        json=payload,
                    ) as response:
                        body = await response.text()

                        if response.status == 429:
                            self._cooldowns[idx] = asyncio.get_event_loop().time() + 60
                            last_error = RuntimeError(
                                f"OpenRouter HTTP 429 (clave #{idx + 1}): {body[:500]}"
                            )
                            continue
                        if response.status in (401, 403):
                            self._cooldowns[idx] = asyncio.get_event_loop().time() + 60
                            last_error = RuntimeError(
                                f"OpenRouter HTTP {response.status} (clave #{idx + 1}): {body[:500]}"
                            )
                            continue
                        if response.status >= 200 and response.status < 300:
                            self._key_index = idx
                            return json.loads(body)

                        last_error = RuntimeError(
                            f"OpenRouter HTTP {response.status}: {body[:500]}"
                        )
                except aiohttp.ClientError as exc:
                    last_error = exc
                    continue

        raise last_error or RuntimeError(
            "No se pudo conectar con OpenRouter."
        )

    def _available_keys(self) -> list[str]:
        """Devuelve las claves disponibles, rotando para distribuir el uso."""
        now = asyncio.get_event_loop().time()
        total = len(config.openrouter_api_keys)
        result: list[str] = []
        for offset in range(total):
            idx = (self._key_index + offset) % total
            cooldown = self._cooldowns.get(idx, 0)
            if now >= cooldown:
                result.append(config.openrouter_api_keys[idx])
        return result


connector = AIConnector()
