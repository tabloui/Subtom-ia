from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from config import config, AISlot


class AIConnector:

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

        print(f"[CONNECTOR] {len(config.ai_slots)} slot(s) configurado(s):")
        for slot in config.ai_slots:
            print(f"  #{slot.index}: local | {slot.model} | {config.ai_local_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                tcp = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    force_close=False,
                )
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=config.ai_timeouts_seconds),
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
            "Eres Subtom IA, el asistente personal de Amin. Hablas siempre en español. "
            "Eres directo, técnico y conciso. No inventas información. "
            "Si no sabes algo, lo dices. "
            "Tienes herramientas disponibles: web_search, file_read, file_write, "
            "github_read, github_write, sandbox_run_python, discord_send_message. "
            "Úsalas solo cuando sea necesario. Si no puedes hacer algo, dilo claro."
        )

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

        if not config.ai_slots:
            raise RuntimeError("No hay slot de IA configurado.")

        slot = config.ai_slots[0]
        model_use = slot.model or model or "qwen"

        payload: dict[str, Any] = {
            "model": model_use,
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
            "stream": False,
        }

        session = await self._get_session()
        t0 = time.time()

        try:
            async with session.post(
                f"{config.ai_local_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {slot.key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                body = await response.text()
                dt = time.time() - t0

                print(
                    f"[CONNECTOR] local respondio en {dt:.1f}s "
                    f"(status {response.status})"
                )

                if response.status == 200:
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Respuesta invalida de local: {body[:300]}"
                        ) from exc

                if response.status == 429:
                    raise RuntimeError(
                        f"local 429 (rate limit): {body[:300]}"
                    )
                if response.status in (401, 403):
                    raise RuntimeError(
                        f"local {response.status} auth: {body[:300]}"
                    )
                if response.status == 404:
                    raise RuntimeError(
                        f"local 404 modelo '{model_use}': {body[:300]}"
                    )
                if response.status == 413:
                    raise RuntimeError(
                        f"local 413 prompt demasiado grande: {body[:300]}"
                    )
                if response.status >= 500:
                    raise RuntimeError(
                        f"local {response.status} server: {body[:300]}"
                    )

                raise RuntimeError(
                    f"local {response.status}: {body[:300]}"
                )

        except aiohttp.ClientError as exc:
            dt = time.time() - t0
            print(f"[CONNECTOR] local error de red tras {dt:.1f}s: {exc}")
            raise RuntimeError(
                f"Error de conexion con local: {exc}"
            ) from exc

        except asyncio.TimeoutError as exc:
            dt = time.time() - t0
            print(f"[CONNECTOR] local timeout tras {dt:.1f}s")
            raise RuntimeError(
                f"Timeout esperando a local "
                f"({config.ai_timeouts_seconds}s)"
            ) from exc

    def clean_tool_result(self, result: Any) -> str:
        try:
            if isinstance(result, str):
                return result[:8000]
            return json.dumps(
                result, ensure_ascii=False, default=str
            )[:8000]
        except Exception:
            return str(result)[:8000]

    @property
    def provider(self) -> str:
        return "local"

    @property
    def base_url(self) -> str:
        return config.ai_local_url

    def stats(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "url": config.ai_local_url,
            "slots": [
                {
                    "slot": s.index,
                    "provider": "local",
                    "model": s.model,
                    "activo": True,
                }
                for s in config.ai_slots
            ],
        }


connector = AIConnector()
