from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _required(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"Falta la variable {name}")
    return value


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un entero") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


@dataclass(frozen=True)
class AISlot:
    index: int
    key: str
    model: str


@dataclass(frozen=True)
class Settings:
    discord_token: str
    allowed_user_id: int
    database_url: str
    ai_slots: list[AISlot]
    ai_temperature: float
    ai_max_tokens: int
    ai_max_tool_rounds: int
    ai_timeouts_seconds: int
    bot_name: str
    memory_limit: int
    memory_messages: int
    workspace: str
    port: int
    github_token: str | None
    vercel_token: str | None
    flux_api_key_1: str | None
    flux_api_key_2: str | None
    flux_base_url: str
    flux_endpoint: str

    @property
    def ai_model(self) -> str:
        if self.ai_slots:
            return self.ai_slots[0].model
        return "unknown"


def _load_slots(default_model: str) -> list[AISlot]:
    slots: list[AISlot] = []

    main_key = _env("AI_API_KEY")
    if main_key:
        slots.append(AISlot(index=0, key=main_key, model=default_model))

    for n in range(1, 21):
        key = _env(f"AI_API_KEY_{n}")
        if not key:
            continue
        model = _env(f"AI_MODEL_{n}", default_model)
        slots.append(AISlot(index=n, key=key, model=model))

    return slots


def load_settings() -> Settings:

    discord_token = _required("DISCORD_BOT_TOKEN")

    try:
        allowed_user_id = int(_required("DISCORD_ALLOWED_USER_ID"))
    except ValueError as exc:
        raise RuntimeError("DISCORD_ALLOWED_USER_ID debe ser numérico") from exc

    database_url = _required("DATABASE_URL")

    default_model = _env("AI_MODEL", "qwen/qwen3.6-27b")
    slots = _load_slots(default_model)

    if not slots:
        raise RuntimeError(
            "Configura AI_API_KEY o AI_API_KEY_1, _2, _3..."
        )

    temperature_raw = _env("AI_TEMPERATURE", "0.7")
    try:
        ai_temperature = float(temperature_raw)
    except ValueError as exc:
        raise RuntimeError("AI_TEMPERATURE debe ser un número") from exc

    return Settings(
        discord_token=discord_token,
        allowed_user_id=allowed_user_id,
        database_url=database_url,
        ai_slots=slots,
        ai_temperature=ai_temperature,
        ai_max_tokens=_int("AI_MAX_TOKENS", 5000, 256, 16000),
        ai_max_tool_rounds=_int("MAX_TOOL_ROUNDS", 10, 1, 30),
        ai_timeouts_seconds=_int("AI_TIMEOUT_SECONDS", 90, 10, 300),
        bot_name=_env("BOT_NAME", "Subtom"),
        memory_limit=_int("MEMORY_LIMIT", 30, 0, 10080),
        memory_messages=_int("MEMORY_MESSAGES", 20, 4, 200),
        workspace=_env("SUBTOM_WORKSPACE", "./workspace"),
        port=_int("PORT", 3000, 1, 65535),
        github_token=_env("GITHUB_TOKEN"),
        vercel_token=_env("VERCEL_TOKEN"),
        flux_api_key_1=_env("FLUX_API_KEY_1"),
        flux_api_key_2=_env("FLUX_API_KEY_2"),
        flux_base_url=_env("FLUX_BASE_URL", "https://api.bfl.ai/v1"),
        flux_endpoint=_env("FLUX_ENDPOINT", "flux-2-flex"),
    )


config = load_settings()
