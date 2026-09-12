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
        raise RuntimeError(f"Falta la variable obligatoria: {name}")
    return value


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un número entero") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    discord_token: str
    allowed_user_id: int
    database_url: str

    openrouter_api_keys: list[str]
    ai_model: str
    ai_temperature: float
    ai_max_tokens: int
    ai_max_tool_rounds: int
    ai_timeout_seconds: int

    bot_name: str
    bot_language: str
    bot_personality: str
    bot_style: str
    bot_system_text: str

    memory_minutes: int
    memory_messages: int

    workspace: str
    port: int

    github_token: str | None
    vercel_token: str | None

    flux_api_key_1: str | None
    flux_api_key_2: str | None
    flux_base_url: str
    flux_endpoint: str


def load_settings() -> Settings:
    try:
        allowed_user_id = int(_required("DISCORD_ALLOWED_USER_ID"))
    except ValueError as exc:
        raise RuntimeError("DISCORD_ALLOWED_USER_ID debe ser un ID numérico de Discord") from exc

    database_url = _env("NEON_DATABASE_URL") or _env("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Falta NEON_DATABASE_URL o DATABASE_URL")

    # ---- Rotación de claves OpenRouter ----
    keys: list[str] = []
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"):
        value = _env(var)
        if value:
            keys.append(value)
    if not keys:
        raise RuntimeError("Falta OPENROUTER_API_KEY (y opcionalmente _2 y _3)")

    temperature_raw = _env("AI_TEMPERATURE", "0.35")
    try:
        temperature = float(temperature_raw)
    except ValueError as exc:
        raise RuntimeError("AI_TEMPERATURE debe ser un número") from exc
    if not 0 <= temperature <= 2:
        raise RuntimeError("AI_TEMPERATURE debe estar entre 0 y 2")

    model = _env("AI_MODEL", "openrouter/free")
    if not model:
        model = "openrouter/free"

    return Settings(
        discord_token=_required("DISCORD_BOT_TOKEN"),
        allowed_user_id=allowed_user_id,
        database_url=database_url,
        openrouter_api_keys=keys,
        ai_model=model,
        ai_temperature=temperature,
        ai_max_tokens=_int("AI_MAX_TOKENS", 5000, 256, 16000),
        ai_max_tool_rounds=_int("MAX_TOOL_ROUNDS", 10, 1, 30),
        ai_timeout_seconds=_int("AI_TIMEOUT_SECONDS", 90, 10, 300),
        bot_name=_env("BOT_NAME", "Subtom") or "Subtom",
        bot_language=_env("BOT_LANGUAGE", "español") or "español",
        bot_personality=_env(
            "BOT_PERSONALITY",
            "inteligente, útil, directo, natural y competente",
        ) or "inteligente, útil, directo, natural y competente",
        bot_style=_env(
            "BOT_STYLE",
            "responde de forma clara y práctica, sin tutoriales innecesarios",
        ) or "responde de forma clara y práctica, sin tutoriales innecesarios",
        bot_system_text=_env("BOT_SYSTEM_TEXT", "") or "",
        memory_minutes=_int("MEMORY_MINUTES", 30, 0, 10080),
        memory_messages=_int("MEMORY_MESSAGES", 60, 4, 200),
        workspace=_env("SUBTOM_WORKSPACE", "./workspace") or "./workspace",
        port=_int("PORT", 3000, 1, 65535),
        github_token=_env("GITHUB_TOKEN"),
        vercel_token=_env("VERCEL_TOKEN"),
        flux_api_key_1=_env("FLUX_API_KEY_1"),
        flux_api_key_2=_env("FLUX_API_KEY_2"),
        flux_base_url=_env("FLUX_BASE_URL", "https://api.bfl.ai/v1") or "https://api.bfl.ai/v1",
        flux_endpoint=_env("FLUX_ENDPOINT", "flux-schnell") or "flux-schnell",
    )


config = load_settings()
