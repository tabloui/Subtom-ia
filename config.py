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
        raise RuntimeError(
            f"Falta la variable {name}"
        )

    return value


def _int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:

    raw = _env(name)

    if raw is None:
        return default

    try:
        value = int(raw)

    except ValueError as exc:
        raise RuntimeError(
            f"{name} debe ser un entero"
        ) from exc

    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} debe estar entre "
            f"{minimum} y {maximum}"
        )

    return value


@dataclass(frozen=True)
class Settings:

    # Discord
    discord_token: str
    allowed_user_id: int

    # Database
    database_url: str

    # OpenRouter
    openrouter_api_keys: list[str]
    ai_model: str
    ai_temperature: float
    ai_max_tokens: int
    ai_max_tool_rounds: int
    ai_timeouts_seconds: int

    # Bot
    bot_name: str
    bot_language: str
    bot_personality: str
    bot_style: str
    bot_system_text: str

    # Memory
    memory_limit: int
    memory_messages: int

    # Server
    workspace: str
    port: int

    # Optional APIs
    github_token: str | None
    vercel_token: str | None
    flux_api_key_1: str | None
    flux_api_key_2: str | None
    flux_base_url: str
    flux_endpoint: str


def load_settings() -> Settings:

    # ========================================================
    # DISCORD
    # ========================================================

    discord_token = _required(
        "DISCORD_BOT_TOKEN"
    )

    try:

        allowed_user_id = int(
            _required(
                "DISCORD_ALLOWED_USER_ID"
            )
        )

    except ValueError as exc:

        raise RuntimeError(
            "DISCORD_ALLOWED_USER_ID "
            "debe ser un ID numérico"
        ) from exc

    # ========================================================
    # DATABASE
    # ========================================================

    database_url = _env(
        "DATABASE_URL"
    )

    if not database_url:

        raise RuntimeError(
            "Falta la variable DATABASE_URL"
        )

    # ========================================================
    # OPENROUTER
    #
    # Ahora admite:
    #
    # OPENROUTER_API_KEY
    # OPENROUTER_API_KEY_2
    # OPENROUTER_API_KEY_3
    # OPENROUTER_API_KEY_4
    #
    # ========================================================

    openrouter_api_keys: list[str] = []

    for variable in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY_2",
        "OPENROUTER_API_KEY_3",
        "OPENROUTER_API_KEY_4",
    ):

        value = _env(variable)

        if value:
            openrouter_api_keys.append(
                value
            )

    if not openrouter_api_keys:

        raise RuntimeError(
            "Falta OPENROUTER_API_KEY "
            "(o _2, _3, _4)"
        )

    # ========================================================
    # AI TEMPERATURE
    # ========================================================

    temperature_raw = _env(
        "AI_TEMPERATURE",
        "0.35",
    )

    try:

        ai_temperature = float(
            temperature_raw
        )

    except ValueError as exc:

        raise RuntimeError(
            "AI_TEMPERATURE debe ser un número"
        ) from exc

    if not 0 <= ai_temperature <= 2:

        raise RuntimeError(
            "AI_TEMPERATURE debe estar "
            "entre 0 y 2"
        )

    # ========================================================
    # AI MODEL
    # ========================================================

    ai_model = _env(
        "AI_MODEL",
        "openrouter/free",
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    return Settings(

        # ----------------------------------------------------
        # Discord
        # ----------------------------------------------------

        discord_token=discord_token,

        allowed_user_id=allowed_user_id,

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        database_url=database_url,

        # ----------------------------------------------------
        # OpenRouter
        # ----------------------------------------------------

        openrouter_api_keys=openrouter_api_keys,

        ai_model=ai_model or "openrouter/free",

        ai_temperature=ai_temperature,

        ai_max_tokens=_int(
            "AI_MAX_TOKENS",
            5000,
            256,
            16000,
        ),

        ai_max_tool_rounds=_int(
            "MAX_TOOL_ROUNDS",
            10,
            1,
            30,
        ),

        ai_timeouts_seconds=_int(
            "AI_TIMEOUT_SECONDS",
            90,
            10,
            300,
        ),

        # ----------------------------------------------------
        # Bot
        # ----------------------------------------------------

        bot_name=_env(
            "BOT_NAME",
            "Subtom",
        ) or "Subtom",

        bot_language=_env(
            "BOT_LANGUAGE",
            "español",
        ) or "español",

        bot_personality=_env(
            "BOT_PERSONALITY",
            "inteligente, útil, directo, "
            "natural y competente",
        ) or (
            "inteligente, útil, directo, "
            "natural y competente"
        ),

        bot_style=_env(
            "BOT_STYLE",
            "responde de forma clara y práctica",
        ) or (
            "responde de forma clara y práctica"
        ),

        bot_system_text=_env(
            "BOT_SYSTEM_TEXT",
            "",
        ) or "",

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        memory_limit=_int(
            "MEMORY_LIMIT",
            30,
            0,
            10080,
        ),

        memory_messages=_int(
            "MEMORY_MESSAGES",
            60,
            4,
            200,
        ),

        # ----------------------------------------------------
        # Server
        # ----------------------------------------------------

        workspace=_env(
            "SUBTOM_WORKSPACE",
            "./workspace",
        ) or "./workspace",

        port=_int(
            "PORT",
            3000,
            1,
            65535,
        ),

        # ----------------------------------------------------
        # Optional APIs
        # ----------------------------------------------------

        github_token=_env(
            "GITHUB_TOKEN"
        ),

        vercel_token=_env(
            "VERCEL_TOKEN"
        ),

        flux_api_key_1=_env(
            "FLUX_API_KEY_1"
        ),

        flux_api_key_2=_env(
            "FLUX_API_KEY_2"
        ),

        flux_base_url=_env(
            "FLUX_BASE_URL",
            "https://api.bfl.ai/v1",
        ) or "https://api.bfl.ai/v1",

        flux_endpoint=_env(
            "FLUX_ENDPOINT",
            "flux-schnell",
        ) or "flux-schnell",
    )


# ============================================================
# CONFIG GLOBAL
# ============================================================

config = load_settings()
