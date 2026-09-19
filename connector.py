from __future__ import annotations

import os
from dataclasses import dataclass, field


# ============================================================
# HELPERS
# ============================================================

def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
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
        raise RuntimeError(f"{name} debe ser un entero (recibido: {raw!r})") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} debe estar entre {minimum} y {maximum} (recibido: {value})"
        )
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un número (recibido: {raw!r})") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} debe estar entre {minimum} y {maximum} (recibido: {value})"
        )
    return value


# ============================================================
# MODELOS
# ============================================================

@dataclass(frozen=True)
class AISlot:
    index: int
    key: str
    model: str
    base_url: str = ""


@dataclass(frozen=True)
class Settings:
    # Discord
    discord_token: str
    allowed_user_id: int

    # Database
    database_url: str

    # IA
    ai_slots: list[AISlot]
    ai_temperature: float
    ai_max_tokens: int
    ai_max_tool_rounds: int
    ai_timeouts_seconds: int

    # Bot
    bot_name: str

    # Memory
    memory_limit: int
    memory_messages: int

    # Server
    workspace: str
    port: int

    # Optional APIs
    github_token: str | None
    vercel_token: str | None

    # FLUX
    flux_api_key_1: str | None
    flux_api_key_2: str | None
    flux_base_url: str
    flux_endpoint: str

    @property
    def ai_model(self) -> str:
        if self.ai_slots:
            return self.ai_slots[0].model
        return "unknown"


# ============================================================
# CARGA DE SLOTS (ILIMITADOS)
# ============================================================

def _load_slots(default_model: str) -> list[AISlot]:
    """
    Carga todos los slots posibles:
    - AI_API_KEY (sin número) como slot 0 (compatibilidad)
    - AI_API_KEY_1, AI_API_KEY_2, ... hasta que no haya más

    Para cada slot N:
    - AI_API_KEY_N     (obligatorio para que el slot exista)
    - AI_MODEL_N       (opcional, usa default_model)
    - AI_API_BASE_URL_N (opcional, para forzar proveedor)
    """
    slots: list[AISlot] = []

    # Slot 0: compatibilidad con AI_API_KEY sin número
    main_key = _env("AI_API_KEY")
    if main_key:
        main_model = _env("AI_MODEL", default_model) or default_model
        main_url = _env("AI_API_BASE_URL", "") or ""
        slots.append(AISlot(index=0, key=main_key, model=main_model, base_url=main_url))

    # Slots 1, 2, 3... sin límite, paramos cuando no haya key
    n = 1
    empty_streak = 0
    while True:
        key = _env(f"AI_API_KEY_{n}")
        if not key:
            empty_streak += 1
            # Tolerar hasta 5 huecos consecutivos antes de parar
            if empty_streak >= 5:
                break
            n += 1
            continue
        empty_streak = 0

        model = _env(f"AI_MODEL_{n}") or default_model
        base_url = _env(f"AI_API_BASE_URL_{n}") or ""
        slots.append(AISlot(index=n, key=key, model=model, base_url=base_url))
        n += 1

        # Límite de seguridad para no entrar en bucle infinito si hay muchas vars
        if n > 100:
            print("[CONFIG] Aviso: más de 100 slots detectados, parando.")
            break

    return slots


# ============================================================
# CARGA DE SETTINGS
# ============================================================

def load_settings() -> Settings:

    # --- DISCORD ---
    discord_token = _required("DISCORD_BOT_TOKEN")

    try:
        allowed_user_id = int(_required("DISCORD_ALLOWED_USER_ID"))
    except ValueError as exc:
        raise RuntimeError("DISCORD_ALLOWED_USER_ID debe ser numérico") from exc

    # --- DATABASE ---
    database_url = _required("DATABASE_URL")

    # --- IA SLOTS ---
    default_model = _env("AI_MODEL", "gemini-flash-lite-latest") or "gemini-flash-lite-latest"
    slots = _load_slots(default_model)

    if not slots:
        raise RuntimeError("Configura al menos AI_API_KEY_1")

    # Ordenar por índice
    slots.sort(key=lambda s: s.index)

    # --- TEMPERATURA ---
    ai_temperature = _float("AI_TEMPERATURE", 0.7, 0.0, 2.0)

    # --- SETTINGS ---
    return Settings(
        # Discord
        discord_token=discord_token,
        allowed_user_id=allowed_user_id,

        # Database
        database_url=database_url,

        # IA
        ai_slots=slots,
        ai_temperature=ai_temperature,
        ai_max_tokens=_int("AI_MAX_TOKENS", 4096, 16, 32768),
        ai_max_tool_rounds=_int("MAX_SAFETY_ROUNDS", 100, 1, 500),
        ai_timeouts_seconds=_int("AI_TIMEOUT_SECONDS", 600, 10, 3600),

        # Bot
        bot_name=_env("BOT_NAME", "Subtom IA") or "Subtom IA",

        # Memory
        memory_limit=_int("MEMORY_LIMIT", 120, 0, 10080),
        memory_messages=_int("MEMORY_MESSAGES", 12, 2, 200),

        # Server
        workspace=_env("SUBTOM_WORKSPACE", "./workspace") or "./workspace",
        port=_int("PORT", 8080, 1, 65535),

        # Optional APIs
        github_token=_env("GITHUB_TOKEN"),
        vercel_token=_env("VERCEL_TOKEN"),

        # FLUX
        flux_api_key_1=_env("FLUX_API_KEY_1"),
        flux_api_key_2=_env("FLUX_API_KEY_2"),
        flux_base_url=_env("FLUX_BASE_URL", "https://api.bfl.ai/v1") or "https://api.bfl.ai/v1",
        flux_endpoint=_env("FLUX_ENDPOINT", "flux-2-flex") or "flux-2-flex",
    )


# ============================================================
# INSTANCIA GLOBAL + RESUMEN EN LOG
# ============================================================

config = load_settings()

print("=" * 60)
print(f"[CONFIG] Bot: {config.bot_name}")
print(f"[CONFIG] Slots de IA: {len(config.ai_slots)}")
for slot in config.ai_slots:
    key_preview = (slot.key[:12] + "...") if len(slot.key) > 12 else slot.key
    url_preview = slot.base_url or "(auto)"
    print(f"  #{slot.index}: {key_preview} | modelo={slot.model} | url={url_preview}")
print(f"[CONFIG] Temperatura: {config.ai_temperature}")
print(f"[CONFIG] Max tokens: {config.ai_max_tokens}")
print(f"[CONFIG] Timeout: {config.ai_timeouts_seconds}s")
print(f"[CONFIG] Memoria: {config.memory_limit} min, {config.memory_messages} mensajes")
print(f"[CONFIG] Workspace: {config.workspace}")
print(f"[CONFIG] Puerto HTTP: {config.port}")
print("=" * 60)
