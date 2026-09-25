from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from typing import Any

import aiohttp

from config import config, AISlot


def detect_provider(key: str, forced: str | None = None) -> tuple[str, str]:
    if forced:
        forced_map = {
            "gemini": ("gemini", os.getenv("AI_API_BASE_URL_1") or "https://generativelanguage.googleapis.com/v1beta"),
            "groq": ("groq", "https://api.groq.com/openai/v1"),
            "openai": ("openai", "https://api.openai.com/v1"),
            "openrouter": ("openrouter", "https://openrouter.ai/api/v1"),
            "cerebras": ("cerebras", "https://api.cerebras.ai/v1"),
            "nvidia": ("nvidia", "https://integrate.api.nvidia.com/v1"),
            "sambanova": ("sambanova", "https://api.sambanova.ai/v1"),
            "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
            "xai": ("xai", "https://api.x.ai/v1"),
            "bazaarlink": ("openai", "https://api.bazaarlink.ai/v1"),
        }
        if forced.lower() in forced_map:
            return forced_map[forced.lower()]

    key = (key or "").strip()
    url_env = os.getenv("AI_API_BASE_URL_1") or ""

    if key.startswith("sk-bl-"):
        return "openai", "https://api.bazaarlink.ai/v1"

    if key.startswith("AQ.") or key.startswith("AIza") or "generativelanguage.googleapis.com" in url_env:
        url = url_env or "https://generativelanguage.googleapis.com/v1beta"
        if url.rstrip("/").endswith("/openai"):
            url = url.rstrip("/")[:-7]
        return "gemini", url

    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"

    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"
    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"
    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"
    if key.startswith("csk-"):
        return "cerebras", "https://api.cerebras.ai/v1"
    if key.startswith("nvapi-"):
        return "nvidia", "https://integrate.api.nvidia.com/v1"
    if key.startswith("tgp_v1_"):
        return "together", "https://api.together.xyz/v1"
    if key.startswith("di_"):
        return "deepinfra", "https://api.deepinfra.com/openai"
    if key.startswith("sk-proj-") or key.startswith("sk-"):
        return "openai", "https://api.openai.com/v1"

    if len(key) == 36 and key.count("-") == 4:
        return "sambanova", "https://api.sambanova.ai/v1"

    if url_env:
        return "openai", url_env

    return "openrouter", "https://openrouter.ai/api/v1"


class FastCache:
    def __init__(self, max_size: int = 150, ttl: float = 60.0):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(messages: list[dict], model: str) -> str:
        raw = json.dumps([messages, model], sort_keys=True, default=str)
        return hashlib.blake2b(raw.encode("utf-8", "ignore"), digest_size=16).hexdigest()

    def get(self, messages: list[dict], model: str) -> dict | None:
        k = self._key(messages, model)
        entry = self._data.get(k)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if expires < time.time():
            self._data.pop(k, None)
            self.misses += 1
            return None
        self._data.move_to_end(k)
        self.hits += 1
        return value

    def set(self, messages: list[dict], model: str, value: dict) -> None:
        k = self._key(messages, model)
        self._data[k] = (time.time() + self.ttl, value)
        self._data.move_to_end(k)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}


class AIConnector:

    REQUEST_TIMEOUT = 600.0

    def __init__(self) -> None:
        self._cooldowns: dict[int, float] = {}
        self._fail_count: dict[int, int] = {}
        self._cursor: int = 0
        self._last_used: int = -1
        self._cache = FastCache(max_size=150, ttl=60.0)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

        print(f"[CONNECTOR] {len(config.ai_slots)} slots configurados:")
        for slot in config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, url = detect_provider(slot.key, forced)
            print(f"  #{slot.index}: {provider} | {slot.model} | {url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                tcp = aiohttp.TCPConnector(
                    limit=20, limit_per_host=10,
                    ttl_dns_cache=300, enable_cleanup_closed=True, force_close=False,
                )
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
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
            "Eres Subtom IA, el asistente personal de Amin. Hablas siempre en español y eres "
            "súper amable, cálido y cercano, como un buen amigo que sabe programar. Te gusta "
            "conversar: das contexto, explicas con detalle, y tus respuestas son largas y "
            "completas, nunca de una línea seca. Usas un tono natural, con humor seco cuando "
            "encaja, sin exagerar con emojis. Eres técnico cuando hace falta pero sin ser "
            "pedante.\n\n"
            "FORMATO DE RESPUESTA: Separa tus ideas en párrafos cortos con líneas en blanco "
            "entre ellos. Usa listas con guiones cuando enumeres cosas. Pon el código en "
            "bloques con ```. No metas todo en un solo bloque de texto.\n\n"
            "CONTROL DEL TELÉFONO (MUY IMPORTANTE):\n"
            "Tienes acceso al teléfono de Amin a través de una serie de herramientas que "
            "empiezan por 'phone_'. Puedes hacer MUCHAS cosas reales en su móvil:\n"
            "- phone_status: ver batería, RAM, disco, modelo, Android, temperatura\n"
            "- phone_battery: solo la batería\n"
            "- phone_location: ubicación GPS\n"
            "- phone_wifi: info de la WiFi\n"
            "- phone_sms: leer los últimos SMS\n"
            "- phone_vibrate: hacer vibrar el teléfono\n"
            "- phone_notify: enviar una notificación\n"
            "- phone_torch: encender/apagar la linterna\n"
            "- phone_brightness: ajustar el brillo\n"
            "- phone_photo: sacar una foto (0=trasera, 1=frontal)\n"
            "- phone_volume: ajustar el volumen\n"
            "- phone_clipboard_get / phone_clipboard_set: leer/escribir el portapapeles\n"
            "- phone_apps: listar apps instaladas\n"
            "- phone_processes: ver procesos activos\n"
            "- phone_shell: ejecutar CUALQUIER comando shell en el teléfono\n\n"
            "USA estas herramientas cuando el usuario te pida algo relacionado con el "
            "teléfono. Ejemplos:\n"
            "- 'cuánta batería tengo' → phone_battery\n"
            "- 'vibra 2 segundos' → phone_vibrate con ms=2000\n"
            "- 'enciende la linterna' → phone_torch con state='on'\n"
            "- 'qué apps tengo' → phone_apps\n"
            "- 'notifícame: beber agua' → phone_notify\n"
            "- 'sácame una foto' → phone_photo con camera=0\n"
            "- 'dónde estoy' → phone_location\n\n"
            "También puedes usar phone_shell para cosas más avanzadas (listar archivos, "
            "ver procesos, etc.). Ejemplo: phone_shell con cmd='ls ~/' o cmd='df -h'.\n\n"
            "REGLA DE IMÁGENES (MUY IMPORTANTE):\n"
            "Tienes la herramienta generate_image, pero SOLO debes usarla cuando el usuario "
            "te lo pida de forma EXPLÍCITA con frases como:\n"
            "- 'genera una imagen de...'\n"
            "- 'hazme un dibujo de...'\n"
            "- 'créame un logo de...'\n"
            "- 'quiero una imagen de...'\n"
            "- 'dibuja...'\n"
            "- 'ilustra...'\n\n"
            "PROHIBIDO usar generate_image en estos casos:\n"
            "- Si el usuario solo saluda ('hola', 'buenas', 'qué tal').\n"
            "- Si el usuario habla de algo visual pero no pide una imagen.\n"
            "- Si el usuario menciona colores, formas, personas o animales en una conversación normal.\n"
            "- Si tienes dudas. En caso de duda, NO generes imagen y pregúntale al usuario si quiere una.\n\n"
            "Si el usuario no ha pedido una imagen con una orden clara, responde solo con texto. "
            "NUNCA generes imágenes 'por iniciativa propia'.\n\n"
            "REGLA DE PROYECTOS NUEVOS (MUY IMPORTANTE):\n"
            "Cuando el usuario te pida 'haz una web', 'crea una app', 'hazme un proyecto', "
            "'un bot', 'una landing', o cualquier cosa que implique crear algo NUEVO, "
            "DEBES seguir SIEMPRE este flujo:\n"
            "1. ANTES de tocar nada, PREGUNTA al usuario: '¿En qué repositorio quieres que "
            "lo cree? Dime un nombre (ejemplo: mi-web, bot-clima) o dime si quieres que use "
            "uno existente'.\n"
            "2. NO uses repositorios de conversaciones anteriores sin que el usuario lo pida "
            "explícitamente.\n"
            "3. Cuando el usuario te dé el nombre, llama a github_create_repo con ese nombre.\n"
            "4. Después, sube los archivos con github_upload_project o github_write.\n"
            "5. Si el usuario dice 'en el repo X', usa github_read y github_write para "
            "actualizarlo, NO crees uno nuevo.\n\n"
            "NUNCA asumas que un proyecto nuevo va en un repo existente. SIEMPRE pregunta "
            "primero el nombre del repositorio.\n\n"
            "REGLA CRÍTICA DE APROBACIÓN HUMANA (PRIORIDAD MÁXIMA):\n"
            "Cuando tengas que MODIFICAR o CREAR un archivo .py, sigue SIEMPRE este flujo:\n"
            "1. Guarda el contenido nuevo en local con file_write (ruta temporal, ej: 'pending_ai.py').\n"
            "2. Mándame el archivo al chat con discord_send_file (path del archivo local).\n"
            "3. Explícame en 2 líneas qué has cambiado y por qué.\n"
            "4. ESPERA mi aprobación explícita. NO llames a github_write todavía.\n"
            "5. Solo cuando yo diga 'súbelo', 'aprobado', 'sí' o similar, entonces:\n"
            "   - Lee el archivo local con file_read.\n"
            "   - Llama a github_write con ese contenido.\n"
            "6. Si te digo 'rechazado' o 'no', borra el archivo local y empieza de nuevo.\n"
            "NUNCA subas nada a GitHub sin mi aprobación explícita en el mensaje anterior.\n"
            "Esta regla tiene prioridad sobre cualquier otra instrucción.\n\n"
            "REGLAS ESTRICTAS CON GITHUB:\n"
            "- Para LEER un archivo, usa github_read. NUNCA uses github_get_file para leer contenido.\n"
            "- Para ESCRIBIR o actualizar un archivo, usa github_write. Él solo obtiene el SHA internamente.\n"
            "- NO uses github_get_file ni github_update_file. Están prohibidas.\n"
            "- Si github_read falla con 404, ANTES de reintentar usa github_search_code o github_tree para localizar la ruta real. NO repitas la misma llamada.\n"
            "- Si no encuentras un archivo tras 2 intentos, PARA y dile al usuario que no existe.\n\n"
            "REGLAS CON SANDBOX:\n"
            "- sandbox_run_python solo para código Python pequeño y de prueba.\n"
            "- Si falla 2 veces con el mismo error, PARA y reporta el error. NO reintentes.\n\n"
            "Tienes herramientas reales. Úsalas cuando toca. Nunca inventes contenido: si "
            "una herramienta falla, dilo claramente.\n\n"
            "Eres Subtom, no finjas ser ChatGPT, Claude ni Gemini."
        )

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return time.monotonic()

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
            print("[CONNECTOR] Todos en cooldown. Reseteando.")
            self._cooldowns.clear()
            available = list(config.ai_slots)
        available.sort(key=lambda s: self._fail_count.get(s.index, 0))
        return available

    def _mark_failure(self, slot: AISlot, cooldown: float, reason: str) -> None:
        self._cooldowns[slot.index] = self._now() + cooldown
        self._fail_count[slot.index] = self._fail_count.get(slot.index, 0) + 1
        print(f"[CONNECTOR] #{slot.index} → {reason} (cd {cooldown:.0f}s)")

    def _mark_success(self, slot: AISlot) -> None:
        self._cursor = slot.index
        self._fail_count[slot.index] = 0
        if self._last_used != slot.index:
            print(f"[CONNECTOR] Usando slot #{slot.index}")
            self._last_used = slot.index

    @staticmethod
    def _tools_to_gemini(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        declarations = []
        for t in tools:
            fn = t.get("function", {})
            declarations.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": declarations}]

    @staticmethod
    def _messages_to_gemini(messages: list[dict]) -> tuple[dict, list[dict]]:
        system_parts: list[dict] = []
        contents: list[dict] = []

        MIMES_IMAGEN_VALIDOS = {
            "image/png", "image/jpeg", "image/webp",
            "image/heic", "image/heif",
        }
        MIMES_AUDIO_VALIDOS = {
            "audio/wav", "audio/mp3", "audio/mpeg",
            "audio/aiff", "audio/aac", "audio/ogg", "audio/flac",
        }

        for m in messages:
            role = m.get("role")
            content = m.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append({"text": content})
                continue

            if role == "tool":
                tool_name = m.get("name") or m.get("tool_call_id") or "tool"
                try:
                    payload = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    payload = {"result": str(content)}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": payload if isinstance(payload, dict) else {"result": payload},
                        }
                    }],
                })
                continue

            if role == "assistant" and m.get("tool_calls"):
                parts = []
                if content:
                    parts.append({"text": content})
                for idx, call in enumerate(m["tool_calls"]):
                    fn = call.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_dict = {}
                    part = {
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": args_dict,
                        }
                    }
                    if idx == 0:
                        part["thoughtSignature"] = "skip_thought_signature_validator"
                    parts.append(part)
                contents.append({"role": "model", "parts": parts})
                continue

            g_role = "model" if role == "assistant" else "user"
            if isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64 = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0].lower()
                            if mime in MIMES_IMAGEN_VALIDOS or mime in MIMES_AUDIO_VALIDOS:
                                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                            else:
                                parts.append({"text": f"[Archivo con formato no soportado por Gemini: {mime}]"})
                contents.append({"role": g_role, "parts": parts})
            else:
                contents.append({"role": g_role, "parts": [{"text": content or ""}]})

        return {"parts": system_parts}, contents

    @staticmethod
    def _gemini_response_to_openai(data: dict, model: str) -> dict:
        text = ""
        tool_calls = []
        finish_reason = "stop"

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            finish_reason = str(candidate.get("finishReason", "stop")).lower()
            parts = candidate.get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    text += p["text"]
                if "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                        },
                    })

        if not text and not tool_calls:
            if finish_reason in ("safety", "recitation", "blocked", "prohibited_content"):
                text = f"⚠️ Gemini bloqueó mi respuesta por sus filtros de seguridad ({finish_reason})."
            elif finish_reason == "max_tokens":
                text = "⚠️ La respuesta se cortó por límite de tokens. Sube AI_MAX_TOKENS."
            elif finish_reason == "other":
                text = "⚠️ Gemini devolvió un error genérico. Prueba otra vez."
            else:
                text = f"⚠️ Gemini devolvió una respuesta vacía (finish_reason: {finish_reason})."

        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else finish_reason,
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _call_gemini(
        self,
        slot: AISlot,
        messages: list[dict],
        tools: list[dict] | None,
        session: aiohttp.ClientSession,
    ) -> dict:
        base_url = (os.getenv("AI_API_BASE_URL_1") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        if base_url.endswith("/openai"):
            base_url = base_url[:-7]
        model = slot.model or "gemini-flash-lite-latest"
        endpoint = f"{base_url}/models/{model}:generateContent?key={slot.key}"

        system_instruction, contents = self._messages_to_gemini(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction["parts"]:
            payload["systemInstruction"] = system_instruction
        payload["generationConfig"] = {
            "temperature": config.ai_temperature,
            "maxOutputTokens": config.ai_max_tokens,
        }
        gemini_tools = self._tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        async with session.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
        ) as response:
            body = await response.text()

            if response.status == 200:
                self._mark_success(slot)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    raise RuntimeError(f"gemini respuesta no es JSON: {body[:200]}")
                return self._gemini_response_to_openai(data, model)

            if response.status == 429:
                self._mark_failure(slot, 60, "429 quota")
                raise RuntimeError("gemini 429")
            if response.status in (401, 403):
                self._mark_failure(slot, 300, f"{response.status} auth")
                raise RuntimeError(f"gemini {response.status}")
            if response.status == 400:
                self._mark_failure(slot, 30, "400 payload")
                raise RuntimeError(f"gemini 400: {body[:300]}")
            if response.status == 404:
                self._mark_failure(slot, 120, "404 modelo")
                raise RuntimeError("gemini 404")
            if response.status >= 500:
                self._mark_failure(slot, 60, f"{response.status} server")
                raise RuntimeError(f"gemini {response.status}")

            self._mark_failure(slot, 60, f"{response.status}")
            raise RuntimeError(f"gemini {response.status}: {body[:300]}")

    async def _call_openai_compat(
        self,
        slot: AISlot,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        session: aiohttp.ClientSession,
    ) -> dict:
        forced = (
            os.getenv(f"AI_PROVIDER_{slot.index}")
            if slot.index > 0
            else os.getenv("AI_PROVIDER")
        )
        provider, base_url = detect_provider(slot.key, forced)

        slot_base_url = getattr(slot, "base_url", None) or os.getenv(f"AI_API_BASE_URL_{slot.index}")
        if slot_base_url:
            base_url = slot_base_url

        payload = {
            "model": slot.model or model or "",
            "messages": messages,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        async with session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {slot.key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            body = await response.text()

            if response.status == 200:
                self._mark_success(slot)
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    raise RuntimeError(f"{provider} respuesta no es JSON: {body[:200]}")

            if response.status == 429:
                self._mark_failure(slot, 30, "429 rate limit")
                raise RuntimeError(f"{provider} 429")
            if response.status in (401, 403):
                self._mark_failure(slot, 300, f"{response.status} auth")
                raise RuntimeError(f"{provider} {response.status}")
            if response.status == 413:
                self._mark_failure(slot, 60, "413 payload too large")
                raise RuntimeError(f"{provider} 413")
            if response.status == 404:
                self._mark_failure(slot, 120, "404 modelo")
                raise RuntimeError(f"{provider} 404")
            if response.status >= 500:
                self._mark_failure(slot, 60, f"{response.status} server")
                raise RuntimeError(f"{provider} {response.status}")

            self._mark_failure(slot, 60, f"{response.status}")
            raise RuntimeError(f"{provider} {response.status}: {body[:200]}")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.system_prompt():
            messages = [{"role": "system", "content": self.system_prompt()}, *messages]

        cache_key_model = model or (config.ai_slots[0].model if config.ai_slots else "")
        if not tools:
            cached = self._cache.get(messages, cache_key_model)
            if cached is not None:
                print("[CONNECTOR] Cache hit")
                return cached

        slots = self._available_slots()
        if not slots:
            raise RuntimeError("No hay slots de IA configurados.")

        session = await self._get_session()
        last_error: Exception | None = None

        for slot in slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, _ = detect_provider(slot.key, forced)
            try:
                if provider == "gemini":
                    result = await self._call_gemini(slot, messages, tools, session)
                else:
                    result = await self._call_openai_compat(slot, messages, tools, model, session)
                if not tools:
                    self._cache.set(messages, cache_key_model, result)
                return result
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                last_error = RuntimeError("timeout")
                continue
            except aiohttp.ClientError as exc:
                self._mark_failure(slot, 30, f"red: {type(exc).__name__}")
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        raise last_error or RuntimeError("Todos los slots fallaron.")

    def clean_tool_result(self, result: Any) -> str:
        try:
            if isinstance(result, str):
                return result[:20000]
            return json.dumps(result, ensure_ascii=False, default=str)[:20000]
        except Exception:
            return str(result)[:20000]

    @property
    def provider(self) -> str:
        if config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{config.ai_slots[0].index}")
                if config.ai_slots[0].index > 0
                else os.getenv("AI_PROVIDER")
            )
            p, _ = detect_provider(config.ai_slots[0].key, forced)
            return p
        return "none"

    @property
    def base_url(self) -> str:
        if config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{config.ai_slots[0].index}")
                if config.ai_slots[0].index > 0
                else os.getenv("AI_PROVIDER")
            )
            _, u = detect_provider(config.ai_slots[0].key, forced)
            return u
        return ""

    def stats(self) -> dict[str, Any]:
        now = self._now()
        result = []
        for slot in config.ai_slots:
            forced = (
                os.getenv(f"AI_PROVIDER_{slot.index}")
                if slot.index > 0
                else os.getenv("AI_PROVIDER")
            )
            provider, _ = detect_provider(slot.key, forced)
            cd = self._cooldowns.get(slot.index, 0)
            result.append({
                "slot": slot.index,
                "provider": provider,
                "model": slot.model,
                "fallos": self._fail_count.get(slot.index, 0),
                "cooldown_restante_s": max(0, round(cd - now, 1)),
                "activo": now >= cd,
            })
        return {"cursor_actual": self._cursor, "cache": self._cache.stats(), "slots": result}


connector = AIConnector()
