# api_motor.py — Motor de API de Subtom IA
# Usa el mismo agente de ai.py, no duplica herramientas.

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from config import config
from connector import connector
from ai import agent


# ============================================================
# DETECCIÓN DE PLATAFORMA / UBICACIÓN
# ============================================================

PLATFORM_HINTS = {
    "discord":   ["discord", "discordapp"],
    "whatsapp":  ["whatsapp", "wa.me", "whatsapp-web"],
    "telegram":  ["telegram", "t.me", "tg://"],
    "railway":   ["railway.app", "railway.com"],
    "termux":    ["termux"],
    "sandbox":   ["sandbox", "playground", "codesandbox", "replit", "stackblitz"],
    "web":       ["mozilla", "chrome", "safari", "firefox", "edge/", "opera"],
    "robot":     ["robot", "iot", "esp32", "arduino", "raspberry"],
    "bluetooth": ["bluetooth", "bt/"],
    "wifi":      ["wifi", "wlan"],
    "api":       ["curl", "python-requests", "postman", "insomnia",
                  "httpie", "aiohttp", "axios", "node-fetch"],
    "mobile":    ["okhttp", "android", "iphone", "ipad"],
    "desconocido": [],
}


def detect_platform(request: web.Request) -> str:
    ua = (request.headers.get("User-Agent") or "").lower()
    referer = (request.headers.get("Referer") or "").lower()
    origin = (request.headers.get("Origin") or "").lower()
    x_platform = (request.headers.get("X-Platform") or "").lower()
    q_platform = (request.query.get("platform") or "").lower()

    texto = f"{x_platform} {q_platform} {ua} {referer} {origin}"

    for platform, hints in PLATFORM_HINTS.items():
        if not hints:
            continue
        for hint in hints:
            if hint in texto:
                return platform
    return "desconocido"


def location_info(request: web.Request) -> dict[str, Any]:
    return {
        "platform": detect_platform(request),
        "user_agent": (request.headers.get("User-Agent") or "")[:300],
        "referer": (request.headers.get("Referer") or "")[:300],
        "origin": (request.headers.get("Origin") or "")[:300],
        "host": request.headers.get("Host") or "",
        "ip": request.remote or "desconocida",
        "path": request.path,
        "method": request.method,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================
# MEMORIA GLOBAL (compartida entre plataformas)
# ============================================================

class GlobalMemory:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[MEMORY] Error guardando: {exc}")

    def add(self, user_id: str, role: str, content: str, platform: str = "") -> None:
        if user_id not in self._data:
            self._data[user_id] = []
        self._data[user_id].append({
            "role": role,
            "content": content[:30000],
            "platform": platform,
            "ts": time.time(),
        })
        if len(self._data[user_id]) > 500:
            self._data[user_id] = self._data[user_id][-300:]
        self._save()

    def history(self, user_id: str, limit: int = 40) -> list[dict]:
        items = self._data.get(user_id, [])
        return [{"role": m["role"], "content": m["content"]} for m in items[-limit:]]

    def platforms_for(self, user_id: str) -> list[str]:
        return list({m.get("platform", "") for m in self._data.get(user_id, []) if m.get("platform")})

    def clear(self, user_id: str | None = None) -> None:
        if user_id:
            self._data.pop(user_id, None)
        else:
            self._data = {}
        self._save()


_memory = GlobalMemory(Path(config.workspace) / "global_memory.json")


# ============================================================
# AGENTE (usa el mismo de ai.py)
# ============================================================

async def agent_ask(prompt: str, user_id: str, platform: str, location: dict, max_rounds: int = 8) -> dict:
    # Contexto de ubicación para que la IA sepa dónde está
    contexto_ubicacion = (
        f"[CONTEXTO DE UBICACIÓN]\n"
        f"- Plataforma: {platform}\n"
        f"- IP: {location.get('ip', 'desconocida')}\n"
        f"- User-Agent: {location.get('user_agent', '')[:120]}\n"
        f"- Host: {location.get('host', '')}\n"
        f"- Referer: {location.get('referer', '')}\n"
        f"- El usuario está hablando desde esta plataforma.\n"
        f"- Si te pregunta dónde está o desde dónde te habla, usa where_am_i.\n"
    )

    # Historial global del usuario (compartido entre plataformas)
    history = _memory.history(user_id, limit=40)

    # Construir mensajes: contexto + historial + prompt nuevo
    messages: list[dict] = [{"role": "user", "content": contexto_ubicacion}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # Guardar en memoria
    _memory.add(user_id, "user", prompt, platform)

    # Bucle del agente usando las herramientas de ai.py
    used = []
    for _ in range(max_rounds):
        response = await connector.complete(messages, agent.tool_schemas())
        choice = response["choices"][0]
        msg = choice["message"]
        tool_calls = agent._extract_tool_calls(msg)

        a_msg = {"role": "assistant", "content": msg.get("content")}
        if tool_calls:
            a_msg["tool_calls"] = tool_calls
        messages.append(a_msg)

        if not tool_calls:
            answer = msg.get("content") or ""
            _memory.add(user_id, "assistant", answer, platform)
            return {
                "response": answer,
                "platform": platform,
                "user_id": user_id,
                "tools_used": used,
                "platforms_history": _memory.platforms_for(user_id),
            }

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}

            # Ejecutar la herramienta del agente de ai.py
            result = await agent.run_tool(name, args)

            # Si es where_am_i, lo resolvemos aquí (no está en ai.py)
            if name == "where_am_i":
                result = {
                    "platform": location.get("platform", "desconocido"),
                    "ip": location.get("ip", "desconocida"),
                    "user_agent": location.get("user_agent", ""),
                    "host": location.get("host", ""),
                    "referer": location.get("referer", ""),
                }

            used.append(name)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(result, ensure_ascii=False, default=str)[:20000],
            })

    return {
        "response": "[Límite de rondas alcanzado]",
        "platform": platform,
        "user_id": user_id,
        "tools_used": used,
    }


# ============================================================
# RUTAS HTTP
# ============================================================

async def route_location_api(request: web.Request) -> web.Response:
    info = location_info(request)
    return web.json_response({"ok": True, "location": info})


async def route_api_chat(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt vacío"}, status=400)

    user_id = str(body.get("user_id") or "anonimo")
    platform = (body.get("platform") or "").strip() or detect_platform(request)
    location = location_info(request)

    try:
        result = await agent_ask(prompt, user_id, platform, location)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)


async def route_api_tool(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    tool = (body.get("tool") or "").strip()
    if not tool:
        return web.json_response({"error": "tool vacío"}, status=400)

    if tool == "where_am_i":
        return web.json_response({"tool": tool, "result": location_info(request)})

    result = await agent.run_tool(tool, body.get("args", {}))
    return web.json_response({"tool": tool, "result": result})


async def route_api_status(request: web.Request) -> web.Response:
    location = location_info(request)
    return web.json_response({
        "ok": True,
        "service": config.bot_name,
        "provider": connector.provider,
        "model": config.ai_model,
        "location": location,
        "stats": connector.stats(),
    })


async def route_api_tools(request: web.Request) -> web.Response:
    schemas = agent.tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    names.append("where_am_i")
    return web.json_response({"total": len(names), "tools": names})


async def route_memory_clear(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_id = body.get("user_id")
    _memory.clear(str(user_id) if user_id else None)
    return web.json_response({"ok": True, "cleared": user_id or "all"})


async def route_memory_history(request: web.Request) -> web.Response:
    user_id = request.query.get("user_id", "")
    limit = int(request.query.get("limit", "40"))
    if not user_id:
        return web.json_response({"error": "falta user_id"}, status=400)
    return web.json_response({
        "user_id": user_id,
        "platforms": _memory.platforms_for(user_id),
        "history": _memory.history(user_id, limit),
    })


# ============================================================
# SETUP
# ============================================================

def setup_api(app: web.Application) -> None:
    app.router.add_get("/location_api", route_location_api)
    app.router.add_post("/api/chat", route_api_chat)
    app.router.add_post("/api/tool", route_api_tool)
    app.router.add_get("/api/status", route_api_status)
    app.router.add_get("/api/tools", route_api_tools)
    app.router.add_post("/api/memory/clear", route_memory_clear)
    app.router.add_get("/api/memory/history", route_memory_history)
    print("[API_MOTOR] Rutas: /location_api, /api/chat, /api/tool, /api/status, /api/tools, /api/memory/*")
