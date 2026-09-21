from __future__ import annotations

import logging
import time
from pathlib import Path

from aiohttp import web
from aiohttp_jinja2 import template, setup

import jinja2


_START_TIME = time.monotonic()
LOG_FILE = Path("bot.log")


# ============================================================
# LOGS: guardar en archivo para leerlos desde el panel
# ============================================================

def setup_logging() -> None:
    """Configura logging para que TODO se guarde en bot.log."""
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Redirige también print() al log
    import sys

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout = Tee(sys.stdout, open(LOG_FILE, "a", encoding="utf-8", buffering=1))
    sys.stderr = sys.stdout


def _read_logs(max_lines: int = 200) -> str:
    if not LOG_FILE.exists():
        return "(No hay logs todavía. Espera a que el bot haga algo.)"
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as exc:
        return f"(Error leyendo logs: {exc})"


def _clear_logs() -> None:
    try:
        LOG_FILE.write_text("", encoding="utf-8")
    except Exception:
        pass


def _uptime_human() -> str:
    secs = int(time.monotonic() - _START_TIME)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if mins or hours or days:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


# ============================================================
# RUTAS
# ============================================================

@template("panel.html")
async def panel(request: web.Request):
    from connector import connector
    from config import config

    try:
        from motor import motor
        motor_data = motor.snapshot()
    except Exception:
        motor_data = {}

    try:
        stats = connector.stats()
    except Exception as exc:
        stats = {"error": str(exc)}
    slots = stats.get("slots", []) if isinstance(stats, dict) else []

    return {
        "bot_name": getattr(config, "bot_name", "Subtom IA"),
        "provider": connector.provider,
        "model": getattr(config, "ai_model", "unknown"),
        "uptime": _uptime_human(),
        "total_slots": len(slots),
        "active_slots": sum(1 for s in slots if s.get("activo")),
        "slots": slots,
        "motor": motor_data,
        "logs": _read_logs(200),
    }


async def api_status(request: web.Request) -> web.Response:
    from connector import connector
    from config import config
    return web.json_response({
        "bot_name": getattr(config, "bot_name", "Subtom IA"),
        "provider": connector.provider,
        "model": getattr(config, "ai_model", "unknown"),
        "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
        "uptime_human": _uptime_human(),
        "slots": connector.stats().get("slots", []),
    })


async def api_logs(request: web.Request) -> web.Response:
    return web.json_response({"logs": _read_logs(500)})


async def clear_logs(request: web.Request) -> web.Response:
    _clear_logs()
    return web.json_response({"ok": True})


# ============================================================
# SETUP
# ============================================================

def _ensure_templates() -> None:
    """Crea la carpeta templates/ y el panel.html si no existen."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    panel_html = templates_dir / "panel.html"
    if not panel_html.exists():
        panel_html.write_text(_PANEL_HTML, encoding="utf-8")


def setup_panel(app: web.Application) -> None:
    setup_logging()
    _ensure_templates()

    templates_dir = Path(__file__).parent / "templates"
    setup(app, loader=jinja2.FileSystemLoader(str(templates_dir)))

    app.router.add_get("/panel", panel)
    app.router.add_get("/api/panel/status", api_status)
    app.router.add_get("/api/panel/logs", api_logs)
    app.router.add_post("/api/panel/logs/clear", clear_logs)


# ============================================================
# HTML EMBEBIDO
# ============================================================

_PANEL_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ bot_name }} - Panel</title>
<meta http-equiv="refresh" content="30">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
    min-height: 100vh;
  }
  h1 { color: #58a6ff; margin-bottom: 20px; font-size: 26px; }
  h2 { color: #f0883e; margin: 25px 0 12px; font-size: 18px;
       border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px; margin-bottom: 10px;
  }
  .card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 15px;
  }
  .card .label {
    font-size: 11px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .card .value {
    font-size: 22px; font-weight: bold; color: #58a6ff; margin-top: 5px;
    word-break: break-word;
  }
  .card .value.small { font-size: 14px; }
  table {
    width: 100%; border-collapse: collapse;
    background: #161b22; border-radius: 8px; overflow: hidden;
    font-size: 14px;
  }
  th, td { padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }
  th { background: #21262d; color: #8b949e; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.5px; }
  tr:last-child td { border-bottom: none; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: bold;
  }
  .badge-ok { background: #1f6feb33; color: #58a6ff; }
  .badge-bad { background: #f8514933; color: #f85149; }
  .badge-warn { background: #d2992233; color: #d29922; }
  pre {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 15px; overflow-x: auto; font-size: 12px; color: #7ee787;
    max-height: 500px; overflow-y: auto; white-space: pre-wrap;
    word-break: break-word; font-family: "Consolas", "Monaco", monospace;
  }
  .toolbar {
    display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap;
  }
  button {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    padding: 8px 16px; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 600;
  }
  button:hover { background: #30363d; border-color: #8b949e; }
  button.danger { color: #f85149; }
  button.danger:hover { background: #f8514922; }
  .footer {
    margin-top: 30px; color: #8b949e; font-size: 12px;
    text-align: center;
  }
</style>
</head>
<body>

<h1>🤖 {{ bot_name }} — Panel de control</h1>

<div class="grid">
  <div class="card">
    <div class="label">Uptime</div>
    <div class="value">{{ uptime }}</div>
  </div>
  <div class="card">
    <div class="label">Proveedor</div>
    <div class="value">{{ provider }}</div>
  </div>
  <div class="card">
    <div class="label">Modelo</div>
    <div class="value small">{{ model }}</div>
  </div>
  <div class="card">
    <div class="label">Slots activos</div>
    <div class="value">{{ active_slots }}/{{ total_slots }}</div>
  </div>
</div>

<h2>📊 Estado de los slots</h2>
<table>
  <thead>
    <tr>
      <th>Slot</th><th>Proveedor</th><th>Modelo</th>
      <th>Fallos</th><th>Cooldown</th><th>Estado</th>
    </tr>
  </thead>
  <tbody>
    {% for slot in slots %}
    <tr>
      <td>#{{ slot.slot }}</td>
      <td>{{ slot.provider }}</td>
      <td>{{ slot.model }}</td>
      <td>{{ slot.fallos }}</td>
      <td>{{ slot.cooldown_restante_s }}s</td>
      <td>
        {% if slot.activo %}
          <span class="badge badge-ok">ACTIVO</span>
        {% else %}
          <span class="badge badge-bad">COOLDOWN</span>
        {% endif %}
      </td>
    </tr>
    {% else %}
    <tr><td colspan="6" style="text-align:center;color:#8b949e">
      No hay slots configurados.
    </td></tr>
    {% endfor %}
  </tbody>
</table>

<h2>📜 Logs (últimas 200 líneas)</h2>
<div class="toolbar">
  <button onclick="location.reload()">🔄 Recargar</button>
  <button onclick="copyLogs()">📋 Copiar logs</button>
  <button class="danger" onclick="clearLogs()">🗑️ Borrar logs</button>
</div>
<pre id="logsBox">{{ logs }}</pre>

<div class="footer">
  Actualiza cada 30 segundos. Generado por {{ bot_name }}.
</div>

<script>
function copyLogs() {
  const text = document.getElementById("logsBox").innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("✅ Logs copiados al portapapeles (" + text.length + " caracteres)");
  }).catch(err => {
    alert("❌ Error copiando: " + err);
  });
}
function clearLogs() {
  if (!confirm("¿Seguro que quieres borrar los logs?")) return;
  fetch("/api/panel/logs/clear", { method: "POST" })
    .then(r => r.json())
    .then(() => location.reload());
}
</script>

</body>
</html>
"""
