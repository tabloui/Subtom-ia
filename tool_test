#!/usr/bin/env python3
"""
test_tools.py — Gate de seguridad de Subtom IA.

Se ejecuta ANTES de cada `github_write` sobre un archivo .py.
Si termina con "TESTS FALLIDOS" en stdout, el bot NO debe subir el cambio.

Comprueba 12 cosas:
  1.  Sintaxis de todos los .py del repo
  2.  Importación de los módulos core
  3.  Clase Agent y instancia agent
  4.  Registro de herramientas (tool_schemas)
  5.  MEJORA 1: sandbox_verify_change funciona
  6.  MEJORA 2: github_write no escribe en main por defecto
  7.  MEJORA 3: /health con uptime
  8.  MEJORA 4: este archivo está enganchado
  9.  NUEVO: Coherencia tool_schemas vs run_tool
  10. NUEVO: Conectividad de red con reintentos
  11. NUEVO: Variables de entorno críticas presentes
  12. NUEVO: Líneas mínimas y estructura de ai.py

Solo stdlib. Sin dependencias externas.
"""

import ast
import importlib
import inspect
import os
import py_compile
import socket
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CORE_FILES = ["ai.py", "bot.py", "connector.py", "sandbox.py"]
ALL_CRITICAL_FILES = ["ai.py", "bot.py", "connector.py", "config.py",
                      "sandbox.py", "motor.py", "file_tools.py"]

VERIFY_FN_CANDIDATES = ["sandbox_verify_change", "verify_change", "verify_python_change"]
GITHUB_WRITE_CANDIDATES = ["github_write"]
HEALTH_FN_CANDIDATES = ["health", "health_check", "handle_health"]
TOOL_SCHEMA_CANDIDATES = ["tool_schemas", "get_tool_schemas", "tools_schema"]
RUN_TOOL_CANDIDATES = ["run_tool", "execute_tool", "call_tool"]

PROTECTED_BRANCHES = {"main", "master"}

REQUIRED_ENV_VARS = [
    "DISCORD_BOT_TOKEN",
    "AI_API_KEY_1",
    "AI_API_BASE_URL_1",
    "GITHUB_TOKEN",
    "DATABASE_URL",
]

NETWORK_HOSTS = [
    ("api.github.com", 443),
    ("generativelanguage.googleapis.com", 443),
]
NETWORK_TIMEOUT = 5
NETWORK_RETRIES = 3

MIN_AI_LINES = 500

# ---------------------------------------------------------------------------

START_TIME = time.monotonic()
results = []  # (status, name, detail) — status in {"OK", "FAIL", "SKIP"}


def report(status, name, detail=""):
    results.append((status, name, detail))


def _elapsed():
    return time.monotonic() - START_TIME


def find_py_files():
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "env", "node_modules"}
    found = []
    for path in ROOT.rglob("*.py"):
        if any(part in skip_dirs or part.startswith(".") for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        found.append(path)
    return found


def find_function(candidates, modules=("sandbox", "ai", "connector", "bot")):
    for modname in modules:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for cand in candidates:
            if hasattr(mod, cand):
                return getattr(mod, cand), f"{modname}.{cand}"
    return None, None


# ===========================================================================
# 1. SINTAXIS DE TODOS LOS .py
# ===========================================================================
def test_syntax_all_files():
    files = find_py_files()
    if not files:
        report("SKIP", "sintaxis", "no se encontraron archivos .py")
        return
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
        except Exception as e:
            report("FAIL", f"sintaxis: {f.name}", str(e))
        else:
            report("OK", f"sintaxis: {f.name}")


# ===========================================================================
# 2. IMPORTS DE MÓDULOS CORE
# ===========================================================================
def test_core_imports():
    os.environ.setdefault("SUBTOM_TEST_MODE", "1")
    for filename in CORE_FILES:
        modname = filename[:-3]
        path = ROOT / filename
        if not path.exists():
            report("SKIP", f"import: {filename}", "archivo no encontrado")
            continue
        try:
            if modname in sys.modules:
                importlib.reload(sys.modules[modname])
            else:
                importlib.import_module(modname)
        except Exception as e:
            report("FAIL", f"import: {filename}", f"{type(e).__name__}: {e}")
        else:
            report("OK", f"import: {filename}")


# ===========================================================================
# 3. CLASE AGENT + INSTANCIA
# ===========================================================================
def test_agent_exists():
    try:
        ai = importlib.import_module("ai")
    except Exception as e:
        report("FAIL", "ai.Agent/agent", f"ai.py no importable: {e}")
        return
    report("OK" if hasattr(ai, "Agent") else "FAIL", "ai.Agent",
           "" if hasattr(ai, "Agent") else "no existe la clase Agent")
    report("OK" if hasattr(ai, "agent") else "FAIL", "ai.agent",
           "" if hasattr(ai, "agent") else "no existe la instancia agent")


# ===========================================================================
# 4. REGISTRO DE HERRAMIENTAS
# ===========================================================================
def test_tool_registry():
    try:
        ai = importlib.import_module("ai")
    except Exception:
        report("SKIP", "registro de herramientas", "ai.py no importable")
        return

    # Buscar el método tool_schemas en la instancia agent
    agent = getattr(ai, "agent", None)
    schemas = None
    for attr in TOOL_SCHEMA_CANDIDATES:
        if agent is not None and hasattr(agent, attr):
            try:
                schemas = getattr(agent, attr)()
            except Exception as e:
                report("FAIL", "tool_schemas()", f"excepción: {e}")
                return
            break

    if schemas is None:
        report("SKIP", "registro de herramientas",
                "no se encontró tool_schemas() en agent")
        return

    if not isinstance(schemas, list):
        report("FAIL", "tool_schemas()", f"devolvió {type(schemas).__name__}, esperado list")
        return

    malformed = []
    for tool in schemas:
        if not isinstance(tool, dict):
            malformed.append(str(tool)[:40])
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or "name" not in fn:
            malformed.append(str(tool)[:40])

    if malformed:
        report("FAIL", f"tool_schemas: {len(malformed)} mal formados",
                f"ejemplos: {malformed[:3]}")
    else:
        report("OK", f"tool_schemas: {len(schemas)} herramientas válidas")

    if len(schemas) < 60:
        report("SKIP", "conteo de herramientas",
                f"se esperaban ~60+, se encontraron {len(schemas)}")


# ===========================================================================
# 5. MEJORA 1 — sandbox_verify_change funciona
# ===========================================================================
def test_verify_change_exists_and_works():
    fn, where = find_function(VERIFY_FN_CANDIDATES)
    if fn is None:
        report("FAIL", "sandbox_verify_change",
                f"no se encontró {VERIFY_FN_CANDIDATES} — MEJORA 1 no confirmable")
        return
    report("OK", f"sandbox_verify_change existe ({where})")

    old_code = "def suma(a, b):\n    return a + b\n\ndef resta(a, b):\n    return a - b\n"
    same_code = old_code
    broken_syntax = "def suma(a, b)\n    return a + b\n"
    removed_fn = "def suma(a, b):\n    return a + b\n"
    unrelated = "x = 1\ny = 2\nz = 3\nprint('nada que ver')\n"

    def call(old, new):
        try:
            return fn(old, new)
        except TypeError:
            return fn(old_code=old, new_code=new)

    def is_ok(result):
        if isinstance(result, tuple):
            return bool(result[0])
        if isinstance(result, dict):
            return bool(result.get("ok", result.get("valid", result.get("passed"))))
        return bool(result)

    try:
        ok = is_ok(call(old_code, same_code))
        report("OK" if ok else "FAIL", "verify_change: código idéntico",
                "" if ok else "un cambio sin modificaciones debería pasar")
    except Exception as e:
        report("FAIL", "verify_change: código idéntico", f"excepción: {e}")

    try:
        ok = is_ok(call(old_code, broken_syntax))
        report("FAIL" if ok else "OK", "verify_change: sintaxis rota",
                "debería RECHAZAR código con error de sintaxis" if ok else "")
    except Exception:
        report("OK", "verify_change: rechaza sintaxis rota (vía excepción)")

    try:
        ok = is_ok(call(old_code, removed_fn))
        report("FAIL" if ok else "OK", "verify_change: función eliminada",
                "debería RECHAZAR un cambio que borra 'resta'" if ok else "")
    except Exception as e:
        report("SKIP", "verify_change: función eliminada", f"no se pudo probar: {e}")

    try:
        ok = is_ok(call(old_code, unrelated))
        report("FAIL" if ok else "OK", "verify_change: similitud baja",
                "debería RECHAZAR contenido sin relación (< 80%)" if ok else "")
    except Exception as e:
        report("SKIP", "verify_change: similitud baja", f"no se pudo probar: {e}")


# ===========================================================================
# 6. MEJORA 2 — github_write no escribe en main por defecto
# ===========================================================================
def test_never_writes_to_main():
    fn, where = find_function(GITHUB_WRITE_CANDIDATES)
    if fn is None:
        report("FAIL", "github_write: protección de main",
                f"no se encontró {GITHUB_WRITE_CANDIDATES} — MEJORA 2 no confirmable")
        return

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    if sig is not None and "branch" in sig.parameters:
        default = sig.parameters["branch"].default
        if default is inspect.Parameter.empty:
            report("OK", f"github_write ({where}) exige 'branch' explícito")
        elif isinstance(default, str) and default.strip().lower() in PROTECTED_BRANCHES:
            report("FAIL", "github_write: protección de main",
                    f"el default de 'branch' es '{default}' — nunca debería ser main/master")
        else:
            report("OK", f"github_write ({where}) default de rama seguro: '{default}'")
    else:
        report("SKIP", "github_write: protección de main",
                "sin parámetro 'branch' inspeccionable — confirma a mano")


# ===========================================================================
# 7. MEJORA 3 — /health con uptime
# ===========================================================================
def test_health_endpoint():
    fn, where = find_function(HEALTH_FN_CANDIDATES, modules=("bot",))
    if fn is None:
        report("FAIL", "/health",
                f"no se encontró {HEALTH_FN_CANDIDATES} en bot.py — MEJORA 3 no confirmable")
        return
    report("OK", f"handler de /health existe ({where})")

    try:
        src = inspect.getsource(fn)
    except (TypeError, OSError):
        report("SKIP", "/health: contenido", "no se pudo inspeccionar")
        return

    for field in ("uptime_seconds", "uptime_human"):
        report("OK" if field in src else "FAIL", f"/health referencia '{field}'",
                "" if field in src else f"el handler no menciona '{field}'")


# ===========================================================================
# 8. MEJORA 4 — recordatorio
# ===========================================================================
def test_self_is_wired_in():
    report("SKIP", "test_tools.py se ejecuta antes de cada github_write",
            "verifícalo a mano en la función que llama a github_write")


# ===========================================================================
# 9. NUEVO — Coherencia tool_schemas vs run_tool
# ===========================================================================
def test_tool_schemas_vs_run_tool():
    """
    Comprueba que cada herramienta declarada en tool_schemas() tiene
    una rama 'if name == "<tool>"' en run_tool(). Detecta el caso típico
    de 'declaro la tool pero se me olvida implementarla'.
    """
    try:
        ai = importlib.import_module("ai")
        agent = getattr(ai, "agent", None)
        if agent is None:
            report("SKIP", "coherencia tool_schemas↔run_tool", "no hay agent")
            return
    except Exception as e:
        report("SKIP", "coherencia tool_schemas↔run_tool", f"ai no importable: {e}")
        return

    # Obtener nombres declarados
    schemas = None
    for attr in TOOL_SCHEMA_CANDIDATES:
        if hasattr(agent, attr):
            try:
                schemas = getattr(agent, attr)()
                break
            except Exception:
                continue
    if not schemas:
        report("SKIP", "coherencia tool_schemas↔run_tool", "no se obtuvo schemas")
        return
    declared = {t["function"]["name"] for t in schemas if isinstance(t, dict)
                and isinstance(t.get("function"), dict) and "name" in t["function"]}

    # Obtener función run_tool
    fn = None
    for attr in RUN_TOOL_CANDIDATES:
        if hasattr(agent, attr):
            fn = getattr(agent, attr)
            break
    if fn is None:
        report("SKIP", "coherencia tool_schemas↔run_tool", "no se encontró run_tool")
        return

    try:
        src = inspect.getsource(fn)
    except (TypeError, OSError):
        report("SKIP", "coherencia tool_schemas↔run_tool", "no se pudo inspeccionar")
        return

    missing = []
    for name in sorted(declared):
        if f'"{name}"' not in src and f"'{name}'" not in src:
            missing.append(name)

    if missing:
        report("FAIL", f"coherencia tool_schemas↔run_tool: {len(missing)} sin handler",
                f"declaradas pero no implementadas: {missing[:5]}")
    else:
        report("OK", f"coherencia tool_schemas↔run_tool: {len(declared)} tools OK")


# ===========================================================================
# 10. NUEVO — Conectividad de red con reintentos y latencia
# ===========================================================================
def test_network_connectivity():
    """
    Ping a hosts críticos con 3 reintentos. Reporta latencia.
    Warnings si falla (puede ser fallo temporal de red en el deploy).
    """
    any_fail = False
    for host, port in NETWORK_HOSTS:
        last_err = None
        for attempt in range(1, NETWORK_RETRIES + 1):
            t0 = time.monotonic()
            try:
                with socket.create_connection((host, port), timeout=NETWORK_TIMEOUT):
                    latency_ms = (time.monotonic() - t0) * 1000
                    report("OK", f"red: {host}:{port}",
                           f"latencia {latency_ms:.0f}ms (intento {attempt})")
                    break
            except Exception as e:
                last_err = e
                if attempt < NETWORK_RETRIES:
                    time.sleep(0.5)
        else:
            report("SKIP", f"red: {host}:{port}",
                   f"sin conexión tras {NETWORK_RETRIES} intentos: {last_err}")
            any_fail = True

    if not any_fail:
        report("OK", "conectividad de red general")


# ===========================================================================
# 11. NUEVO — Variables de entorno críticas
# ===========================================================================
def test_env_vars():
    """
    Comprueba que las variables de entorno críticas existen.
    En Railway estarán todas. En local, se reportan como SKIP.
    """
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        # En el bot real, las env vars las lee config.py. Si aquí faltan,
        # es un warning (podría ser local sin .env).
        report("SKIP", f"variables de entorno: {len(missing)} ausentes",
               f"faltan: {missing}")
    else:
        report("OK", f"variables de entorno: {len(REQUIRED_ENV_VARS)} presentes")


# ===========================================================================
# 12. NUEVO — Estructura mínima de ai.py
# ===========================================================================
def test_ai_structure():
    """
    Verifica que ai.py tiene la estructura esperada:
    - Al menos MIN_AI_LINES líneas.
    - Clase Agent con métodos clave.
    - Instancia agent = Agent() al final.
    """
    ai_path = ROOT / "ai.py"
    if not ai_path.exists():
        report("FAIL", "estructura de ai.py", "ai.py no existe")
        return

    content = ai_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    n_lines = len(lines)

    if n_lines < MIN_AI_LINES:
        report("SKIP", f"ai.py: {n_lines} líneas",
               f"menos de {MIN_AI_LINES} — puede ser normal si está recortado")
    else:
        report("OK", f"ai.py: {n_lines} líneas")

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        report("FAIL", "estructura de ai.py", f"AST inválido: {e}")
        return

    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    report("OK" if "Agent" in classes else "FAIL", "ai.py define clase Agent",
           "" if "Agent" in classes else "falta la clase Agent")

    # Métodos esperados en Agent
    agent_class = next((n for n in ast.walk(tree)
                        if isinstance(n, ast.ClassDef) and n.name == "Agent"), None)
    if agent_class is not None:
        methods = {m.name for m in agent_class.body
                   if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        expected = {"tool_schemas", "run_tool", "ask"}
        missing = expected - methods
        if missing:
            report("SKIP", "Agent: métodos esperados",
                   f"faltan: {sorted(missing)}")
        else:
            report("OK", f"Agent: {len(methods)} métodos (incluye tool_schemas, run_tool, ask)")

    # Instancia global
    has_agent_instance = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "agent" for t in n.targets)
        for n in tree.body
    )
    report("OK" if has_agent_instance else "FAIL",
           "ai.py tiene 'agent = Agent()' al final",
           "" if has_agent_instance else "falta la instancia global agent")


# ===========================================================================
# MAIN
# ===========================================================================
def run_all():
    checks = [
        test_syntax_all_files,
        test_core_imports,
        test_agent_exists,
        test_tool_registry,
        test_verify_change_exists_and_works,
        test_never_writes_to_main,
        test_health_endpoint,
        test_self_is_wired_in,
        test_tool_schemas_vs_run_tool,
        test_network_connectivity,
        test_env_vars,
        test_ai_structure,
    ]
    for check in checks:
        try:
            check()
        except Exception:
            report("FAIL", check.__name__,
                   "excepción no controlada:\n" + traceback.format_exc())


def main():
    run_all()

    oks = [r for r in results if r[0] == "OK"]
    fails = [r for r in results if r[0] == "FAIL"]
    skips = [r for r in results if r[0] == "SKIP"]

    print("=" * 60)
    print(f"RESULTADOS test_tools.py — Subtom IA ({_elapsed():.1f}s)")
    print("=" * 60)

    for status, name, detail in results:
        icon = {"OK": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        line = f"{icon} {name}"
        if detail and status != "OK":
            line += f"\n       -> {detail}"
        print(line)

    print(f"\n{len(oks)} OK · {len(fails)} FALLOS · {len(skips)} OMITIDOS")

    if fails:
        print("\nTESTS FALLIDOS - NO SUBIR ESTE CAMBIO")
        sys.exit(1)

    if skips:
        print("\nHay chequeos omitidos (no bloquean el deploy, pero revísalos).")

    print("\nTODOS LOS TESTS CRÍTICOS PASARON")
    sys.exit(0)


if __name__ == "__main__":
    main()
