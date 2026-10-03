#!/usr/bin/env python3
"""
test_tools.py — Gate de seguridad que corre dentro de E2B.

Uso:
    python test_tools.py <nombre_archivo> <ruta_repo_github> <token_github>

Hace:
    1. Clona el repo de GitHub dentro de E2B.
    2. Analiza los .py del repo con AST (sin importarlos).
    3. Verifica que el archivo a subir es válido.
    4. Sale con "TESTS FALLIDOS" si algo va mal.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def _log(msg):
    print(msg, flush=True)


def _fail(msg):
    _log(f"[FAIL] {msg}")


def _ok(msg):
    _log(f"[OK] {msg}")


def _warn(msg):
    _log(f"[WARN] {msg}")


# ============================================================
# ANÁLISIS DE ARCHIVOS
# ============================================================

def check_python_syntax(content, filename="<file>"):
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, [f"SINTAXIS {filename}: {e}"], None
    try:
        compile(content, filename, "exec")
    except Exception as e:
        return False, [f"COMPILE {filename}: {e}"], None
    return True, [], tree


def check_json(content):
    try:
        data = json.loads(content)
    except Exception as e:
        return False, [f"JSON inválido: {e}"]
    _ok(f"JSON: tipo {type(data).__name__}")
    return True, []


def check_yaml(content):
    try:
        import yaml
    except ImportError:
        _ok("YAML: PyYAML no instalado, se omite")
        return True, []
    try:
        yaml.safe_load(content)
    except Exception as e:
        return False, [f"YAML inválido: {e}"]
    _ok("YAML válido")
    return True, []


def check_toml(content):
    try:
        tomllib.loads(content)
    except Exception as e:
        return False, [f"TOML inválido: {e}"]
    _ok("TOML válido")
    return True, []


def check_markdown(content):
    if not content.strip():
        return False, ["MARKDOWN vacío"]
    _ok(f"Markdown: {len(content.splitlines())} líneas")
    return True, []


def check_text(content):
    if not content.strip():
        return False, ["ARCHIVO vacío"]
    if "\x00" in content:
        return False, ["ARCHIVO contiene bytes nulos"]
    _ok(f"Texto: {len(content.splitlines())} líneas")
    return True, []


def check_unknown(content):
    if not content.strip():
        return False, ["ARCHIVO vacío"]
    if "\x00" in content[:4096]:
        return False, ["ARCHIVO contiene bytes nulos"]
    _ok(f"Binario/texto: {len(content)} bytes")
    return True, []


CHECKERS = {
    ".py": check_python_syntax,
    ".json": check_json,
    ".yaml": check_yaml,
    ".yml": check_yaml,
    ".toml": check_toml,
    ".md": check_markdown,
    ".markdown": check_markdown,
    ".txt": check_text,
    ".env": check_text,
    ".cfg": check_text,
    ".ini": check_text,
    ".html": check_text,
    ".css": check_text,
    ".js": check_text,
    ".ts": check_text,
    ".sh": check_text,
    ".bash": check_text,
    ".sql": check_text,
    ".csv": check_text,
    ".xml": check_text,
    ".svg": check_text,
    ".log": check_text,
}


# ============================================================
# ANÁLISIS DEL REPO (AST, sin importar)
# ============================================================

def analyze_ai_py(repo_path):
    """Analiza ai.py con AST. No importa, solo parsea."""
    errors = []
    ai_path = repo_path / "ai.py"

    if not ai_path.exists():
        return False, ["ai.py no existe en el repo"]

    try:
        content = ai_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, [f"ai.py no se pudo leer: {e}"]

    lines = len(content.splitlines())
    _ok(f"ai.py: {lines} líneas")

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, [f"ai.py sintaxis rota: {e}"]

    # Clase Agent
    agent_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Agent":
            agent_class = node
            break

    if agent_class is None:
        errors.append("ai.py no define clase Agent")
    else:
        _ok("ai.py: clase Agent encontrada")
        methods = {
            m.name
            for m in agent_class.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        expected = {"tool_schemas", "run_tool", "ask"}
        missing = expected - methods
        if missing:
            errors.append(f"Agent sin métodos: {sorted(missing)}")
        else:
            _ok(f"Agent: {len(methods)} métodos (tool_schemas, run_tool, ask presentes)")

    # Instancia agent = Agent()
    has_agent_instance = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "agent" for t in n.targets)
        for n in tree.body
    )
    if has_agent_instance:
        _ok("ai.py: 'agent = Agent()' presente")
    else:
        errors.append("ai.py no tiene 'agent = Agent()' al final")

    # Contar herramientas en tool_schemas
    if agent_class is not None:
        tool_schemas_method = None
        for m in agent_class.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "tool_schemas":
                tool_schemas_method = m
                break

        if tool_schemas_method is not None:
            tool_names = set()
            for node in ast.walk(tool_schemas_method):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == "name":
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                tool_names.add(value.value)

            if tool_names:
                _ok(f"tool_schemas: {len(tool_names)} herramientas declaradas")
            else:
                _warn("tool_schemas: no se pudieron contar herramientas (formato inesperado)")

            # Coherencia con run_tool
            run_tool_method = None
            for m in agent_class.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "run_tool":
                    run_tool_method = m
                    break

            if run_tool_method is not None:
                try:
                    run_tool_src = ast.unparse(run_tool_method)
                except Exception:
                    run_tool_src = ""

                missing_handlers = []
                for tool in sorted(tool_names):
                    if f'"{tool}"' not in run_tool_src and f"'{tool}'" not in run_tool_src:
                        missing_handlers.append(tool)

                if missing_handlers:
                    errors.append(
                        f"run_tool sin handler para {len(missing_handlers)} tools: "
                        f"{missing_handlers[:5]}"
                    )
                else:
                    _ok("Coherencia tool_schemas ↔ run_tool: OK")

    return not errors, errors


def analyze_repo_syntax(repo_path):
    """Compila todos los .py del repo."""
    errors = []
    py_files = list(repo_path.rglob("*.py"))
    py_files = [
        f for f in py_files
        if ".git" not in f.parts and "__pycache__" not in f.parts
    ]

    for f in py_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{f.name}: {e}")

    if errors:
        return False, errors
    _ok(f"Sintaxis OK en {len(py_files)} archivos .py del repo")
    return True, []


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        _fail("Uso: test_tools.py <nombre_archivo> [<repo> <token>]")
        _log("TESTS FALLIDOS")
        sys.exit(1)

    filename = sys.argv[1]
    repo = sys.argv[2] if len(sys.argv) > 2 else None
    token = sys.argv[3] if len(sys.argv) > 3 else None

    # 1. Validar el archivo que se va a subir
    if len(sys.argv) > 4 and sys.argv[4] != "--stdin":
        content = sys.argv[4]
    else:
        content = sys.stdin.read()

    suffix = Path(filename).suffix.lower()
    checker = CHECKERS.get(suffix, check_unknown)

    _log(f"==> Verificando archivo: {filename} ({suffix or 'sin ext'})")

    try:
        ok, errors = checker(content)
    except Exception as e:
        _fail(f"Checker falló: {type(e).__name__}: {e}")
        _log("TESTS FALLIDOS")
        sys.exit(1)

    if not ok:
        for err in errors:
            _fail(err)
        _log("TESTS FALLIDOS")
        sys.exit(1)

    _ok(f"Archivo {filename} válido")

    # 2. Si hay repo, clonar y analizar
    if repo:
        work_dir = Path("/tmp/_repo_check")
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

        url = f"https://{token}@github.com/{repo}.git" if token else f"https://github.com/{repo}.git"

        _log(f"==> Clonando {repo}...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(work_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            _fail("git clone timeout")
            _log("TESTS FALLIDOS")
            sys.exit(1)
        except Exception as e:
            _fail(f"git clone error: {e}")
            _log("TESTS FALLIDOS")
            sys.exit(1)

        if result.returncode != 0:
            _fail(f"git clone falló: {result.stderr[:500]}")
            _log("TESTS FALLIDOS")
            sys.exit(1)

        _ok("Repo clonado")

        # 3. Analizar sintaxis de todo el repo
        ok, errors = analyze_repo_syntax(work_dir)
        if not ok:
            for err in errors:
                _fail(err)
            _log("TESTS FALLIDOS")
            sys.exit(1)

        # 4. Analizar ai.py
        ok, errors = analyze_ai_py(work_dir)
        if not ok:
            for err in errors:
                _fail(err)
            _log("TESTS FALLIDOS")
            sys.exit(1)

        # Limpiar
        shutil.rmtree(work_dir, ignore_errors=True)

    _log("TESTS OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
