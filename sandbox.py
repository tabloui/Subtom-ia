from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp


class Sandbox:
    def __init__(
        self,
        timeout: float = 15.0,
        max_output: int = 100000,
    ):
        self.timeout = timeout
        self.max_output = max_output
        self._sem = asyncio.Semaphore(4)

    # =========================================================
    # EJECUCIÓN DE CÓDIGO
    # =========================================================

    async def _exec(self, cmd: list[str], stdin: str | None = None) -> dict[str, Any]:
        """Ejecuta un comando y devuelve stdout, stderr y returncode."""
        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(input=(stdin.encode() if stdin else None)),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return {"error": f"Timeout ({self.timeout}s)", "stdout": "", "stderr": "", "returncode": -1}
                stdout = stdout_b.decode("utf-8", "replace")[: self.max_output]
                stderr = stderr_b.decode("utf-8", "replace")[: self.max_output]
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": proc.returncode,
                }
            except FileNotFoundError as exc:
                return {"error": f"Comando no encontrado: {exc}", "stdout": "", "stderr": "", "returncode": -1}
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}", "stdout": "", "stderr": "", "returncode": -1}

    async def run_python(self, code: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            path = f.name
        try:
            return await self._exec([sys.executable, path])
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    async def run_shell(self, command: str) -> dict[str, Any]:
        return await self._exec(["bash", "-c", command])

    async def run_bash(self, script: str) -> dict[str, Any]:
        return await self._exec(["bash", "-c", script])

    async def run_node(self, code: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
            path = f.name
        try:
            return await self._exec(["node", path])
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    async def pip_install(self, packages: list[str]) -> dict[str, Any]:
        if not packages:
            return {"error": "Sin paquetes que instalar."}
        return await self._exec([sys.executable, "-m", "pip", "install", *packages])

    # =========================================================
    # UTILIDADES
    # =========================================================

    async def analyze_json(self, json_str: str) -> dict[str, Any]:
        try:
            data = json.loads(json_str)
            return {"ok": True, "type": type(data).__name__, "keys": list(data.keys()) if isinstance(data, dict) else None, "length": len(data) if isinstance(data, (list, dict, str)) else None}
        except Exception as exc:
            return {"error": f"JSON inválido: {exc}"}

    async def regex_test(self, pattern: str, text: str, flags: str = "") -> dict[str, Any]:
        try:
            fl = 0
            if "i" in flags:
                fl |= re.IGNORECASE
            if "m" in flags:
                fl |= re.MULTILINE
            if "s" in flags:
                fl |= re.DOTALL
            rx = re.compile(pattern, fl)
            matches = [m.group(0) for m in rx.finditer(text)]
            return {"ok": True, "matches": matches, "count": len(matches)}
        except Exception as exc:
            return {"error": f"Regex inválida: {exc}"}

    async def http_request(self, url: str, method: str = "GET", body: str | None = None) -> dict[str, Any]:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                kwargs: dict[str, Any] = {}
                if body:
                    kwargs["data"] = body
                async with session.request(method.upper(), url, **kwargs) as resp:
                    text = await resp.text()
                    return {"status": resp.status, "headers": dict(resp.headers), "body": text[: self.max_output]}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def hash_text(self, text: str, algorithm: str = "sha256") -> dict[str, Any]:
        try:
            h = hashlib.new(algorithm)
            h.update(text.encode("utf-8"))
            return {"ok": True, "algorithm": algorithm, "hash": h.hexdigest()}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def ocr_image(self, path: str, lang: str = "spa+eng") -> dict[str, Any]:
        return await self._exec(["tesseract", path, "stdout", "-l", lang])

    async def extract_text_pdf(self, path: str) -> dict[str, Any]:
        code = f"from pypdf import PdfReader; r=PdfReader({path!r}); print('\\n'.join((p.extract_text() or '') for p in r.pages))"
        return await self.run_python(code)

    async def git_clone(self, repo_url: str, dest: str | None = None) -> dict[str, Any]:
        cmd = ["git", "clone", repo_url]
        if dest:
            cmd.append(dest)
        return await self._exec(cmd)

    async def download(self, url: str, output: str) -> dict[str, Any]:
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return {"error": f"HTTP {resp.status}"}
                    data = await resp.read()
                    Path(output).write_bytes(data)
                    return {"ok": True, "path": output, "size": len(data)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # =========================================================
    # VERIFICACIÓN DE CAMBIOS EN PYTHON (MEJORA 1)
    # =========================================================

    async def verify_python_change(self, file_path: str, old_content: str, new_content: str) -> dict[str, Any]:
        """
        Verifica un cambio en un archivo Python:
        - Sintaxis con ast.parse y py_compile.
        - Similitud mínima (80%).
        - Clases, funciones e imports eliminados.

        Devuelve SIEMPRE:
            {"ok": bool, "errors": [...], "warnings": [...], "old_lines": int, "new_lines": int}
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Sintaxis
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            return {"ok": False, "errors": [f"SINTAXIS: {e}"], "warnings": [], "old_lines": 0, "new_lines": 0}

        # 2. py_compile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.py"
            path.write_text(new_content, encoding="utf-8")
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as e:
                return {"ok": False, "errors": [f"COMPILE: {e}"], "warnings": [], "old_lines": 0, "new_lines": 0}

        old_lines_count = len(old_content.splitlines())
        new_lines_count = len(new_content.splitlines())

        # 3. Similitud (80%, más permisivo para permitir añadir funciones)
        old_set = set(old_content.splitlines())
        new_set = set(new_content.splitlines())
        if old_set:
            similarity = len(old_set & new_set) / max(len(old_set), len(new_set))
            if similarity < 0.80:
                errors.append(f"SIMILITUD BAJA: {similarity:.2%} (< 80%)")

        # 4. Estructura AST
        try:
            old_tree = ast.parse(old_content)
            new_tree = ast.parse(new_content)
        except SyntaxError as e:
            return {"ok": False, "errors": [f"AST: {e}"], "warnings": [], "old_lines": old_lines_count, "new_lines": new_lines_count}

        old_classes = {n.name for n in ast.walk(old_tree) if isinstance(n, ast.ClassDef)}
        new_classes = {n.name for n in ast.walk(new_tree) if isinstance(n, ast.ClassDef)}
        missing_classes = old_classes - new_classes
        if missing_classes:
            errors.append(f"CLASES ELIMINADAS: {sorted(missing_classes)}")

        old_funcs = {n.name for n in ast.walk(old_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        new_funcs = {n.name for n in ast.walk(new_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing_funcs = old_funcs - new_funcs
        if missing_funcs:
            errors.append(f"FUNCIONES ELIMINADAS: {sorted(missing_funcs)}")

        def extract_imports(tree):
            result = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result.add(node.module)
            return result

        old_imports = extract_imports(old_tree)
        new_imports = extract_imports(new_tree)
        missing_imports = old_imports - new_imports
        if missing_imports:
            warnings.append(f"IMPORTS ELIMINADOS: {sorted(missing_imports)}")

        # 5. Archivo cortado drásticamente
        if old_lines_count > 20 and new_lines_count < old_lines_count * 0.9:
            errors.append(f"ARCHIVO CORTADO: {old_lines_count} -> {new_lines_count} lineas")

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "old_lines": old_lines_count,
            "new_lines": new_lines_count,
        }


sandbox = Sandbox()
