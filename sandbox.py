from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp
from e2b_code_interpreter import AsyncSandbox


class Sandbox:

    def __init__(
        self,
        timeout: float = 30.0,
        max_output: int = 100000,
        max_download: int = 50 * 1024 * 1024,
        max_retries: int = 2,
    ):
        self.timeout = timeout
        self.max_output = max_output
        self.max_download = max_download
        self.max_retries = max_retries
        self._sandbox: AsyncSandbox | None = None
        self._lock = asyncio.Lock()

    def _success(self, **data: Any) -> dict[str, Any]:
        return {"ok": True, **data}

    def _error(
        self,
        operation: str,
        exc: Exception | str,
        *,
        retryable: bool = False,
        **data: Any,
    ) -> dict[str, Any]:
        if isinstance(exc, Exception):
            error_type = type(exc).__name__
            message = str(exc)
        else:
            error_type = "Error"
            message = str(exc)
        return {
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
                "operation": operation,
                "retryable": retryable,
            },
            **data,
        }

    def _limit(self, value: Any) -> str:
        return str(value or "")[: self.max_output]

    async def _get_sandbox(self) -> AsyncSandbox:
        async with self._lock:
            if self._sandbox is not None:
                return self._sandbox
            api_key = os.getenv("E2B_API_KEY")
            if not api_key:
                raise RuntimeError("E2B_API_KEY no configurada en las variables de entorno.")
            self._sandbox = await AsyncSandbox.create(timeout=3600, api_key=api_key)
            return self._sandbox

    async def close(self) -> None:
        async with self._lock:
            if self._sandbox is None:
                return
            try:
                await self._sandbox.kill()
            except Exception:
                pass
            finally:
                self._sandbox = None

    async def _reset_sandbox(self) -> None:
        await self.close()
        await self._get_sandbox()

    async def health_check(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            result = await sandbox.commands.run("printf 'SANDBOX_OK\\n'", timeout=10)
            duration = time.monotonic() - start
            if result.exit_code != 0:
                return self._error(
                    "health_check",
                    "El sandbox respondió con código distinto de 0.",
                    retryable=True,
                    stdout=self._limit(result.stdout),
                    stderr=self._limit(result.stderr),
                    returncode=result.exit_code,
                    duration=round(duration, 3),
                )
            return self._success(
                operation="health_check",
                status="healthy",
                stdout=self._limit(result.stdout),
                stderr=self._limit(result.stderr),
                returncode=result.exit_code,
                duration=round(duration, 3),
            )
        except Exception as exc:
            try:
                await self._reset_sandbox()
            except Exception:
                pass
            return self._error(
                "health_check",
                exc,
                retryable=True,
                duration=round(time.monotonic() - start, 3),
            )

    async def run_python(self, code: str) -> dict[str, Any]:
        if not code.strip():
            return self._error("run_python", "No se proporcionó código Python.")
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                sandbox = await self._get_sandbox()
                execution = await sandbox.run_code(code, timeout=self.timeout)
                stdout = self._limit(execution.text)
                stderr = ""
                returncode = 0
                if execution.error:
                    stderr = f"{execution.error.name}: {execution.error.value}"
                    if execution.error.traceback:
                        stderr += "\n" + execution.error.traceback
                    stderr = self._limit(stderr)
                    returncode = 1
                return {
                    "ok": returncode == 0,
                    "operation": "run_python",
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "duration": round(time.monotonic() - start, 3),
                    "attempt": attempt + 1,
                }
            except Exception as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    try:
                        await self._reset_sandbox()
                    except Exception:
                        pass
                    continue
                try:
                    await self._reset_sandbox()
                except Exception:
                    pass
                return self._error(
                    "run_python",
                    exc,
                    retryable=True,
                    stdout="",
                    stderr="",
                    returncode=-1,
                    duration=round(time.monotonic() - start, 3),
                    attempt=attempt + 1,
                )
        return self._error("run_python", "Ejecución agotada.", retryable=True)

    async def run_shell(
        self,
        command: str,
        *,
        allow_dangerous: bool = False,
    ) -> dict[str, Any]:
        if not command.strip():
            return self._error("run_shell", "No se proporcionó ningún comando.")
        if not allow_dangerous:
            blocked = self._check_dangerous_command(command)
            if blocked:
                return self._error("run_shell", blocked, retryable=False)
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            result = await sandbox.commands.run(command, timeout=self.timeout)
            return {
                "ok": result.exit_code == 0,
                "operation": "run_shell",
                "command": command,
                "stdout": self._limit(result.stdout),
                "stderr": self._limit(result.stderr),
                "returncode": result.exit_code,
                "duration": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            await self._safe_reset()
            return self._error(
                "run_shell",
                exc,
                retryable=True,
                stdout="",
                stderr="",
                returncode=-1,
                duration=round(time.monotonic() - start, 3),
            )

    def _check_dangerous_command(self, command: str) -> str | None:
        normalized = command.strip().lower()
        dangerous_patterns = [
            r"\brm\s+-rf\s+/",
            r"\brm\s+-rf\s+\*",
            r"\bmkfs\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
            r"\binit\s+0\b",
            r"\bdd\s+if=",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, normalized):
                return "Comando bloqueado por seguridad. Usa allow_dangerous=True si realmente necesitas ejecutarlo."
        return None

    async def run_bash(
        self,
        script: str,
        *,
        allow_dangerous: bool = False,
    ) -> dict[str, Any]:
        return await self.run_shell(script, allow_dangerous=allow_dangerous)

    async def run_node(self, code: str) -> dict[str, Any]:
        if not code.strip():
            return self._error("run_node", "No se proporcionó código JavaScript.")
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            command = (
                "cat > /tmp/sandbox_script.js << 'SANDBOX_EOF'\n"
                + code
                + "\nSANDBOX_EOF\n"
                "node /tmp/sandbox_script.js"
            )
            result = await sandbox.commands.run(command, timeout=self.timeout)
            return {
                "ok": result.exit_code == 0,
                "operation": "run_node",
                "stdout": self._limit(result.stdout),
                "stderr": self._limit(result.stderr),
                "returncode": result.exit_code,
                "duration": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            await self._safe_reset()
            return self._error(
                "run_node",
                exc,
                retryable=True,
                stdout="",
                stderr="",
                returncode=-1,
                duration=round(time.monotonic() - start, 3),
            )

    async def pip_install(self, packages: list[str]) -> dict[str, Any]:
        if not packages:
            return self._error("pip_install", "Sin paquetes que instalar.")
        for package in packages:
            if not re.fullmatch(
                r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:[<>=!~]=?.*)?",
                package,
            ):
                return self._error("pip_install", f"Nombre de paquete no válido: {package}")
        command = (
            "python -m pip install "
            "--disable-pip-version-check "
            "--no-input "
            + " ".join(shlex.quote(p) for p in packages)
        )
        return await self.run_shell(command)

    async def check_python(self, code: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True,
            "operation": "check_python",
            "syntax": {"ok": False},
            "compile": {"ok": False},
            "imports": [],
            "functions": [],
            "classes": [],
            "errors": [],
            "warnings": [],
        }
        try:
            tree = ast.parse(code)
            result["syntax"] = {"ok": True}
        except SyntaxError as exc:
            result["ok"] = False
            result["errors"].append(f"SINTAXIS: {exc}")
            return result
        try:
            compile(code, "<sandbox_check>", "exec")
            result["compile"] = {"ok": True}
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"COMPILE: {exc}")
        imports: set[str] = set()
        functions: set[str] = set()
        classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.add(node.name)
        result["imports"] = sorted(imports)
        result["functions"] = sorted(functions)
        result["classes"] = sorted(classes)
        return result

    async def check_python_file(self, path: str) -> dict[str, Any]:
        safe_path = shlex.quote(path)
        result = await self.run_shell(f"cat {safe_path}")
        if not result.get("ok"):
            return result
        return await self.check_python(result.get("stdout", ""))

    async def run_tests(
        self,
        path: str = ".",
        *,
        test_command: str | None = None,
    ) -> dict[str, Any]:
        if test_command:
            command = test_command
        else:
            command = f"cd {shlex.quote(path)} && python -m pytest -q"
        return await self.run_shell(command)

    async def run_lint(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && ruff check ."
        return await self.run_shell(command)

    async def compile_python_project(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && python -m compileall -q ."
        return await self.run_shell(command)

    async def full_python_check(
        self,
        path: str = ".",
        *,
        run_tests: bool = False,
        run_lint: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True,
            "operation": "full_python_check",
            "path": path,
            "checks": {},
        }
        compile_result = await self.compile_python_project(path)
        result["checks"]["compile"] = compile_result
        if not compile_result.get("ok"):
            result["ok"] = False
        if run_tests:
            tests = await self.run_tests(path)
            result["checks"]["tests"] = tests
            if not tests.get("ok"):
                result["ok"] = False
        if run_lint:
            lint = await self.run_lint(path)
            result["checks"]["lint"] = lint
            if not lint.get("ok"):
                result["ok"] = False
        return result

    async def analyze_json(self, json_str: str) -> dict[str, Any]:
        try:
            data = json.loads(json_str)
            return self._success(
                operation="analyze_json",
                type=type(data).__name__,
                keys=(list(data.keys()) if isinstance(data, dict) else None),
                length=(len(data) if isinstance(data, (list, dict, str)) else None),
            )
        except Exception as exc:
            return self._error("analyze_json", f"JSON inválido: {exc}")

    async def regex_test(
        self,
        pattern: str,
        text: str,
        flags: str = "",
    ) -> dict[str, Any]:
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
            return self._success(
                operation="regex_test",
                matches=matches,
                count=len(matches),
            )
        except Exception as exc:
            return self._error("regex_test", f"Regex inválida: {exc}")

    async def http_request(
        self,
        url: str,
        method: str = "GET",
        body: str | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout_seconds: float = 30,
    ) -> dict[str, Any]:
        method = method.upper()
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        request_headers = {"User-Agent": "Sandbox/2.0"}
        if headers:
            request_headers.update(headers)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=request_headers,
            ) as session:
                kwargs: dict[str, Any] = {}
                if body is not None:
                    kwargs["data"] = body
                if params:
                    kwargs["params"] = params
                async with session.request(method, url, **kwargs) as resp:
                    text = await resp.text()
                    content_type = resp.headers.get("Content-Type", "")
                    parsed_json = None
                    if "json" in content_type.lower():
                        try:
                            parsed_json = json.loads(text)
                        except Exception:
                            parsed_json = None
                    return self._success(
                        operation="http_request",
                        status=resp.status,
                        method=method,
                        url=str(resp.url),
                        headers=dict(resp.headers),
                        body=text[: self.max_output],
                        json=parsed_json,
                    )
        except Exception as exc:
            return self._error("http_request", exc, retryable=True)

    async def hash_text(
        self,
        text: str,
        algorithm: str = "sha256",
    ) -> dict[str, Any]:
        try:
            h = hashlib.new(algorithm)
            h.update(text.encode("utf-8"))
            return self._success(
                operation="hash_text",
                algorithm=algorithm,
                hash=h.hexdigest(),
            )
        except Exception as exc:
            return self._error("hash_text", exc)

    async def ocr_image(
        self,
        path: str,
        lang: str = "spa+eng",
    ) -> dict[str, Any]:
        command = (
            "which tesseract >/dev/null 2>&1 || "
            "(apt-get update -qq && "
            "apt-get install -y -qq "
            "tesseract-ocr "
            "tesseract-ocr-spa); "
            f"tesseract {shlex.quote(path)} "
            f"stdout -l {shlex.quote(lang)}"
        )
        return await self.run_shell(command)

    async def extract_text_pdf(self, path: str) -> dict[str, Any]:
        code = f"""
from pypdf import PdfReader
reader = PdfReader({path!r})
for page in reader.pages:
    print(page.extract_text() or "")
"""
        return await self.run_python(code)

    async def git_clone(
        self,
        repo_url: str,
        dest: str | None = None,
    ) -> dict[str, Any]:
        command = "git clone " + shlex.quote(repo_url)
        if dest:
            command += " " + shlex.quote(dest)
        return await self.run_shell(command)

    async def git_status(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && git status --short --branch"
        return await self.run_shell(command)

    async def git_diff(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && git diff"
        return await self.run_shell(command)

    async def file_exists(self, path: str) -> dict[str, Any]:
        command = f"test -e {shlex.quote(path)}"
        result = await self.run_shell(command)
        return {
            **result,
            "path": path,
            "exists": result.get("returncode") == 0,
        }

    async def read_file(self, path: str) -> dict[str, Any]:
        return await self.run_shell(f"cat {shlex.quote(path)}")

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            sandbox = await self._get_sandbox()
            command = (
                "cat > "
                + shlex.quote(path)
                + " << 'SANDBOX_EOF'\n"
                + content
                + "\nSANDBOX_EOF"
            )
            result = await sandbox.commands.run(command, timeout=self.timeout)
            return {
                "ok": result.exit_code == 0,
                "operation": "write_file",
                "path": path,
                "stdout": self._limit(result.stdout),
                "stderr": self._limit(result.stderr),
                "returncode": result.exit_code,
            }
        except Exception as exc:
            await self._safe_reset()
            return self._error("write_file", exc, retryable=True)

    async def list_files(self, path: str = ".") -> dict[str, Any]:
        command = f"find {shlex.quote(path)} -maxdepth 2 -type f -print"
        return await self.run_shell(command)

    async def search_files(
        self,
        pattern: str,
        path: str = ".",
    ) -> dict[str, Any]:
        command = (
            f"grep -RIn "
            f"--exclude-dir=.git "
            f"{shlex.quote(pattern)} "
            f"{shlex.quote(path)}"
        )
        return await self.run_shell(command)

    async def delete_file(self, path: str) -> dict[str, Any]:
        if path in {"/", ".", "..", ""}:
            return self._error("delete_file", "Ruta no permitida.")
        return await self.run_shell(f"rm -f -- {shlex.quote(path)}")

    async def download(self, url: str, output: str) -> dict[str, Any]:
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return self._error("download", f"HTTP {resp.status}")
                    total = 0
                    with open(output, "wb") as file:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            total += len(chunk)
                            if total > self.max_download:
                                return self._error("download", "Archivo demasiado grande.")
                            file.write(chunk)
                    return self._success(
                        operation="download",
                        path=output,
                        size=total,
                        status=resp.status,
                    )
        except Exception as exc:
            return self._error("download", exc, retryable=True)

    async def verify_python_change(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        old_lines_count = len(old_content.splitlines())
        new_lines_count = len(new_content.splitlines())

        try:
            ast.parse(new_content)
        except SyntaxError as exc:
            return {
                "ok": False,
                "operation": "verify_python_change",
                "file": file_path,
                "errors": [f"SINTAXIS: {exc}"],
                "warnings": [],
                "old_lines": old_lines_count,
                "new_lines": new_lines_count,
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.py"
            path.write_text(new_content, encoding="utf-8")
            try:
                import py_compile
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                return {
                    "ok": False,
                    "operation": "verify_python_change",
                    "file": file_path,
                    "errors": [f"COMPILE: {exc}"],
                    "warnings": [],
                    "old_lines": old_lines_count,
                    "new_lines": new_lines_count,
                }

        old_set = set(old_content.splitlines())
        new_set = set(new_content.splitlines())
        similarity = 1.0
        if old_set:
            similarity = len(old_set & new_set) / max(len(old_set), len(new_set))
        if similarity < 0.80:
            warnings.append(f"SIMILITUD BAJA: {similarity:.2%} (< 80%).")

        old_tree = ast.parse(old_content)
        new_tree = ast.parse(new_content)

        old_classes = {node.name for node in ast.walk(old_tree) if isinstance(node, ast.ClassDef)}
        new_classes = {node.name for node in ast.walk(new_tree) if isinstance(node, ast.ClassDef)}
        missing_classes = old_classes - new_classes
        if missing_classes:
            errors.append(f"CLASES ELIMINADAS: {sorted(missing_classes)}")

        old_funcs = {
            node.name
            for node in ast.walk(old_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        new_funcs = {
            node.name
            for node in ast.walk(new_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing_funcs = old_funcs - new_funcs
        if missing_funcs:
            errors.append(f"FUNCIONES ELIMINADAS: {sorted(missing_funcs)}")

        def extract_imports(tree: ast.AST) -> set[str]:
            result: set[str] = set()
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

        if old_lines_count > 20 and new_lines_count < old_lines_count * 0.90:
            errors.append(f"ARCHIVO CORTADO: {old_lines_count} -> {new_lines_count} líneas")

        return {
            "ok": len(errors) == 0,
            "operation": "verify_python_change",
            "file": file_path,
            "errors": errors,
            "warnings": warnings,
            "similarity": round(similarity, 4),
            "old_lines": old_lines_count,
            "new_lines": new_lines_count,
        }

    async def verify_and_test_python_change(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        *,
        project_path: str = ".",
        run_tests: bool = True,
    ) -> dict[str, Any]:
        verification = await self.verify_python_change(file_path, old_content, new_content)
        result = {
            "ok": verification.get("ok", False),
            "operation": "verify_and_test_python_change",
            "verification": verification,
            "tests": None,
        }
        if not verification.get("ok"):
            return result
        if run_tests:
            tests = await self.run_tests(project_path)
            result["tests"] = tests
            if not tests.get("ok"):
                result["ok"] = False
        return result

    async def create_zip(self, source: str, output: str) -> dict[str, Any]:
        command = f"cd {shlex.quote(source)} && zip -r {shlex.quote(output)} ."
        return await self.run_shell(command)

    async def extract_zip(self, archive: str, destination: str) -> dict[str, Any]:
        command = (
            f"mkdir -p {shlex.quote(destination)} && "
            f"unzip -o {shlex.quote(archive)} "
            f"-d {shlex.quote(destination)}"
        )
        return await self.run_shell(command)

    async def _safe_reset(self) -> None:
        try:
            await self._reset_sandbox()
        except Exception:
            pass


sandbox = Sandbox()
