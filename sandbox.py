from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURACIÓN
# ============================================================

SAFE_COMMANDS = {
    # Navegación y sistema
    "ls", "pwd", "cd", "tree", "find", "which", "whereis", "stat",
    "file", "du", "df", "date", "cal", "uptime", "whoami", "id",
    "env", "printenv", "uname", "hostname", "free", "top", "ps",
    # Texto
    "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep",
    "awk", "sed", "sort", "uniq", "cut", "tr", "diff", "cmp", "tee",
    "echo", "printf", "yes", "fold", "fmt", "expand", "nl", "paste",
    "join", "split", "csplit", "comm", "shuf", "base64", "md5sum",
    "sha1sum", "sha256sum", "sha512sum", "cksum", "strings", "xxd",
    # Compresión
    "zip", "unzip", "tar", "gzip", "gunzip", "bzip2", "bunzip2",
    "xz", "unxz", "7z", "zstd", "unzstd",
    # Red
    "curl", "wget", "ping", "traceroute", "nslookup", "dig", "host",
    "netstat", "ss", "ip", "ifconfig", "arp",
    # Desarrollo
    "python", "python3", "pip", "pip3", "node", "npm", "npx", "yarn",
    "deno", "bun", "ruby", "gem", "go", "cargo", "rustc", "java",
    "javac", "javap", "kotlin", "swift", "php", "composer", "lua",
    "perl", "r", "julia",
    # Build y test
    "make", "cmake", "gcc", "g++", "clang", "clang++", "ld", "ar",
    "pytest", "tox", "unittest", "nose", "jest", "mocha", "vitest",
    "eslint", "prettier", "black", "ruff", "flake8", "mypy", "pylint",
    # Data
    "jq", "yq", "csvkit", "sqlite3", "psql", "mysql", "redis-cli",
    "mongo", "mongosh",
    # Multimedia
    "ffmpeg", "ffprobe", "convert", "magick", "identify", "mogrify",
    "sox", "opusenc", "lame", "flac",
    # Git
    "git", "gh", "glab",
    # Documentos
    "pandoc", "pdftotext", "pdfinfo", "pdftoppm", "tesseract",
    # Otros
    "man", "help", "type", "test", "[", "sleep", "timeout", "watch",
    "xargs", "parallel", "expr", "bc", "dc", "seq", "factor",
}

DANGEROUS_COMMANDS = {
    "rm", "rmdir", "mv", "cp", "chmod", "chown", "ln", "mkdir", "touch",
    "truncate", "install", "kill", "killall", "pkill", "dd",
    "apt", "apt-get", "yum", "dnf", "pacman", "apk", "brew",
    "docker", "podman", "kubectl", "systemctl", "service",
    "ssh", "scp", "rsync", "sftp", "ftp", "telnet",
}

ALLOWED_CWD = ["/tmp", "/app", "/workspace", os.getcwd()]

MAX_TIMEOUT = 60.0
DEFAULT_TIMEOUT = 15.0
MAX_OUTPUT = 100000
MAX_CONCURRENT = 4


# ============================================================
# AUDITORÍA
# ============================================================

@dataclass
class AuditEntry:
    timestamp: float
    kind: str
    content: str
    result: str
    details: str = ""


class AuditLog:
    def __init__(self, max_size: int = 1000):
        self._entries: deque[AuditEntry] = deque(maxlen=max_size)

    def add(self, kind: str, content: str, result: str, details: str = "") -> None:
        self._entries.append(AuditEntry(
            timestamp=time.time(),
            kind=kind,
            content=content[:500],
            result=result,
            details=details[:500],
        ))

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return [
            {"t": e.timestamp, "kind": e.kind, "content": e.content,
             "result": e.result, "details": e.details}
            for e in list(self._entries)[-n:]
        ]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._entries:
            counts[e.result] = counts.get(e.result, 0) + 1
        return counts

    def clear(self) -> None:
        self._entries.clear()


# ============================================================
# PERMISOS
# ============================================================

@dataclass
class PermissionRequest:
    id: str
    kind: str
    content: str
    reason: str
    created_at: float = field(default_factory=time.time)
    approved: bool | None = None


class PermissionManager:
    def __init__(self):
        self._pending: dict[str, PermissionRequest] = {}
        self._approved: set[str] = set()

    def _make_id(self, content: str) -> str:
        return hashlib.blake2b(content.encode(), digest_size=6).hexdigest()

    def request(self, kind: str, content: str, reason: str) -> PermissionRequest:
        req_id = self._make_id(content)
        req = PermissionRequest(id=req_id, kind=kind, content=content, reason=reason)
        self._pending[req_id] = req
        return req

    def approve(self, req_id: str, remember: bool = False) -> bool:
        req = self._pending.get(req_id)
        if not req:
            return False
        req.approved = True
        if remember:
            self._approved.add(req.content)
        return True

    def deny(self, req_id: str) -> bool:
        req = self._pending.get(req_id)
        if not req:
            return False
        req.approved = False
        return True

    def is_approved(self, content: str) -> bool:
        return content in self._approved

    def pending(self) -> list[dict[str, Any]]:
        return [
            {"id": r.id, "kind": r.kind, "content": r.content,
             "reason": r.reason, "created_at": r.created_at}
            for r in self._pending.values()
            if r.approved is None
        ]


# ============================================================
# SANDBOX
# ============================================================

class Sandbox:
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_output: int = MAX_OUTPUT,
    ):
        self.timeout = min(timeout, MAX_TIMEOUT)
        self.max_output = max_output
        self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        self.audit = AuditLog()
        self.permissions = PermissionManager()

    # --------------------------------------------------------
    # EJECUCIÓN GENÉRICA
    # --------------------------------------------------------

    async def _exec(
        self,
        args: list[str],
        cwd: str | None = None,
        env_extra: dict | None = None,
        stdin: str | None = None,
    ) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if env_extra:
            env.update(env_extra)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin else None,
                cwd=cwd,
                env=env,
            )
        except FileNotFoundError:
            return {"error": f"Comando no encontrado: {args[0]}"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin else None),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {"error": f"Timeout {self.timeout}s", "returncode": -1}

        return {
            "stdout": stdout.decode("utf-8", "replace")[: self.max_output],
            "stderr": stderr.decode("utf-8", "replace")[: self.max_output],
            "returncode": proc.returncode,
        }

    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    async def run_python(
        self,
        code: str,
        cwd: str | None = None,
        stdin: str | None = None,
    ) -> dict[str, Any]:
        async with self._sem:
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "script.py"
                script.write_text(code, encoding="utf-8")

                env = {
                    "PYTHONPATH": "",
                    "HOME": tmp,
                    "TMPDIR": tmp,
                    "LANG": "C.UTF-8",
                }

                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-u", str(script),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.PIPE if stdin else None,
                        cwd=cwd or tmp,
                        env={**os.environ, **env},
                    )
                except Exception as exc:
                    self.audit.add("python", code, "error", str(exc))
                    return {"error": f"No se pudo lanzar: {exc}"}

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(stdin.encode() if stdin else None),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    self.audit.add("python", code, "timeout")
                    return {"error": f"Timeout {self.timeout}s", "returncode": -1}

                result = {
                    "stdout": stdout.decode("utf-8", "replace")[: self.max_output],
                    "stderr": stderr.decode("utf-8", "replace")[: self.max_output],
                    "returncode": proc.returncode,
                }
                self.audit.add("python", code,
                               "ok" if proc.returncode == 0 else "error")
                return result

    # --------------------------------------------------------
    # NODE
    # --------------------------------------------------------

    async def run_node(self, code: str, cwd: str | None = None) -> dict[str, Any]:
        async with self._sem:
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "script.js"
                script.write_text(code, encoding="utf-8")
                self.audit.add("node", code, "start")
                return await self._exec(["node", str(script)], cwd=cwd or tmp)

    # --------------------------------------------------------
    # SHELL
    # --------------------------------------------------------

    async def run_shell(
        self,
        command: str,
        cwd: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        parts = command.strip().split()
        if not parts:
            return {"error": "Comando vacío"}

        first = parts[0].lower()

        if first in DANGEROUS_COMMANDS and not force:
            if not self.permissions.is_approved(command):
                req = self.permissions.request(
                    "shell", command,
                    f"Comando '{first}' requiere permiso"
                )
                self.audit.add("shell", command, "blocked", "requiere permiso")
                return {
                    "requires_permission": True,
                    "request_id": req.id,
                    "command": command,
                    "message": f"Comando peligroso '{first}'. Requiere aprobación.",
                }

        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            except Exception as exc:
                self.audit.add("shell", command, "error", str(exc))
                return {"error": f"No se pudo lanzar: {exc}"}

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                self.audit.add("shell", command, "timeout")
                return {"error": f"Timeout {self.timeout}s"}

            result = {
                "stdout": stdout.decode("utf-8", "replace")[: self.max_output],
                "stderr": stderr.decode("utf-8", "replace")[: self.max_output],
                "returncode": proc.returncode,
            }
            self.audit.add("shell", command,
                           "ok" if proc.returncode == 0 else "error")
            return result

    async def run_bash(self, script: str, cwd: str | None = None) -> dict[str, Any]:
        async with self._sem:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "script.sh"
                path.write_text(script, encoding="utf-8")
                os.chmod(path, 0o755)
                self.audit.add("bash", script, "start")
                return await self._exec(["bash", str(path)], cwd=cwd or tmp)

    # --------------------------------------------------------
    # INSTALAR PAQUETES
    # --------------------------------------------------------

    async def pip_install(self, packages: list[str]) -> dict[str, Any]:
        self.audit.add("pip", " ".join(packages), "start")
        async with self._sem:
            return await self._exec(
                [sys.executable, "-m", "pip", "install",
                 "--user", "--no-cache-dir", *packages],
                env_extra={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            )

    async def pip_list(self) -> dict[str, Any]:
        return await self._exec([sys.executable, "-m", "pip", "list", "--format=json"])

    async def npm_install(self, packages: list[str]) -> dict[str, Any]:
        self.audit.add("npm", " ".join(packages), "start")
        async with self._sem:
            return await self._exec(["npm", "install", *packages])

    # --------------------------------------------------------
    # UTILIDADES ESPECIALIZADAS
    # --------------------------------------------------------

    async def analyze_json(self, json_str: str) -> dict[str, Any]:
        code = f"""
import json
data = json.loads({json_str!r})
print(json.dumps({{
    'type': type(data).__name__,
    'size': len(data) if hasattr(data, '__len__') else None,
    'keys': list(data.keys()) if isinstance(data, dict) else None,
    'preview': data if isinstance(data, (int, float, str, bool, type(None))) else str(data)[:500]
}}, indent=2, ensure_ascii=False, default=str))
"""
        return await self.run_python(code)

    async def analyze_csv(self, csv_path: str) -> dict[str, Any]:
        code = f"""
import csv
with open({csv_path!r}, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    rows = list(r)
print(f"Filas: {{len(rows)}}")
if rows:
    print(f"Columnas: {{len(rows[0])}}")
    print(f"Cabecera: {{rows[0]}}")
    print(f"Primeras 3 filas:")
    for row in rows[1:4]:
        print(row)
"""
        return await self.run_python(code)

    async def hash_text(self, text: str, algorithm: str = "sha256") -> dict[str, Any]:
        try:
            h = hashlib.new(algorithm)
            h.update(text.encode("utf-8"))
            return {"algorithm": algorithm, "hash": h.hexdigest()}
        except Exception as exc:
            return {"error": str(exc)}

    async def base64_encode(self, text: str) -> dict[str, Any]:
        import base64
        return {"encoded": base64.b64encode(text.encode()).decode()}

    async def base64_decode(self, encoded: str) -> dict[str, Any]:
        import base64
        try:
            return {"decoded": base64.b64decode(encoded).decode("utf-8", "replace")}
        except Exception as exc:
            return {"error": str(exc)}

    async def regex_test(
        self, pattern: str, text: str, flags: str = ""
    ) -> dict[str, Any]:
        import re
        flag_map = {
            "i": re.IGNORECASE, "m": re.MULTILINE,
            "s": re.DOTALL, "x": re.VERBOSE,
        }
        f = 0
        for c in flags.lower():
            f |= flag_map.get(c, 0)
        try:
            r = re.compile(pattern, f)
            matches = [
                {"match": m.group(0), "groups": list(m.groups()),
                 "start": m.start(), "end": m.end()}
                for m in r.finditer(text)
            ]
            return {"count": len(matches), "matches": matches[:50]}
        except Exception as exc:
            return {"error": str(exc)}

    async def http_request(
        self, url: str, method: str = "GET",
        headers: dict | None = None, body: str | None = None,
    ) -> dict[str, Any]:
        code = f"""
import urllib.request, json
req = urllib.request.Request(
    {url!r},
    method={method!r},
    headers={headers or {}},
    data={(body.encode() if body else None)!r} if {body!r} else None
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        content = r.read().decode('utf-8', 'replace')
        print(json.dumps({{
            'status': r.status,
            'headers': dict(r.headers),
            'body': content[:5000]
        }}, indent=2))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""
        return await self.run_python(code)

    async def convert_image(
        self, input_path: str, output_path: str, size: str | None = None,
    ) -> dict[str, Any]:
        args = ["convert", input_path]
        if size:
            args += ["-resize", size]
        args.append(output_path)
        return await self._exec(args)

    async def image_info(self, path: str) -> dict[str, Any]:
        return await self._exec(["identify", path])

    async def extract_text_pdf(self, path: str) -> dict[str, Any]:
        return await self._exec(["pdftotext", path, "-"])

    async def ocr_image(self, path: str, lang: str = "spa+eng") -> dict[str, Any]:
        return await self._exec(["tesseract", path, "-", "-l", lang])

    async def ffprobe_info(self, path: str) -> dict[str, Any]:
        return await self._exec([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ])

    async def git_clone(self, repo_url: str, dest: str | None = None) -> dict[str, Any]:
        args = ["git", "clone", "--depth", "1", repo_url]
        if dest:
            args.append(dest)
        return await self._exec(args, cwd="/tmp")

    async def git_status(self, cwd: str) -> dict[str, Any]:
        return await self._exec(["git", "status"], cwd=cwd)

    async def git_log(self, cwd: str, n: int = 10) -> dict[str, Any]:
        return await self._exec(
            ["git", "log", f"-{n}", "--oneline"], cwd=cwd
        )

    async def unzip(self, path: str, dest: str) -> dict[str, Any]:
        os.makedirs(dest, exist_ok=True)
        return await self._exec(["unzip", "-o", path, "-d", dest])

    async def create_zip(self, source: str, output: str) -> dict[str, Any]:
        return await self._exec(["zip", "-r", output, source])

    async def download(self, url: str, output: str) -> dict[str, Any]:
        return await self._exec(["curl", "-L", "-o", output, url])

    async def sqlite_query(self, db_path: str, query: str) -> dict[str, Any]:
        return await self._exec(["sqlite3", db_path, query])

    async def run_tests(self, cwd: str) -> dict[str, Any]:
        return await self._exec(["pytest", "-v", "--tb=short"], cwd=cwd)

    async def which(self, program: str) -> str | None:
        return shutil.which(program)

    # --------------------------------------------------------
    # PERMISOS
    # --------------------------------------------------------

    def approve(self, request_id: str, remember: bool = False) -> bool:
        return self.permissions.approve(request_id, remember)

    def deny(self, request_id: str) -> bool:
        return self.permissions.deny(request_id)

    def pending_permissions(self) -> list[dict[str, Any]]:
        return self.permissions.pending()

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    def available_commands(self) -> dict[str, Any]:
        return {
            "safe": sorted(c for c in SAFE_COMMANDS if shutil.which(c)),
            "dangerous": sorted(c for c in DANGEROUS_COMMANDS if shutil.which(c)),
        }

    def audit_recent(self, n: int = 20) -> list[dict[str, Any]]:
        return self.audit.recent(n)

    def audit_stats(self) -> dict[str, int]:
        return self.audit.stats()

    def info(self) -> dict[str, Any]:
        return {
            "timeout": self.timeout,
            "max_output": self.max_output,
            "max_concurrent": MAX_CONCURRENT,
            "capabilities": [
                "python", "node", "bash", "shell",
                "pip_install", "npm_install",
                "analyze_json", "analyze_csv",
                "hash_text", "base64_encode", "base64_decode",
                "regex_test", "http_request",
                "convert_image", "image_info", "extract_text_pdf",
                "ocr_image", "ffprobe_info",
                "git_clone", "git_status", "git_log",
                "unzip", "create_zip", "download",
                "sqlite_query", "run_tests",
            ],
            "safe_commands": sorted(c for c in SAFE_COMMANDS if shutil.which(c))[:50],
            "dangerous_commands": sorted(c for c in DANGEROUS_COMMANDS if shutil.which(c)),
            "audit_stats": self.audit_stats(),
            "pending_permissions": len(self.permissions.pending()),
        }


sandbox = Sandbox()
