#!/usr/bin/env python3
"""
file_tools.py
Herramientas de archivos para una IA como Gemini.

Incluye:
- Crear archivos con cualquier nombre y extensión.
- Leer y escribir archivos de texto.
- Copiar, mover, renombrar y borrar archivos/directorios.
- Crear ZIP.
- Extraer ZIP.
- Listar contenido de directorios y ZIP.
- Crear directorios.
- Obtener información de archivos.

NUEVO (análisis):
- Leer imágenes y devolverlas en base64 (para modelos con visión).
- Info de imágenes (ancho, alto, formato).
- Leer texto autodetectando encoding.
- Leer PDF (texto).
- Detección de tipo de archivo.
- Leer "cualquier archivo" y devolver algo útil a la IA.

NUEVO (robustez / utilidades extra):
- exists / is_file / is_dir
- hash de archivos (md5, sha1, sha256)
- tamaño humano (KB, MB, GB)
- tree recursivo
- head / tail / read_lines / count_lines
- grep por contenido
- replace_in_file
- concat_files / split_file
- diff_files
- sanitize_filename / safe_name
- read_json / write_json
- read_csv
- read_yaml (opcional)
- download_url
- image_resize / image_thumbnail / image_convert / image_exif
- find_duplicates
- backup (copia con timestamp)  ← clave para automejora
- ensure_inside (verifica rutas)
- touch (crea vacío)
- empty_dir / clear_dir

NUEVO (búsqueda web):
- web_search: motor DuckDuckGo HTML (async, sin API key).
- TOOL_SCHEMAS: esquemas compatibles con OpenRouter / OpenAI tools.
- dispatch_tool: ejecuta la tool que pida el LLM.

IMPORTANTE:
Este módulo permite operaciones reales sobre el sistema de archivos.
Para una IA, es recomendable ejecutar estas funciones con una carpeta de
trabajo permitida (workspace) y no con rutas arbitrarias del sistema.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import inspect
import io
import json
import mimetypes
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# IMPORTS OPCIONALES (no rompen si no están instalados)
# ---------------------------------------------------------------------------

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from pypdf import PdfReader  # type: ignore
    _HAS_PYPDF = True
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        _HAS_PYPDF = True
    except Exception:
        _HAS_PYPDF = False

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# EXTENSIONES CONOCIDAS
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".htm", ".css", ".scss", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".xml", ".csv", ".tsv", ".log", ".rst",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt",
    ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r",
    ".swift", ".dart", ".vue", ".svelte", ".gitignore",
    ".dockerfile", ".makefile",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}

PDF_EXTENSIONS = {".pdf"}


# ---------------------------------------------------------------------------
# EXCEPCIÓN PROPIA
# ---------------------------------------------------------------------------

class FileToolsError(Exception):
    """Error controlado de FileTools."""


class FileTools:
    """API sencilla para que una IA pueda trabajar con archivos."""

    def __init__(self, workspace: str | Path = "."):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # HELPERS INTERNOS
    # ======================================================================

    def _path(self, name: str | Path) -> Path:
        """Convierte una ruta relativa en una ruta dentro del workspace."""
        p = Path(name)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()

        try:
            p.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(
                f"Ruta fuera del workspace no permitida: {p}"
            ) from exc

        return p

    def ensure_inside(self, name: str | Path) -> str:
        """Lanza PermissionError si la ruta escapa del workspace."""
        p = self._path(name)
        return str(p.relative_to(self.workspace))

    # ======================================================================
    # CRUD BÁSICO
    # ======================================================================

    def create_file(
        self,
        name: str,
        content: str = "",
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> str:
        path = self._path(name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Ya existe: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return str(path)

    def write_file(
        self,
        name: str,
        content: str,
        encoding: str = "utf-8",
    ) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return str(path)

    def append_file(
        self,
        name: str,
        content: str,
        encoding: str = "utf-8",
    ) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as f:
            f.write(content)
        return str(path)

    def read_file(self, name: str, encoding: str = "utf-8") -> str:
        return self._path(name).read_text(encoding=encoding)

    def create_binary_file(self, name: str, data_base64: str) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data_base64))
        return str(path)

    def read_binary_file_base64(self, name: str) -> str:
        return base64.b64encode(self._path(name).read_bytes()).decode("ascii")

    def mkdir(self, name: str) -> str:
        path = self._path(name)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def touch(self, name: str) -> str:
        """Crea un archivo vacío si no existe."""
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return str(path)

    def list_files(self, directory: str = ".") -> list[dict[str, Any]]:
        path = self._path(directory)
        if not path.is_dir():
            raise NotADirectoryError(str(path))

        result = []
        for item in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            result.append({
                "name": item.name,
                "path": str(item.relative_to(self.workspace)),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return result

    def info(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.relative_to(self.workspace)),
            "absolute_path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
        }

    def copy(self, source: str, destination: str) -> str:
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return str(dst)

    def move(self, source: str, destination: str) -> str:
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return str(shutil.move(str(src), str(dst)))

    def delete(self, name: str, recursive: bool = False) -> bool:
        path = self._path(name)

        if path.is_dir():
            if not recursive:
                path.rmdir()
            else:
                shutil.rmtree(path)
        else:
            path.unlink()

        return True

    # ======================================================================
    # ZIP
    # ======================================================================

    def zip_create(
        self,
        output_zip: str,
        sources: list[str],
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> str:
        zip_path = self._path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for source in sources:
                src = self._path(source)

                if not src.exists():
                    raise FileNotFoundError(str(src))

                if src.is_dir():
                    for item in src.rglob("*"):
                        if item.is_file():
                            arcname = item.relative_to(self.workspace)
                            zf.write(item, arcname)
                else:
                    zf.write(src, src.relative_to(self.workspace))

        return str(zip_path)

    def zip_list(self, zip_name: str) -> list[dict[str, Any]]:
        zip_path = self._path(zip_name)

        with zipfile.ZipFile(zip_path, "r") as zf:
            return [
                {
                    "name": info.filename,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                }
                for info in zf.infolist()
            ]

    def zip_extract(
        self,
        zip_name: str,
        destination: str = ".",
    ) -> list[str]:
        zip_path = self._path(zip_name)
        dest = self._path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        extracted = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()

                try:
                    target.relative_to(dest)
                except ValueError as exc:
                    raise PermissionError(
                        f"ZIP rechazado por ruta insegura: {member.filename}"
                    ) from exc

                zf.extract(member, dest)
                extracted.append(str(target))

        return extracted

    def search(
        self,
        pattern: str,
        directory: str = ".",
    ) -> list[str]:
        root = self._path(directory)
        return [
            str(p.relative_to(self.workspace))
            for p in root.rglob(pattern)
        ]

    # ======================================================================
    # DETECCIÓN DE TIPO
    # ======================================================================

    def detect_kind(self, name: str) -> str:
        path = self._path(name)

        if path.is_dir():
            return "directory"

        ext = path.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            return "image"

        if ext in PDF_EXTENSIONS:
            return "pdf"

        if ext in TEXT_EXTENSIONS:
            return "text"

        mime, _ = mimetypes.guess_type(str(path))

        if mime:
            if mime.startswith("image/"):
                return "image"
            if mime == "application/pdf":
                return "pdf"
            if mime.startswith("text/"):
                return "text"
            if mime in ("application/json", "application/xml"):
                return "text"

        return "binary"

    # ======================================================================
    # IMÁGENES
    # ======================================================================

    def read_image_base64(self, name: str) -> dict[str, Any]:
        path = self._path(name)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        raw = path.read_bytes()

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"

        b64 = base64.b64encode(raw).decode("ascii")

        return {
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "mime": mime,
            "base64": b64,
            "data_url": f"data:{mime};base64,{b64}",
            "size": len(raw),
        }

    def image_info(self, name: str) -> dict[str, Any]:
        path = self._path(name)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        info: dict[str, Any] = {
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "size": path.stat().st_size,
        }

        if _HAS_PIL:
            try:
                with Image.open(path) as img:
                    info.update(
                        {
                            "width": img.width,
                            "height": img.height,
                            "format": img.format,
                            "mode": img.mode,
                        }
                    )
            except Exception as exc:
                info["pillow_error"] = str(exc)
        else:
            info["note"] = "Pillow no instalado, dimensiones no disponibles"

        return info

    def image_resize(
        self,
        name: str,
        output: str,
        width: int,
        height: int | None = None,
        keep_aspect: bool = True,
    ) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow no está instalado.")

        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as img:
            if keep_aspect:
                if height is None:
                    ratio = width / img.width
                    height = int(img.height * ratio)
                else:
                    img.thumbnail((width, height))
                    img.save(dst)
                    return str(dst)

            resized = img.resize((width, height or img.height))
            resized.save(dst)

        return str(dst)

    def image_thumbnail(
        self,
        name: str,
        output: str,
        size: int = 256,
    ) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow no está instalado.")

        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as img:
            img.thumbnail((size, size))
            img.save(dst)

        return str(dst)

    def image_convert(
        self,
        name: str,
        output: str,
        format: str | None = None,
    ) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow no está instalado.")

        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)

        fmt = (format or dst.suffix.lstrip(".")).upper() or "PNG"

        with Image.open(src) as img:
            if fmt == "JPG":
                fmt = "JPEG"
            img.save(dst, format=fmt)

        return str(dst)

    def image_exif(self, name: str) -> dict[str, Any]:
        if not _HAS_PIL:
            raise RuntimeError("Pillow no está instalado.")

        path = self._path(name)
        result: dict[str, Any] = {}

        with Image.open(path) as img:
            exif = img.getexif()

            for key, value in exif.items():
                try:
                    result[str(key)] = str(value)
                except Exception:
                    pass

        return result

    # ======================================================================
    # TEXTO
    # ======================================================================

    def read_text_auto(self, name: str) -> str:
        path = self._path(name)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue

        return path.read_text(encoding="utf-8", errors="replace")

    def head_file(self, name: str, lines: int = 20) -> str:
        path = self._path(name)
        out: list[str] = []

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    break
                out.append(line)

        return "".join(out)

    def tail_file(self, name: str, lines: int = 20) -> str:
        path = self._path(name)
        from collections import deque

        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))

    def read_lines(
        self,
        name: str,
        start: int = 1,
        end: int | None = None,
    ) -> str:
        path = self._path(name)
        out: list[str] = []

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if end is not None and i > end:
                    break
                out.append(line)

        return "".join(out)

    def count_lines(self, name: str) -> int:
        path = self._path(name)
        count = 0

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1

        return count

    def replace_in_file(
        self,
        name: str,
        old: str,
        new: str,
        count: int = -1,
    ) -> dict[str, Any]:
        path = self._path(name)
        content = self.read_text_auto(name)

        replaced = content.replace(old, new, count) if count >= 0 else content.replace(old, new)

        path.write_text(replaced, encoding="utf-8")

        return {
            "path": str(path.relative_to(self.workspace)),
            "replaced": content != replaced,
            "occurrences": content.count(old) if count < 0 else min(count, content.count(old)),
        }

    def grep(
        self,
        pattern: str,
        directory: str = ".",
        extensions: list[str] | None = None,
        max_results: int = 200,
    ) -> list[dict[str, Any]]:
        root = self._path(directory)
        results: list[dict[str, Any]] = []

        if not root.is_dir():
            raise NotADirectoryError(str(root))

        ext_filter = {e.lower() for e in extensions} if extensions else None

        for file in root.rglob("*"):
            if not file.is_file():
                continue

            if ext_filter and file.suffix.lower() not in ext_filter:
                continue

            try:
                with file.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, start=1):
                        if pattern in line:
                            results.append({
                                "file": str(file.relative_to(self.workspace)),
                                "line": i,
                                "text": line.rstrip("\n"),
                            })

                            if len(results) >= max_results:
                                return results
            except Exception:
                continue

        return results

    # ======================================================================
    # PDF
    # ======================================================================

    def read_pdf_text(self, name: str) -> str:
        if not _HAS_PYPDF:
            raise RuntimeError(
                "pypdf no está instalado. "
                "Añade 'pypdf' a requirements.txt"
            )

        path = self._path(name)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        reader = PdfReader(str(path))

        parts: list[str] = []

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                text = f"[error extrayendo página {i}: {exc}]"

            parts.append(f"--- Página {i + 1} ---\n{text}")

        return "\n\n".join(parts)

    # ======================================================================
    # LECTURA GENÉRICA
    # ======================================================================

    def read_any(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        kind = self.detect_kind(name)

        if kind == "directory":
            return {
                "kind": "directory",
                "path": str(path.relative_to(self.workspace)),
                "listing": self.list_files(name),
            }

        if kind == "image":
            data = self.read_image_base64(name)
            data["kind"] = "image"
            data["info"] = self.image_info(name)
            return data

        if kind == "pdf":
            try:
                return {
                    "kind": "pdf",
                    "path": str(path.relative_to(self.workspace)),
                    "filename": path.name,
                    "text": self.read_pdf_text(name),
                }
            except Exception as exc:
                return {
                    "kind": "pdf",
                    "path": str(path.relative_to(self.workspace)),
                    "filename": path.name,
                    "error": str(exc),
                }

        if kind == "text":
            return {
                "kind": "text",
                "path": str(path.relative_to(self.workspace)),
                "filename": path.name,
                "text": self.read_text_auto(name),
            }

        return {
            "kind": "binary",
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "size": path.stat().st_size,
            "note": "Archivo binario, no se puede mostrar como texto.",
        }

    # ======================================================================
    # UTILIDADES EXTRA
    # ======================================================================

    def exists(self, name: str) -> bool:
        try:
            return self._path(name).exists()
        except Exception:
            return False

    def is_file(self, name: str) -> bool:
        try:
            return self._path(name).is_file()
        except Exception:
            return False

    def is_dir(self, name: str) -> bool:
        try:
            return self._path(name).is_dir()
        except Exception:
            return False

    def file_hash(
        self,
        name: str,
        algorithm: str = "sha256",
    ) -> dict[str, Any]:
        path = self._path(name)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        h = hashlib.new(algorithm)

        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)

        return {
            "path": str(path.relative_to(self.workspace)),
            "algorithm": algorithm,
            "hash": h.hexdigest(),
        }

    @staticmethod
    def human_size(size: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        s = float(size)

        for u in units:
            if s < 1024 or u == units[-1]:
                return f"{s:.2f} {u}"
            s /= 1024

        return f"{size} B"

    def tree(
        self,
        directory: str = ".",
        max_depth: int = 3,
    ) -> dict[str, Any]:
        root = self._path(directory)

        if not root.is_dir():
            raise NotADirectoryError(str(root))

        def walk(p: Path, depth: int) -> dict[str, Any]:
            node: dict[str, Any] = {
                "name": p.name,
                "type": "directory",
                "children": [],
            }

            if depth >= max_depth:
                return node

            try:
                for item in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                    if item.is_dir():
                        node["children"].append(walk(item, depth + 1))
                    else:
                        node["children"].append({
                            "name": item.name,
                            "type": "file",
                            "size": item.stat().st_size,
                            "human_size": self.human_size(item.stat().st_size),
                        })
            except Exception as exc:
                node["error"] = str(exc)

            return node

        return walk(root, 0)

    def concat_files(
        self,
        output: str,
        sources: list[str],
        separator: str = "\n",
    ) -> str:
        parts: list[str] = []

        for s in sources:
            parts.append(self.read_text_auto(s))

        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(separator.join(parts), encoding="utf-8")

        return str(dst)

    def split_file(
        self,
        name: str,
        lines_per_chunk: int = 1000,
        output_dir: str = ".",
    ) -> list[str]:
        src = self._path(name)
        out_dir = self._path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if lines_per_chunk <= 0:
            raise ValueError("lines_per_chunk debe ser > 0")

        parts: list[str] = []
        buffer: list[str] = []
        index = 1

        with src.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buffer.append(line)
                if len(buffer) >= lines_per_chunk:
                    dst = out_dir / f"{src.stem}_part{index:03d}{src.suffix}"
                    dst.write_text("".join(buffer), encoding="utf-8")
                    parts.append(str(dst.relative_to(self.workspace)))
                    buffer = []
                    index += 1

            if buffer:
                dst = out_dir / f"{src.stem}_part{index:03d}{src.suffix}"
                dst.write_text("".join(buffer), encoding="utf-8")
                parts.append(str(dst.relative_to(self.workspace)))

        return parts

    def diff_files(self, a: str, b: str) -> str:
        import difflib

        text_a = self.read_text_auto(a).splitlines(keepends=True)
        text_b = self.read_text_auto(b).splitlines(keepends=True)

        return "".join(
            difflib.unified_diff(
                text_a,
                text_b,
                fromfile=a,
                tofile=b,
            )
        )

    @staticmethod
    def sanitize_filename(name: str) -> str:
        keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(c for c in name if c in keep).strip() or "file"

    def safe_name(self, name: str) -> str:
        return self.sanitize_filename(name)

    # ---------------- JSON ----------------

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_text_auto(name))

    def write_json(
        self,
        name: str,
        data: Any,
        indent: int = 2,
    ) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    # ---------------- CSV ----------------

    def read_csv(
        self,
        name: str,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        path = self._path(name)
        rows: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                rows.append(dict(row))

                if max_rows is not None and i + 1 >= max_rows:
                    break

        return rows

    # ---------------- YAML ----------------

    def read_yaml(self, name: str) -> Any:
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML no está instalado. Añade 'pyyaml' a requirements.txt"
            )

        return yaml.safe_load(self.read_text_auto(name))

    # ---------------- DESCARGA ----------------

    def download_url(
        self,
        url: str,
        destination: str,
        timeout: int = 60,
    ) -> dict[str, Any]:
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()

            dst.write_bytes(data)

            return {
                "path": str(dst.relative_to(self.workspace)),
                "size": len(data),
                "human_size": self.human_size(len(data)),
                "url": url,
            }

        except Exception as exc:
            raise FileToolsError(f"Error descargando {url}: {exc}") from exc

    # ---------------- DUPLICADOS ----------------

    def find_duplicates(
        self,
        directory: str = ".",
    ) -> list[list[str]]:
        root = self._path(directory)
        by_hash: dict[str, list[str]] = {}

        for file in root.rglob("*"):
            if not file.is_file():
                continue

            try:
                h = hashlib.sha256(file.read_bytes()).hexdigest()
                by_hash.setdefault(h, []).append(
                    str(file.relative_to(self.workspace))
                )
            except Exception:
                continue

        return [paths for paths in by_hash.values() if len(paths) > 1]

    # ---------------- BACKUP ----------------

    def backup(self, name: str) -> str:
        """
        Copia un archivo o carpeta con timestamp.
        Útil para la automejora del bot (guardar antes de tocar).
        """
        src = self._path(name)

        if not src.exists():
            raise FileNotFoundError(str(src))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if src.is_dir():
            dst = src.with_name(f"{src.name}.bak_{stamp}")
            shutil.copytree(src, dst)
        else:
            dst = src.with_name(f"{src.stem}.bak_{stamp}{src.suffix}")
            shutil.copy2(src, dst)

        return str(dst.relative_to(self.workspace))

    # ---------------- LIMPIEZA ----------------

    def empty_dir(self, name: str) -> str:
        path = self._path(name)

        if not path.is_dir():
            raise NotADirectoryError(str(path))

        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        return str(path)

    def clear_dir(self, name: str) -> str:
        return self.empty_dir(name)

    # ======================================================================
    # BÚSQUEDA WEB (DuckDuckGo HTML, sin API key)
    # ======================================================================

    async def web_search(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 12.0,
    ) -> dict[str, Any]:
        """Busca en DuckDuckGo HTML y devuelve resultados parseados."""

        if not query or not query.strip():
            return {"error": "query vacío", "results": []}

        max_results = max(1, min(int(max_results), 10))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://duckduckgo.com/",
        }

        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(
            headers=headers,
            timeout=client_timeout,
        ) as session:
            async with session.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "wt-wt"},
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                html = await resp.text()

        soup = BeautifulSoup(html, "lxml")

        def clean(u: str) -> str:
            if not u:
                return ""
            if u.startswith("//"):
                u = "https:" + u
            if "uddg=" in u:
                qs = parse_qs(urlparse(u).query)
                t = qs.get("uddg", [""])[0]
                if t:
                    return unquote(t)
            return u

        results: list[dict[str, str]] = []

        for block in soup.select("div.result, div.web-result"):
            title_el = block.select_one("a.result__a")
            if not title_el:
                continue

            title = title_el.get_text(" ", strip=True)
            url = clean(title_el.get("href", ""))
            snippet_el = block.select_one(".result__snippet")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

            if not url or not title:
                continue

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

            if len(results) >= max_results:
                break

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }


# ---------------------------------------------------------------------------
# API para una IA: funciones pequeñas y fáciles de convertir en tools.
# ---------------------------------------------------------------------------

TOOLS = FileTools(os.getenv("SUBTOM_WORKSPACE", "./workspace"))


def create_file(name: str, content: str = "", overwrite: bool = False) -> str:
    return TOOLS.create_file(name, content, overwrite=overwrite)


def read_file(name: str) -> str:
    return TOOLS.read_file(name)


def write_file(name: str, content: str) -> str:
    return TOOLS.write_file(name, content)


def append_file(name: str, content: str) -> str:
    return TOOLS.append_file(name, content)


def mkdir(name: str) -> str:
    return TOOLS.mkdir(name)


def touch(name: str) -> str:
    return TOOLS.touch(name)


def list_files(directory: str = ".") -> list[dict[str, Any]]:
    return TOOLS.list_files(directory)


def copy_file(source: str, destination: str) -> str:
    return TOOLS.copy(source, destination)


def move_file(source: str, destination: str) -> str:
    return TOOLS.move(source, destination)


def delete_file(name: str, recursive: bool = False) -> bool:
    return TOOLS.delete(name, recursive)


def create_zip(output_zip: str, sources: list[str]) -> str:
    return TOOLS.zip_create(output_zip, sources)


def list_zip(zip_name: str) -> list[dict[str, Any]]:
    return TOOLS.zip_list(zip_name)


def unzip_file(zip_name: str, destination: str = ".") -> list[str]:
    return TOOLS.zip_extract(zip_name, destination)


def file_info(name: str) -> dict[str, Any]:
    return TOOLS.info(name)


def search_files(pattern: str, directory: str = ".") -> list[str]:
    return TOOLS.search(pattern, directory)


# ---------------------------------------------------------------------------
# Funciones tool (análisis)
# ---------------------------------------------------------------------------

def detect_kind(name: str) -> str:
    return TOOLS.detect_kind(name)


def read_image_base64(name: str) -> dict[str, Any]:
    return TOOLS.read_image_base64(name)


def image_info(name: str) -> dict[str, Any]:
    return TOOLS.image_info(name)


def read_text_auto(name: str) -> str:
    return TOOLS.read_text_auto(name)


def read_pdf_text(name: str) -> str:
    return TOOLS.read_pdf_text(name)


def read_any(name: str) -> dict[str, Any]:
    return TOOLS.read_any(name)


# ---------------------------------------------------------------------------
# Funciones tool (robustez extra)
# ---------------------------------------------------------------------------

def exists(name: str) -> bool:
    return TOOLS.exists(name)


def is_file(name: str) -> bool:
    return TOOLS.is_file(name)


def is_dir(name: str) -> bool:
    return TOOLS.is_dir(name)


def file_hash(name: str, algorithm: str = "sha256") -> dict[str, Any]:
    return TOOLS.file_hash(name, algorithm)


def human_size(size: int) -> str:
    return TOOLS.human_size(size)


def tree(directory: str = ".", max_depth: int = 3) -> dict[str, Any]:
    return TOOLS.tree(directory, max_depth)


def head_file(name: str, lines: int = 20) -> str:
    return TOOLS.head_file(name, lines)


def tail_file(name: str, lines: int = 20) -> str:
    return TOOLS.tail_file(name, lines)


def read_lines(name: str, start: int = 1, end: int | None = None) -> str:
    return TOOLS.read_lines(name, start, end)


def count_lines(name: str) -> int:
    return TOOLS.count_lines(name)


def replace_in_file(name: str, old: str, new: str, count: int = -1) -> dict[str, Any]:
    return TOOLS.replace_in_file(name, old, new, count)


def grep(pattern: str, directory: str = ".", extensions: list[str] | None = None) -> list[dict[str, Any]]:
    return TOOLS.grep(pattern, directory, extensions)


def concat_files(output: str, sources: list[str], separator: str = "\n") -> str:
    return TOOLS.concat_files(output, sources, separator)


def split_file(name: str, lines_per_chunk: int = 1000, output_dir: str = ".") -> list[str]:
    return TOOLS.split_file(name, lines_per_chunk, output_dir)


def diff_files(a: str, b: str) -> str:
    return TOOLS.diff_files(a, b)


def sanitize_filename(name: str) -> str:
    return TOOLS.sanitize_filename(name)


def read_json(name: str) -> Any:
    return TOOLS.read_json(name)


def write_json(name: str, data: Any, indent: int = 2) -> str:
    return TOOLS.write_json(name, data, indent)


def read_csv(name: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    return TOOLS.read_csv(name, max_rows)


def read_yaml(name: str) -> Any:
    return TOOLS.read_yaml(name)


def download_url(url: str, destination: str, timeout: int = 60) -> dict[str, Any]:
    return TOOLS.download_url(url, destination, timeout)


def find_duplicates(directory: str = ".") -> list[list[str]]:
    return TOOLS.find_duplicates(directory)


def backup(name: str) -> str:
    return TOOLS.backup(name)


def empty_dir(name: str) -> str:
    return TOOLS.empty_dir(name)


def image_resize(name: str, output: str, width: int, height: int | None = None, keep_aspect: bool = True) -> str:
    return TOOLS.image_resize(name, output, width, height, keep_aspect)


def image_thumbnail(name: str, output: str, size: int = 256) -> str:
    return TOOLS.image_thumbnail(name, output, size)


def image_convert(name: str, output: str, format: str | None = None) -> str:
    return TOOLS.image_convert(name, output, format)


def image_exif(name: str) -> dict[str, Any]:
    return TOOLS.image_exif(name)


def ensure_inside(name: str) -> str:
    return TOOLS.ensure_inside(name)


# ---------------------------------------------------------------------------
# BÚSQUEDA WEB (módulo)
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    return await TOOLS.web_search(query, max_results)


# ---------------------------------------------------------------------------
# TOOL SCHEMAS (lo que ve el LLM en OpenRouter)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    # ------------------------- BÚSQUEDA WEB -------------------------
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Busca información actual en internet con DuckDuckGo. "
                "Úsala para noticias, datos recientes, documentación "
                "o verificar algo que no sabes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta en lenguaje natural.",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },

    # ------------------------- LECTURA -------------------------
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee un archivo de texto plano del workspace.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_any",
            "description": (
                "Lee cualquier archivo y devuelve algo útil según su tipo "
                "(texto, imagen base64, PDF extraído, listado si es carpeta)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lista el contenido de un directorio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tree",
            "description": "Árbol recursivo de un directorio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 6,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Busca un texto dentro de todos los archivos de un directorio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string", "default": "."},
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["pattern"],
            },
        },
    },

    # ------------------------- ESCRITURA -------------------------
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Crea un archivo nuevo con contenido.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string", "default": ""},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe (sobrescribe) un archivo de texto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Añade contenido al final de un archivo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Reemplaza texto dentro de un archivo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["name", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Crea un directorio.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Borra un archivo o carpeta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        },
    },

    # ------------------------- BACKUP (automejora) -------------------------
    {
        "type": "function",
        "function": {
            "name": "backup",
            "description": (
                "Copia un archivo/carpeta con timestamp. "
                "SIEMPRE úsala antes de modificar tu propio código."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# DISPATCHER — conecta nombre de tool → función real
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, Any]:
    return {
        # web
        "web_search": web_search,

        # lectura
        "read_file": read_file,
        "read_any": read_any,
        "list_files": list_files,
        "tree": tree,
        "grep": grep,

        # escritura
        "create_file": create_file,
        "write_file": write_file,
        "append_file": append_file,
        "replace_in_file": replace_in_file,
        "mkdir": mkdir,
        "delete_file": delete_file,

        # seguridad
        "backup": backup,
    }


_TOOL_REGISTRY = _build_registry()


async def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Ejecuta una tool y devuelve siempre un string (para el LLM)."""

    fn = _TOOL_REGISTRY.get(name)

    if fn is None:
        return f"Error: herramienta desconocida '{name}'"

    try:
        result = fn(**args)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        return f"Error en {name}: {type(exc).__name__}: {exc}"

    if isinstance(result, str):
        return result

    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# CLI opcional: también puedes usar el archivo desde la terminal.
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Herramientas de archivos para una IA."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create")
    p.add_argument("name")
    p.add_argument("--content", default="")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("read")
    p.add_argument("name")

    p = sub.add_parser("mkdir")
    p.add_argument("name")

    p = sub.add_parser("list")
    p.add_argument("directory", nargs="?", default=".")

    p = sub.add_parser("zip")
    p.add_argument("output_zip")
    p.add_argument("sources", nargs="+")

    p = sub.add_parser("unzip")
    p.add_argument("zip_name")
    p.add_argument("destination", nargs="?", default=".")

    p = sub.add_parser("zip-list")
    p.add_argument("zip_name")

    p = sub.add_parser("info")
    p.add_argument("name")

    p = sub.add_parser("kind")
    p.add_argument("name")

    p = sub.add_parser("image")
    p.add_argument("name")

    p = sub.add_parser("image-info")
    p.add_argument("name")

    p = sub.add_parser("pdf")
    p.add_argument("name")

    p = sub.add_parser("any")
    p.add_argument("name")

    p = sub.add_parser("exists")
    p.add_argument("name")

    p = sub.add_parser("hash")
    p.add_argument("name")
    p.add_argument("--algorithm", default="sha256")

    p = sub.add_parser("tree")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--depth", type=int, default=3)

    p = sub.add_parser("head")
    p.add_argument("name")
    p.add_argument("--lines", type=int, default=20)

    p = sub.add_parser("tail")
    p.add_argument("name")
    p.add_argument("--lines", type=int, default=20)

    p = sub.add_parser("grep")
    p.add_argument("pattern")
    p.add_argument("directory", nargs="?", default=".")

    p = sub.add_parser("backup")
    p.add_argument("name")

    p = sub.add_parser("duplicates")
    p.add_argument("directory", nargs="?", default=".")

    # NUEVO: búsqueda web desde CLI
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--max", type=int, default=5)

    args = parser.parse_args()

    try:
        if args.command == "create":
            result = create_file(args.name, args.content, args.overwrite)
        elif args.command == "read":
            result = read_file(args.name)
        elif args.command == "mkdir":
            result = mkdir(args.name)
        elif args.command == "list":
            result = json.dumps(list_files(args.directory), indent=2, ensure_ascii=False)
        elif args.command == "zip":
            result = create_zip(args.output_zip, args.sources)
        elif args.command == "unzip":
            result = json.dumps(unzip_file(args.zip_name, args.destination), indent=2, ensure_ascii=False)
        elif args.command == "zip-list":
            result = json.dumps(list_zip(args.zip_name), indent=2, ensure_ascii=False)
        elif args.command == "info":
            result = json.dumps(file_info(args.name), indent=2, ensure_ascii=False)
        elif args.command == "kind":
            result = detect_kind(args.name)
        elif args.command == "image":
            data = read_image_base64(args.name)
            data.pop("base64", None)
            data.pop("data_url", None)
            result = json.dumps(data, indent=2, ensure_ascii=False)
        elif args.command == "image-info":
            result = json.dumps(image_info(args.name), indent=2, ensure_ascii=False)
        elif args.command == "pdf":
            result = read_pdf_text(args.name)
        elif args.command == "any":
            data = read_any(args.name)
            data.pop("base64", None)
            data.pop("data_url", None)
            result = json.dumps(data, indent=2, ensure_ascii=False)
        elif args.command == "exists":
            result = str(exists(args.name))
        elif args.command == "hash":
            result = json.dumps(file_hash(args.name, args.algorithm), indent=2, ensure_ascii=False)
        elif args.command == "tree":
            result = json.dumps(tree(args.directory, args.depth), indent=2, ensure_ascii=False)
        elif args.command == "head":
            result = head_file(args.name, args.lines)
        elif args.command == "tail":
            result = tail_file(args.name, args.lines)
        elif args.command == "grep":
            result = json.dumps(grep(args.pattern, args.directory), indent=2, ensure_ascii=False)
        elif args.command == "backup":
            result = backup(args.name)
        elif args.command == "duplicates":
            result = json.dumps(find_duplicates(args.directory), indent=2, ensure_ascii=False)
        elif args.command == "search":
            data = asyncio.run(web_search(args.query, args.max))
            result = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Comando desconocido")

        print(result)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
