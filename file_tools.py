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

IMPORTANTE:
Este módulo permite operaciones reales sobre el sistema de archivos.
Para una IA, es recomendable ejecutar estas funciones con una carpeta de
trabajo permitida (workspace) y no con rutas arbitrarias del sistema.
"""

from __future__ import annotations

import base64
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


class FileTools:
    """API sencilla para que una IA pueda trabajar con archivos."""

    def __init__(self, workspace: str | Path = "."):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str | Path) -> Path:
        """Convierte una ruta relativa en una ruta dentro del workspace."""
        p = Path(name)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()

        # Evita escapar del workspace accidentalmente.
        try:
            p.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(
                f"Ruta fuera del workspace no permitida: {p}"
            ) from exc

        return p

    def create_file(
        self,
        name: str,
        content: str = "",
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> str:
        """Crea un archivo con cualquier nombre/extensión."""
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
        """Crea o reemplaza un archivo de texto."""
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
        """Añade texto al final de un archivo."""
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as f:
            f.write(content)
        return str(path)

    def read_file(self, name: str, encoding: str = "utf-8") -> str:
        """Lee un archivo de texto."""
        return self._path(name).read_text(encoding=encoding)

    def create_binary_file(self, name: str, data_base64: str) -> str:
        """Crea un archivo binario a partir de Base64."""
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data_base64))
        return str(path)

    def read_binary_file_base64(self, name: str) -> str:
        """Lee un archivo binario y lo devuelve como Base64."""
        return base64.b64encode(self._path(name).read_bytes()).decode("ascii")

    def mkdir(self, name: str) -> str:
        """Crea un directorio, incluidos sus padres."""
        path = self._path(name)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def list_files(self, directory: str = ".") -> list[dict[str, Any]]:
        """Lista archivos y carpetas de un directorio."""
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
        """Obtiene información de un archivo o directorio."""
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
        """Copia un archivo o directorio."""
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return str(dst)

    def move(self, source: str, destination: str) -> str:
        """Mueve o renombra un archivo/directorio."""
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return str(shutil.move(str(src), str(dst)))

    def delete(self, name: str, recursive: bool = False) -> bool:
        """Elimina un archivo o directorio."""
        path = self._path(name)

        if path.is_dir():
            if not recursive:
                path.rmdir()
            else:
                shutil.rmtree(path)
        else:
            path.unlink()

        return True

    def zip_create(
        self,
        output_zip: str,
        sources: list[str],
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> str:
        """Crea un ZIP con archivos/directorios."""
        zip_path = self._path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for source in sources:
                src = self._path(source)

                if not src.exists():
                    raise FileNotFoundError(str(src))

                if src.is_dir():
                    # Incluye la carpeta y todos sus archivos.
                    for item in src.rglob("*"):
                        if item.is_file():
                            arcname = item.relative_to(self.workspace)
                            zf.write(item, arcname)
                else:
                    zf.write(src, src.relative_to(self.workspace))

        return str(zip_path)

    def zip_list(self, zip_name: str) -> list[dict[str, Any]]:
        """Lista el contenido de un ZIP."""
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
        """
        Extrae un ZIP de forma segura, evitando rutas como ../../archivo.
        """
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
        """Busca archivos/carpetas usando patrones de pathlib."""
        root = self._path(directory)
        return [
            str(p.relative_to(self.workspace))
            for p in root.rglob(pattern)
        ]


# ---------------------------------------------------------------------------
# API para una IA: funciones pequeñas y fáciles de convertir en tools.
# ---------------------------------------------------------------------------

TOOLS = FileTools("./workspace")


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
        else:
            raise ValueError("Comando desconocido")

        print(result)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
