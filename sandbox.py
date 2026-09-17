from __future__ import annotations

import asyncio
import ast
import py_compile
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

# Clase Sandbox con un nuevo método para verificar cambios en archivos Python
class Sandbox:
    def __init__(
        self,
        timeout: float = 15.0,
        max_output: int = 100000,
    ):
        self.timeout = timeout
        self.max_output = max_output
        self._sem = asyncio.Semaphore(4)

    async def verify_python_change(self, file_path: str, old_content: str, new_content: str) -> dict[str, Any]:
        """
        Verifica cambios en un archivo Python para validar sintaxis, clases, funciones, imports eliminados, etc.
        """
        try:
            # Verificar sintaxis usando ast.parse
            ast.parse(new_content)
        except SyntaxError as e:
            return {"error": f"Error de sintaxis: {e}"}

        # Crear archivos temporales para la comparación
        temp_dir = tempfile.mkdtemp()
        old_file = os.path.join(temp_dir, "old.py")
        new_file = os.path.join(temp_dir, "new.py")

        try:
            with open(old_file, "w", encoding="utf-8") as f:
                f.write(old_content)

            with open(new_file, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Compilar ambos archivos
            py_compile.compile(new_file, doraise=True)

            # Comparar líneas
            old_lines = old_content.splitlines()
            new_lines = new_content.splitlines()
            similarity = len(set(old_lines) & set(new_lines)) / max(len(old_lines), len(new_lines))

            if similarity < 0.95:
                return {"error": "La similitud entre los archivos es menor al 95%."}

            # Validar si se eliminaron clases, funciones o imports
            old_tree = ast.parse(old_content)
            new_tree = ast.parse(new_content)

            old_classes = {node.name for node in ast.walk(old_tree) if isinstance(node, ast.ClassDef)}
            new_classes = {node.name for node in ast.walk(new_tree) if isinstance(node, ast.ClassDef)}

            old_functions = {node.name for node in ast.walk(old_tree) if isinstance(node, ast.FunctionDef)}
            new_functions = {node.name for node in ast.walk(new_tree) if isinstance(node, ast.FunctionDef)}

            old_imports = {node.module for node in ast.walk(old_tree) if isinstance(node, ast.Import)}
            new_imports = {node.module for node in ast.walk(new_tree) if isinstance(node, ast.Import)}

            removed_classes = old_classes - new_classes
            removed_functions = old_functions - new_functions
            removed_imports = old_imports - new_imports

            warnings = []

            if removed_classes:
                warnings.append(f"Clases eliminadas: {removed_classes}")
            if removed_functions:
                warnings.append(f"Funciones eliminadas: {removed_functions}")
            if removed_imports:
                warnings.append(f"Imports eliminados: {removed_imports}")

            return {"ok": True, "warnings": warnings}

        except Exception as e:
            return {"error": str(e)}

        finally:
            shutil.rmtree(temp_dir)

    # Otros métodos existentes en la clase permanecen sin cambios

# Fin del archivo