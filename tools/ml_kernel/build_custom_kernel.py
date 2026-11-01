#!/usr/bin/env python3
"""Create a patched copy of the upstream pyodide-kernel wheel.

The original Notebook files are never modified; the kernel runs an opt-in
website preprocessor before executing each cell.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile


def transform_kernel(src: bytes) -> bytes:
    source = src.decode("utf-8")
    marker = "    async def run(self, code):\n"
    if source.count(marker) != 1:
        raise RuntimeError("Unsupported upstream pyodide_kernel/kernel.py version")
    replacement = marker + (
        "        # Browser ML: adapt execution without editing user cells.\n"
        "        from .kernel_runtime import prepare_cell as _prepare_ml_cell\n"
        "        try:\n"
        "            code = await _prepare_ml_cell(code)\n"
        "        except Exception as _legacy_error:\n"
        "            import traceback as _tb\n"
        "            return dict(status='error', ename=type(_legacy_error).__name__,\n"
        "                        evalue=str(_legacy_error), traceback=_tb.format_exception(\n"
        "                            type(_legacy_error), _legacy_error,\n"
        "                            _legacy_error.__traceback__))\n"
    )
    return source.replace(marker, replacement, 1).encode("utf-8")


def build(wheel: Path, root: Path, destination: Path) -> Path:
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    kernel_path = "pyodide_kernel/kernel.py"
    if kernel_path not in files:
        raise RuntimeError("Not a pyodide-kernel wheel")
    files[kernel_path] = transform_kernel(files[kernel_path])
    files["pyodide_kernel/kernel_runtime.py"] = (root / "tools/ml_kernel/kernel_runtime.py").read_bytes()
    files["pyodide_kernel/imports_bootstrap.py"] = (root / "tools/ml_kernel/imports_bootstrap.py").read_bytes()
    files["pyodide_kernel/borch_compat.py"] = (root / "tools/ml_kernel/borch_compat.py").read_bytes()
    files["pyodide_kernel/chinese_font.py"] = (root / "tools/ml_kernel/chinese_font.py").read_bytes()
    files["pyodide_kernel/legacy_scikit.py"] = (root / "tools/ml_kernel/legacy_scikit.py").read_bytes()
    files["pyodide_kernel/legacy_ui.py"] = (root / "tools/ml_kernel/legacy_ui.py").read_bytes()
    for filename in ("legacy_numpy.py", "legacy_scipy.py"):
        files["pyodide_kernel/" + filename] = (
            root / "tools/ml_kernel" / filename
        ).read_bytes()
    files["pyodide_kernel/legacy_data/boston_house_prices.csv"] = (
        root / "tools/ml_kernel/legacy_data/boston_house_prices.csv"
    ).read_bytes()
    files["pyodide_kernel/auto_tfjs/__init__.py"] = b""
    for name in ("tfjs_bridge.py", "tfjs_compat.py", "tfjs_tensor.py", "tfjs_layers.py"):
        files["pyodide_kernel/auto_tfjs/" + name] = (
            root / "machine_learning/browser_ml" / name
        ).read_bytes()
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise RuntimeError("Missing wheel RECORD")
    record = records[0]
    rows = []
    for name, data in sorted(files.items()):
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        rows.append([name, "sha256=" + digest, len(data)])
    rows.append([record, "", ""])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    files[record] = stream.getvalue().encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / wheel.name
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print("Custom JupyterLite kernel:", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(".ml-wheel"))
    args = parser.parse_args()
    build(args.wheel, args.root, args.output)
