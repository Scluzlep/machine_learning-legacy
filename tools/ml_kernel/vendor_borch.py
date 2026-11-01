#!/usr/bin/env python3
"""Vendor an exact pyborch wheel for the JupyterLite piplite index.

The 1.14.1 release is pure Python and bundles borch_webgpu/_borch.js, so
students do not fetch the runtime from a CDN when executing notebook cells.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

VERSION = "1.14.1"
WHEEL_NAME = f"pyborch-{VERSION}-py3-none-any.whl"
# PyPI's published digest for pyborch 1.14.1, 2026-09-24.
WHEEL_SHA256 = "80809bb34907f5d1a31e919c0affd228f16baa871592a24815d812298992c19c"


def verify(wheel: Path) -> None:
    if wheel.name != WHEEL_NAME:
        raise ValueError(f"Unexpected pyborch wheel: {wheel.name}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise ValueError(f"pyborch wheel SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt pyborch wheel")
        entries = set(archive.namelist())
        required = {
            "borch/__init__.py",
            "borch_webgpu/__init__.py",
            "borch_webgpu/_borch.js",
        }
        if not required.issubset(entries):
            raise ValueError(f"Missing bundled Borch browser runtime: {required - entries}")
        if archive.getinfo("borch_webgpu/_borch.js").file_size < 10000:
            raise ValueError("Borch WebGPU runtime appears incomplete")


def vendor(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    dest = output / WHEEL_NAME
    if not dest.is_file():
        subprocess.run(
            [sys.executable, "-m", "pip", "download",
             "--disable-pip-version-check", "--no-deps", "--only-binary=:all:",
             "--dest", str(output), f"pyborch=={VERSION}"],
            check=True,
        )
    verify(dest)
    print("Pinned pyborch wheel verified:", dest, dest.stat().st_size, "bytes")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".ml-wheel"))
    args = parser.parse_args()
    vendor(args.output)
