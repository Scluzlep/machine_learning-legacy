#!/usr/bin/env python3
"""Repair Webpack's non-native dynamic import in JupyterLite 0.1.1 Worker.

The old kernel source requests import(pyodideUrl) with webpackIgnore, but the
published Webpack worker bundle rewrites it to a synchronous require-context
stub (which always throws MODULE_NOT_FOUND). Restore the native import() in
the *generated* Worker chunk; never modify a Notebook or upstream wheel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

# Scope tightly to the exact Pyodide loader expression discovered in
# @jupyterlite/pyodide-kernel 0.1.1. Do not rewrite unrelated require calls.
_MISCOMPILED = re.compile(
    r'\(await __webpack_require__\(\d+\)\(([A-Za-z_$][\w$]*)\)\)\.loadPyodide'
)


def patch_source(source: str) -> tuple[str, int]:
    return _MISCOMPILED.subn(
        lambda match: f"(await import({match.group(1)})).loadPyodide",
        source,
    )


def patch(dist: Path) -> Path:
    chunks = dist / "extensions/@jupyterlite/pyodide-kernel-extension/static"
    if not chunks.is_dir():
        raise FileNotFoundError(f"Legacy kernel extension is missing: {chunks}")
    matches = []
    for path in sorted(chunks.glob("*.js")):
        original = path.read_text(encoding="utf-8")
        modified, count = patch_source(original)
        if count:
            matches.append((path, modified, count))
    total = sum(n for _, _, n in matches)
    if total != 1:
        raise RuntimeError(
            f"Expected exactly one legacy Pyodide Webpack require-context "
            f"placeholder; found {total} across {chunks}"
        )
    target, patched, _ = matches[0]
    target.write_text(patched, encoding="utf-8")
    assert "(await import(" in patched
    print(f"Restored native Pyodide ESM import in {target}")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dist", type=Path, default=Path("dist"))
    args = p.parse_args()
    patch(args.dist)
