#!/usr/bin/env python3
"""Prepare *deployment-only* JupyterLite copies; never edit course sources.

Map classic ipykernel metadata to the browser kernel and resolve legacy
relative markdown images against the static GitHub Pages /files tree.
Code cells, outputs and lesson contents remain untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

_IMAGE = re.compile(
    r'(?P<start>\]\(\s*|\bsrc\s*=\s*["\'])'
    r'(?P<target>(?:\./)?(?:images|data)/[^)"\']+)'
)


def stage(source: Path, destination: Path, base: str) -> tuple[int, int]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    kernels, images = 0, 0
    for notebook in destination.rglob("*.ipynb"):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        spec = payload.get("metadata", {}).get("kernelspec", {})
        changed = False
        if spec.get("name") == "python3":
            spec["name"] = "python"
            spec["display_name"] = "Python (Pyodide)"
            changed = True
            kernels += 1
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            parts = cell.get("source", [])
            original = "".join(parts) if isinstance(parts, list) else parts

            def resolve(match):
                nonlocal images
                # HTML image names can contain spaces (for example figure 1.png).
                # Preserve whitespace before an optional Markdown image title.
                target = match.group("target").rstrip()
                asset = notebook.parent / target
                if not asset.is_file():
                    return match.group(0)
                rel = asset.relative_to(destination)
                images += 1
                return (match.group("start") + base + "/files/"
                        + quote(rel.as_posix(), safe="/")
                        + match.group("target")[len(target):])

            converted = _IMAGE.sub(resolve, original)
            if converted != original:
                cell["source"] = converted.splitlines(keepends=True) if isinstance(parts, list) else converted
                changed = True
        if changed:
            notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Staged {kernels} notebook kernel specs and {images} static image URLs")
    return kernels, images


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="machine_learning")
    ap.add_argument("--output", default=".ml-contents")
    ap.add_argument("--site-base", default="/" + os.environ.get("GITHUB_REPOSITORY", "Scluzlep/machine_learning").split("/")[-1])
    args = ap.parse_args()
    n, images = stage(Path(args.source), Path(args.output), args.site_base.rstrip("/"))
    if n < 30:
        raise SystemExit(f"Unexpectedly few legacy kernel specs rewritten: {n}")
