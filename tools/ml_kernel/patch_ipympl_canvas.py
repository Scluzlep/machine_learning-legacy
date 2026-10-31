#!/usr/bin/env python3
"""Fix an ipympl 0.9.3 frontend race before an HTML image is decoded.

jupyter-matplotlib 0.11.3 sometimes calls offscreen_context.drawImage(image)
from change:_size or handle_resize immediately after the canvas model is
initialized. The Image object's initial 'src' has not finished loading then,
which throws InvalidStateError and makes JupyterLab log "Error setting state".

The normal image.onload handler still performs the actual draw. Guard only
these premature resize draws; do not hide errors or intercept the Canvas API.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OLD = "this.resize_canvas(),this.offscreen_context.drawImage(this.image,0,0)"
NEW = ("this.resize_canvas(),this.image.complete&&this.image.naturalWidth>0"
       "&&this.offscreen_context.drawImage(this.image,0,0)")


def patch_source(source: str) -> tuple[str, int]:
    if OLD not in source:
        return source, 0
    count = source.count(OLD)
    if count != 2:
        raise RuntimeError(f"Unexpected jupyter-matplotlib resize calls: {count}")
    result = source.replace(OLD, NEW)
    if result.count(NEW) != 2:
        raise AssertionError("Incomplete guarded image draw patch")
    return result, count


def patch_dist(dist: Path) -> tuple[int, int]:
    static = dist / "extensions" / "jupyter-matplotlib" / "static"
    if not static.is_dir():
        raise FileNotFoundError(f"ipympl JupyterLab frontend was not bundled: {static}")
    matches = 0
    patches = 0
    for path in sorted(static.glob("*.js")):
        source = path.read_text(encoding="utf-8")
        if OLD not in source:
            if source.count(NEW) == 2:
                matches += 1
            continue
        updated, count = patch_source(source)
        path.write_text(updated, encoding="utf-8")
        matches += 1
        patches += count
    if matches == 0:
        raise RuntimeError("No recognizable ipympl frontend chunk found; inspect version")
    print(f"ipympl image decode guards: {matches} frontend chunk(s), {patches} calls patched")
    return matches, patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    patch_dist(args.dist)
