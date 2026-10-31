#!/usr/bin/env python3
"""Serve Pyodide 0.23.4 and its own WASM wheels from JupyterLite's origin.

Older Pyodide uses pyodide.js and repodata.json, not the modern Pyodide 314
pyodide.mjs/pyodide-lock.json. Mixed binary wheels cannot load.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

PYODIDE_VERSION = "0.23.4"
FALLBACK = f"https://fastly.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
CORE = ("pyodide.js", "pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm",
        "python_stdlib.zip", "repodata.json")
COURSE_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib",
                   "scikit-learn", "micropip")
ON_DEMAND_COURSE_PACKAGES = ("h5py", "Pillow", "sqlite3")


def _key(packages: dict, name: str) -> str:
    canonical = name.lower().replace("_", "-").replace(".", "-")
    for key in packages:
        if key.lower().replace("_", "-").replace(".", "-") == canonical:
            return key
    raise KeyError(f"Pyodide {PYODIDE_VERSION} lacks package {name}")


def _wheel_closure(packages: dict, roots: tuple[str, ...]) -> set[str]:
    chosen = set()
    todo = list(roots)
    while todo:
        name = _key(packages, todo.pop())
        if name in chosen:
            continue
        chosen.add(name)
        todo.extend(packages[name].get("depends") or [])
    return chosen


def _download(url: str, target: Path, digest: str | None = None) -> int:
    if target.is_file() and (not digest or
                            hashlib.sha256(target.read_bytes()).hexdigest() == digest):
        return target.stat().st_size
    for attempt in range(3):
        try:
            with urlopen(url, timeout=90) as response:
                payload = response.read()
            if digest and hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"SHA256 mismatch: {url}")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return len(payload)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _vendor_scientific_wheels(dest: Path, repodata: Path) -> tuple[int, int]:
    manifest = json.loads(repodata.read_text(encoding="utf-8"))
    packages = manifest["packages"]
    roots = _wheel_closure(packages, COURSE_PACKAGES + ON_DEMAND_COURSE_PACKAGES)
    wanted = []
    for name in sorted(roots):
        metadata = packages[name]
        filename = metadata.get("file_name")
        if not filename:
            continue
        parsed = urlparse(filename)
        if parsed.scheme not in ("", "https"):
            raise ValueError(f"Unsupported wheel URL: {filename}")
        if not parsed.scheme and (Path(filename).name != filename or filename.startswith(".")):
            raise ValueError(f"Unsafe wheel filename: {filename}")
        url = filename if parsed.scheme else urljoin(FALLBACK, filename)
        wanted.append((url, dest / Path(parsed.path).name, metadata.get("sha256")))
    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(lambda entry: _download(*entry), wanted))
    # Keep other official packages accessible from their matching-version CDN.
    for name, metadata in packages.items():
        filename = metadata.get("file_name")
        if name not in roots and filename and not urlparse(filename).scheme:
            metadata["file_name"] = urljoin(FALLBACK, filename)
    repodata.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    return len(wanted), sum(sizes)


def vendor(dist: Path, npm_package: Path, base_path: str) -> None:
    dest = dist / "pyodide"
    dest.mkdir(parents=True, exist_ok=True)
    for filename in CORE:
        source = npm_package / filename
        target = dest / filename
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            _download(urljoin(FALLBACK, filename), target)
    count, size = _vendor_scientific_wheels(dest, dest / "repodata.json")
    config_file = dist / "jupyter-lite.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    settings = config.setdefault("jupyter-config-data", {}).setdefault("litePluginSettings", {})
    kernel = settings.setdefault("@jupyterlite/pyodide-kernel-extension:kernel", {})
    kernel["pyodideUrl"] = f"{base_path.rstrip('/')}/pyodide/pyodide.mjs"
    # Older workers have no loadPyodideOptions support. Package preloading
    # takes place before the first cell in kernel_runtime.py.
    kernel.pop("loadPyodideOptions", None)
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Local Pyodide {PYODIDE_VERSION}: {kernel['pyodideUrl']}")
    print(f"Scientific wheels: {count} files, {size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--npm-package", type=Path, default=Path(".pyodide-vendor/node_modules/pyodide"))
    parser.add_argument("--base-path", default="/machine_learning")
    args = parser.parse_args()
    vendor(args.dist, args.npm_package, args.base_path)
