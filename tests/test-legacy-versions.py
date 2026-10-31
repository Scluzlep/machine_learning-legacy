#!/usr/bin/env python3
"""Check legacy compatibility versions, adapter syntax and optional dist assets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {
    "numpy": "1.24.2",
    "scipy": "1.9.3",
    "scikit-learn": "1.2.2",
    "pandas": "1.5.3",
    "matplotlib": "3.5.2",
    "Pillow": "9.1.1",
    "h5py": "3.7.0",
    "sqlite3": "1.0.0",
}


def verify(dist: Path | None = None) -> None:
    scripts = sorted((ROOT / "tools/ml_kernel").glob("*.py"))
    scripts += sorted((ROOT / "machine_learning/browser_ml").glob("*.py"))
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    requirements = (ROOT / "site-requirements.txt").read_text(encoding="utf-8")
    for package in ("jupyterlite-core[lab,translation]==0.1.3",
                    "jupyterlite-pyodide-kernel==0.1.1",
                    "ipywidgets==7.6.5", "ipympl==0.9.3"):
        assert package in requirements, f"Missing pinned legacy package: {package}"
    vendor = (ROOT / "tools/ml_kernel/vendor_pyodide.py").read_text(encoding="utf-8")
    assert 'PYODIDE_VERSION = "0.23.4"' in vendor
    assert "repodata.json" in vendor and "pyodide.mjs" in vendor
    assert (ROOT / "tools/ml_kernel/legacy_ui.py").is_file(), "Missing UI compatibility shim"

    if dist is not None:
        piplite_index = dist / "pypi/all.json"
        assert piplite_index.is_file(), "Piplite addon missing: no wheel index"
        piplite = json.loads(piplite_index.read_text(encoding="utf-8"))
        assert "pyodide-kernel" in piplite, "Patched legacy kernel is absent from wheel index"
        for name, version in (
            ("ipython", "8.12.3"),
            ("widgetsnbextension", "3.5.2"),
            ("jupyterlab-widgets", "1.1.11"),
            ("ipython-genutils", "0.2.0"),
            ("ipywidgets", "7.6.5"),
            ("ipympl", "0.9.3"),
        ):
            assert name in piplite and version in piplite[name]["releases"], (
                f"Legacy Piplite index missing {name}=={version}"
            )
        assert (dist / "pypi/pyodide_kernel-0.1.1-py3-none-any.whl").is_file()
        site = (dist / "jupyter-lite.json")
        assert site.is_file(), site
        config = json.loads(site.read_text(encoding="utf-8"))
        kernel = config["jupyter-config-data"]["litePluginSettings"][
            "@jupyterlite/pyodide-kernel-extension:kernel"]
        assert kernel["pyodideUrl"].endswith("/pyodide/pyodide.mjs")
        assert any("pypi/all.json" in url for url in kernel.get("pipliteUrls", [])), (
            "JupyterLite did not expose the local Pyodide kernel wheel index"
        )
        assert not kernel.get("loadPyodideOptions"), "Modern settings in legacy kernel"
        pyodide_dir = dist / "pyodide"
        for name in ("pyodide.mjs", "pyodide.asm.wasm", "repodata.json"):
            assert (pyodide_dir / name).is_file(), name
        packages = json.loads((pyodide_dir / "repodata.json").read_text())["packages"]

        # All dependencies of course-shipped native WASM roots must stay
        # same-origin; optional packages alone may use Fastly on demand.
        roots = ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
                 "micropip", "h5py", "pillow", "sqlite3")
        normalized = {k.lower().replace("_", "-"): k for k in packages}
        pending = list(roots)
        bundled = set()
        while pending:
            requirement = pending.pop()
            key = normalized.get(requirement.lower().replace("_", "-"))
            assert key is not None, f"Missing required Pyodide package: {requirement}"
            if key in bundled:
                continue
            bundled.add(key)
            item = packages[key]
            filename = item.get("file_name", "")
            assert filename and "://" not in filename, (
                f"Required package {key} would hit CDN: {filename}"
            )
            assert (pyodide_dir / filename).is_file(), (
                f"Missing local package {key}: {filename}"
            )
            pending.extend(item.get("depends") or [])
        for name, item in packages.items():
            filename = item.get("file_name", "")
            if "://" in filename:
                assert filename.startswith(
                    "https://fastly.jsdelivr.net/pyodide/v0.23.4/full/"
                ), f"Unexpected external package source {name}: {filename}"
        for name in ("tf.min.js", "tf-backend-webgpu.min.js",
                     "course-losses.js"):
            assert (dist / "vendor" / name).is_file(), (
                f"Missing self-hosted TF.js resource: {name}"
            )
        assert (dist / "assets/fonts/SourceHanSansSC-Regular.otf").is_file()
        print(f"Bundled Pyodide native dependency closure: {len(bundled)} packages")
        for name, version in VERSIONS.items():
            key = next((k for k in packages if k.lower() == name.lower()), None)
            assert key, f"Missing official WASM package: {name}"
            item = packages[key]
            assert item["version"] == version, (
                f"{name}: expected {version}, repodata has {item['version']}"
            )
            filename = item.get("file_name")
            assert filename and "://" not in filename, f"{name} is not self-hosted"
            assert (pyodide_dir / filename).is_file(), f"Missing wheel {filename}"
    print(f"Legacy runtime dependency/syntax checks PASS ({len(scripts)} modules)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path)
    verify(parser.parse_args().dist)
