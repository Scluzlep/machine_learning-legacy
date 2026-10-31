#!/usr/bin/env python3
"""No-network regression for the JupyterLite 0.1.1 bundled-worker patch."""
import importlib.util
from pathlib import Path
import tempfile

root = Path(__file__).resolve().parents[1]
name = root / "tools/ml_kernel/patch_legacy_worker.py"
spec = importlib.util.spec_from_file_location("worker_patch_test", name)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sample = ('async initRuntime(t){let e,{pyodideUrl:i,indexUrl:n}=t;'
          'i.endsWith(".mjs")?e=(await __webpack_require__(693)(i)).loadPyodide'
          ':(importScripts(i),e=self.loadPyodide),this._pyodide=await e({indexURL:n})}')
updated, count = module.patch_source(sample)
assert count == 1
assert "(await import(i)).loadPyodide" in updated
assert "__webpack_require__(693)(i)" not in updated
assert module.patch_source(updated) == (updated, 0)
with tempfile.TemporaryDirectory() as temp:
    dist = Path(temp)
    chunk = dist / "extensions/@jupyterlite/pyodide-kernel-extension/static/951.test.js"
    chunk.parent.mkdir(parents=True)
    chunk.write_text(sample, encoding="utf-8")
    assert module.patch(dist) == chunk
    assert "await import(i)" in chunk.read_text(encoding="utf-8")
    try:
        module.patch(dist)
    except RuntimeError as err:
        assert "found 0" in str(err)
    else:
        raise AssertionError("Must reject patches that did not match")
print("Legacy Webpack Worker ESM import regression: PASS")
