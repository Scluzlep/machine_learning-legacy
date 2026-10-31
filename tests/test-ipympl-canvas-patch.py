#!/usr/bin/env python3
"""Pure Python regression: preserve onload drawing and guard only resize calls."""
import importlib.util
from pathlib import Path
import tempfile

root = Path(__file__).resolve().parents[1]
filename = root / "tools/ml_kernel/patch_ipympl_canvas.py"
spec = importlib.util.spec_from_file_location("ipympl_canvas_patch_test", filename)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)

fixture = ("this.on('change:_size',()=>{" + shim.OLD + "});"
           + "handle_resize(e){" + shim.OLD + ";}"
           + "this.image.onload=()=>{this.offscreen_context.drawImage(this.image,0,0)};")
corrected, count = shim.patch_source(fixture)
assert count == 2
assert corrected.count(shim.NEW) == 2
assert "this.image.onload=()=>{this.offscreen_context.drawImage(this.image,0,0)}" in corrected
assert corrected.count(shim.OLD) == 0
assert shim.patch_source(corrected) == (corrected, 0)
with tempfile.TemporaryDirectory() as temp:
    directory = Path(temp) / "dist/extensions/jupyter-matplotlib/static"
    directory.mkdir(parents=True)
    (directory / "316.js").write_text(fixture, encoding="utf-8")
    assert shim.patch_dist(Path(temp) / "dist") == (1, 2)
    assert shim.patch_dist(Path(temp) / "dist") == (1, 0)
print("ipympl premature image draw guards: PASS")
