#!/usr/bin/env python3
"""Fast checks for the patched Pyodide kernel and unmodified course materials."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
source = root / "tools/ml_kernel/kernel_runtime.py"
spec = importlib.util.spec_from_file_location("ml_kernel_runtime_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model_code = "model = Sequential([Dense(3, activation='sigmoid')])"
assert module.transform_tf_fit(model_code) == model_code
# The 44-notebook assignment also trains models stored in a list.
collection_code = (
    "models = [None] * 3\n"
    "for i in range(3):\n"
    "    models[i] = Sequential([Dense(4)])\n"
    "    models[i].fit(X, y, epochs=10)\n"
)
collection_transformed = module.transform_tf_fit(collection_code)
assert "await models[i].fit(" in collection_transformed
compile(collection_transformed, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
assert module.uses_legacy_scipy("img = scipy.misc.imresize(source, (64, 64))")
assert module.uses_legacy_scipy("image = ndimage.imread(filename)")
assert module.uses_legacy_scipy("image = scipy.ndimage.imread(filename)")
assert not module.uses_legacy_scipy("from scipy import ndimage\nfiltered = ndimage.gaussian_filter(X, 1)")
normal = "reg.fit(X, y)\nscaler.fit(X)"
assert module.transform_tf_fit(normal) == normal
transformed = module.transform_tf_fit("model.fit(Xt, Yt, epochs=10)")
assert "await model.fit(" in transformed
compile(transformed, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

func_code = "def train():\n    model.fit(Xt, Yt)"
assert module.transform_tf_fit(func_code) == func_code

original = root / "machine_learning/4. Neural Network_part_1/2. CoffeeRoasting_TF.ipynb"
data = original.read_bytes()
object_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
assert object_id == "c5fbceb7612f7aef47e23cff40d137916823156e", object_id

wheel = root / ".ml-wheel/pyodide_kernel-0.1.1-py3-none-any.whl"
with zipfile.ZipFile(wheel) as archive:
    assert archive.testzip() is None
    kernel = archive.read("pyodide_kernel/kernel.py").decode()
    assert "code = await _prepare_ml_cell(code)" in kernel
    assert "pyodide_kernel/auto_tfjs/tfjs_bridge.py" in archive.namelist()
    assert "pyodide_kernel/kernel_runtime.py" in archive.namelist()
    assert "pyodide_kernel/borch_compat.py" in archive.namelist()
    assert "pyodide_kernel/legacy_scikit.py" in archive.namelist()
    assert "pyodide_kernel/legacy_numpy.py" in archive.namelist()
    assert "pyodide_kernel/legacy_scipy.py" in archive.namelist()
    assert "except Exception as _legacy_error" in kernel
    assert "pyodide_kernel/legacy_data/boston_house_prices.csv" in archive.namelist()
    assert archive.read("pyodide_kernel/legacy_data/boston_house_prices.csv").startswith(b"506,13")

print("Kernel AST transform, wheel integrity and original notebook blob: PASS")
