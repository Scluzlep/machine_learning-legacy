#!/usr/bin/env python3
"""Contract tests: standard torch import paths resolve to Borch objects.

Pure-CPython fakes intentionally test import semantics without claiming GPU
tensor numerical parity. A separate browser test must prove JSPI/WebGPU.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tested_borch_compat", ROOT / "tools/ml_kernel/borch_compat.py"
)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

for source in (
    "import torch",
    "import torch.nn as nn",
    "import torch.optim",
    "from torch import Tensor",
    "from torch.nn.functional import relu",
    "from torch.utils.data import DataLoader",
    "from torch.optim.lr_scheduler import StepLR",
    "%matplotlib widget\nimport torch",
):
    assert bridge.needs_torch(source), source

for source in (
    "import tensorflow as tf",
    "import torchvision",
    "print('import torch')",
    '# import torch',
    '"""\nimport torch\n"""',
    "torch.tensor([1])",  # no import; do not override other runtimes
):
    assert not bridge.needs_torch(source), source


def fake_backend():
    backend = types.ModuleType("fake_borch_webgpu")
    backend.Tensor = type("FakeTensor", (), {})
    backend.tensor = lambda values: ("fake-tensor", list(values))
    nn = types.ModuleType("fake_nn")
    nn.Module = type("Module", (), {})
    nn.Linear = type("Linear", (), {})
    nn.functional = types.SimpleNamespace(relu=lambda x: max(x, 0))
    optim = types.ModuleType("fake_optim")
    optim.Adam = type("Adam", (), {})
    optim.lr_scheduler = types.SimpleNamespace(StepLR=type("StepLR", (), {}))
    backend.nn = nn
    backend.optim = optim
    backend.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(DataLoader=type("DataLoader", (), {}))
    )
    backend.linalg = types.SimpleNamespace(norm=lambda x: x)
    backend.fft = types.SimpleNamespace(fft=lambda x: x)
    backend.special = types.SimpleNamespace(expit=lambda x: x)
    backend.autograd = types.SimpleNamespace(grad=lambda x: x)
    # Dynamic missing attributes are NOT valid module names.
    backend.__getattr__ = lambda name: lambda *a: ("dynamically generated", name)
    return backend


prefixes = ("torch", "torch.")
original = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}
for k in original:
    del sys.modules[k]

try:
    backend = fake_backend()
    installed = bridge.install_aliases(backend, "webgpu")
    assert installed.__borch_compat__ is True
    assert installed.__borch_backend__ == "webgpu"
    assert bridge.install_aliases(backend, "webgpu") is installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.optim.lr_scheduler as sched
    from torch.nn import Linear
    from torch.nn.functional import relu
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torch.linalg import norm
    assert torch.Tensor is backend.Tensor
    assert torch.tensor([1, 2]) == ("fake-tensor", [1, 2])
    assert nn.Module is backend.nn.Module
    assert Linear is nn.Linear
    assert nn.functional is F
    assert F.relu(-5) == 0
    assert relu(-6) == 0
    assert optim.Adam is Adam
    assert sched.StepLR is optim.lr_scheduler.StepLR
    assert DataLoader is backend.utils.data.DataLoader
    assert norm(8) == 8
    assert "torch.distributed" not in sys.modules
    assert "torch.nn.init" not in sys.modules
    print("Standard PyTorch import forms resolve through Borch: PASS")
finally:
    for k in list(sys.modules):
        if k == "torch" or k.startswith("torch."):
            sys.modules.pop(k)
    sys.modules.update(original)


class FakePiplite(types.ModuleType):
    install_calls = []
    async def install(self, package, deps=False):
        assert deps is False
        self.install_calls.append(package)


async def test_cpu_fallback():
    backend = fake_backend()
    fake_piplite = FakePiplite("piplite")
    orig_import = bridge.importlib.import_module
    def import_named(name):
        if name == "borch_webgpu":
            raise ImportError("borch_webgpu needs JSPI (WebAssembly stack switching)")
        if name == "borch":
            return backend
        return orig_import(name)
    with patch.dict(sys.modules, {"piplite": fake_piplite}):
        with patch.object(bridge, "_wheel_url", return_value="/machine_learning/vendor/pyborch-1.14.1-py3-none-any.whl"), \
             patch.object(bridge, "_gpu_supported", return_value=False), \
             patch.object(bridge.importlib, "import_module", side_effect=import_named):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                installed = await bridge.ensure_torch()
                assert installed.__borch_backend__ == "numpy"
                assert fake_piplite.install_calls == ["/machine_learning/vendor/pyborch-1.14.1-py3-none-any.whl"]
                assert any("NumPy CPU" in str(x.message) for x in seen)
    for k in list(sys.modules):
        if k == "torch" or k.startswith("torch."):
            sys.modules.pop(k)


asyncio.run(test_cpu_fallback())
print("CPU fallback is explicit and does not silently claim GPU: PASS")
