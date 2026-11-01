"""Expose Borch's Pyodide API under standard PyTorch import names.

This is an intentionally *partial* compatibility facade, not native PyTorch.
It is installed only in the custom browser kernel, for explicit torch imports.
The upstream teaching notebooks and their kernelspecs remain untouched.
"""
from __future__ import annotations

import ast
import importlib
import importlib.machinery
import re
import sys
import types
import warnings

BORCH_VERSION = "1.14.1"
PYBORCH_DISTRIBUTION = f"pyborch=={BORCH_VERSION}"
PYBORCH_WHEEL = f"pyborch-{BORCH_VERSION}-py3-none-any.whl"
_COMPAT_MARKER = "__borch_compat__"
_ALREADY_READY = False

# An explicit import is required. Ignore comments, docstrings and plain use of
# variable names like "torch_tensor". AST handles nested imports too.
def _parse_cell(code: str):
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        # IPython may leave line magics in the submitted cell.
        ordinary = "\n".join(
            line for line in code.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            return ast.parse(ordinary)
        except (SyntaxError, ValueError):
            return None


def needs_torch(code: str) -> bool:
    tree = _parse_cell(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "torch" or a.name.startswith("torch.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if name == "torch" or name.startswith("torch."):
                return True
    return False


def _proxy_module(fullname: str, target: object, *, package: bool = False) -> types.ModuleType:
    """A real Python module for importlib, backed by a Borch object/module.

    In particular, `from torch.nn.functional import relu` needs an actual entry
    in sys.modules; assigning `torch.nn = borch_webgpu.nn` is not sufficient.
    """
    mod = types.ModuleType(fullname, f"Borch compatibility proxy for {fullname}")
    mod.__package__ = fullname if package else fullname.rpartition(".")[0]
    mod.__spec__ = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=package)
    if package:
        mod.__path__ = []

    def _getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {fullname!r} has no attribute {name!r}")
        return getattr(target, name)

    def _dir():
        return sorted(set(mod.__dict__) | set(dir(target)))

    mod.__getattr__ = _getattr
    mod.__dir__ = _dir
    return mod


def install_aliases(backend: types.ModuleType, backend_name: str) -> types.ModuleType:
    """Bind standard torch package/submodule imports to one consistent backend.

    All imports use either the GPU binding or the NumPy core for the kernel's
    entire lifetime; mixed CPU/GPU Tensor classes would be unsafe.
    """
    existing = sys.modules.get("torch")
    if existing is not None:
        if getattr(existing, _COMPAT_MARKER, False):
            return existing
        raise RuntimeError("A real torch module is already loaded; refusing to replace it with Borch")

    root = _proxy_module("torch", backend, package=True)
    root.__version__ = f"0.0.0+borch{BORCH_VERSION}"
    root.__borch_compat__ = True
    root.__borch_backend__ = backend_name

    # Register only names that the selected Borch backend really exports.
    # Do not fabricate PyTorch subsystems (e.g. torch.distributed or torch.jit).
    specs = [
        ("nn", backend, "nn", True),
        ("optim", backend, "optim", True),
        ("autograd", backend, "autograd", False),
        ("linalg", backend, "linalg", False),
        ("fft", backend, "fft", False),
        ("special", backend, "special", False),
        ("utils", backend, "utils", True),
        ("onnx", backend, "onnx", False),
    ]
    allmods = {"torch": root}
    for short, parent, attr, package in specs:
        try:
            # Module-level __getattr__ on Borch can manufacture a callable for
            # unknown names; require real exports for package namespaces.
            target = vars(parent)[attr]
        except (AttributeError, ImportError, KeyError, TypeError):
            continue
        allmods[f"torch.{short}"] = _proxy_module(f"torch.{short}", target, package=package)

    children = (
        ("nn.functional", "nn", "functional", False),
        ("optim.lr_scheduler", "optim", "lr_scheduler", False),
        ("utils.data", "utils", "data", False),
    )
    for suffix, parent_key, attr, package in children:
        parent_mod = allmods.get("torch." + parent_key)
        if parent_mod is None:
            continue
        try:
            target = getattr(parent_mod, attr)
        except (AttributeError, ImportError):
            continue
        allmods["torch." + suffix] = _proxy_module(
            "torch." + suffix, target, package=package
        )

    # Publish atomically after building the whole submodule map. Empty packages
    # make unsupported submodule imports fail rather than masquerade as success.
    sys.modules.update(allmods)
    for fullname, mod in allmods.items():
        if fullname == "torch":
            continue
        parent, _, attr = fullname.rpartition(".")
        setattr(allmods[parent], attr, mod)
    return root


def _wheel_url() -> str:
    """Wheel comes from this Pages origin, not PyPI or an external CDN."""
    from js import location
    parts = str(location.pathname).split("/")
    if len(parts) < 2 or parts[1] not in ("machine_learning", "machine_learning-legacy"):
        raise RuntimeError("Cannot determine local JupyterLite site base for Borch wheel")
    return f"/{parts[1]}/vendor/{PYBORCH_WHEEL}"


async def _gpu_supported() -> bool:
    """Verify an actual GPU adapter, not just navigator.gpu's presence.

    Headless Chromium and blocklisted GPUs may expose WebGPU but resolve
    requestAdapter() to null. Probing *before* importing Borch avoids
    initializing its global runtime in an unusable state.
    """
    try:
        from js import WebAssembly, navigator
        from pyodide.ffi import run_sync
        if not callable(run_sync):
            return False
        if getattr(WebAssembly, "Suspending", None) is None:
            return False
        gpu = getattr(navigator, "gpu", None)
        if gpu is None:
            return False
        adapter = await gpu.requestAdapter()
        return adapter is not None and type(adapter).__name__ not in ("JsNull", "jsnull")
    except Exception:
        return False


def _missing_gpu_or_jspi(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        x in message for x in (
            "no webgpu",
            "no gpu adapter",
            "needs jspi",
            "no jspi",
            "webassembly stack switching",
        )
    )


async def ensure_torch() -> types.ModuleType:
    """Install the vendored wheel and expose a single coherent torch facade."""
    global _ALREADY_READY
    existing = sys.modules.get("torch")
    if existing is not None:
        if getattr(existing, _COMPAT_MARKER, False):
            return existing
        # Don't overwrite a user-installed real torch, if an environment has one.
        return existing
    import piplite

    # The package is bundled into the site's piplite index at build time.
    await piplite.install(_wheel_url(), deps=False)

    gpu_possible = await _gpu_supported()
    gpu_error = None
    if gpu_possible:
        try:
            backend = importlib.import_module("borch_webgpu")
            name = "webgpu"
        except Exception as exc:
            # JSPI/WebGPU failures can be pyodide.ffi.JsException rather
            # than Python ImportError. Do not mask unrelated import bugs.
            if not _missing_gpu_or_jspi(exc):
                raise
            gpu_error = str(exc)
    else:
        gpu_error = "WebGPU or JavaScript Promise Integration (JSPI) not available"
    if gpu_error is not None:
        # This is an explicit *kernel-wide* CPU fallback, never an automatic
        # mid-training migration. It does not claim feature parity with the GPU.
        backend = importlib.import_module("borch")
        name = "numpy"
        warnings.warn(
            "Borch WebGPU/JSPI unavailable: torch compatibility is running on "
            "the NumPy CPU backend. Check torch.__borch_backend__. "
            f"GPU initialization: {gpu_error}",
            RuntimeWarning,
            stacklevel=2,
        )

    mod = install_aliases(backend, name)
    _ALREADY_READY = True
    return mod


async def prepare_torch(code: str) -> None:
    if needs_torch(code):
        await ensure_torch()
