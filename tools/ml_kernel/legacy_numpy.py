"""Restore removed NumPy 1.19 Python aliases on the Pyodide 1.24 runtime.

Only names are patched: this does not spoof compiled NumPy 1.19 ABI.
"""
from __future__ import annotations


def install():
    import numpy as np
    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "str": str, "long": int, "unicode": str,
    }
    for name, target in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, target)
    if "asscalar" not in np.__dict__:
        np.asscalar = lambda value: np.asarray(value).item()
