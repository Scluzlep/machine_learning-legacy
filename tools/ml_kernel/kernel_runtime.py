"""Transparent TensorFlow.js compatibility for unmodified course cells.

This runs inside a customized Pyodide kernel before IPython executes a cell.
The .ipynb document is never changed.
"""
from __future__ import annotations

import ast
import re
import sys

_TF_MODEL_NAMES: set[str] = set()
_TF_MODEL_COLLECTIONS: set[str] = set()
_COURSE_DEPENDENCIES_READY = False


async def initialize_course_dependencies():
    """Load the course's Python UI packages once per kernel, not per cell.

    Older JupyterLite 0.1 does not preload scientific wheels via load options.
    Load the Pyodide 0.23.4 scientific stack here before the first cell.
    The Notebook source is left untouched.
    """
    global _COURSE_DEPENDENCIES_READY
    if _COURSE_DEPENDENCIES_READY:
        return
    from pyodide_js import loadPackage
    await loadPackage(["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "micropip"])
    from .legacy_numpy import install as install_numpy
    install_numpy()
    # Local course helpers import sklearn estimators internally; patch those
    # symbols before the modules execute, not only when the cell imports them.
    from .legacy_scikit import install_boston, install_linear_regression, install_ridge
    install_boston()
    install_linear_regression()
    install_ridge()
    # matplotlib_inline 0.2.x assumes Matplotlib >=3.7. Restore the
    # harmless internal RcParams getter on the pinned WASM Matplotlib 3.5.2.
    from .legacy_ui import install_matplotlib_compat
    install_matplotlib_compat()
    import piplite
    from .imports_bootstrap import _register_ipympl_backends

    # In the Pyodide worker, install browser widgets without dragging native
    # Jupyter Server / recent ipykernel / tornado dependencies from PyPI.
    # The actual kernel already provides IPython, comm, traitlets and ipykernel;
    # all compiled data/scientific libraries were loaded from Pyodide above.
    await piplite.install(
        ["widgetsnbextension==3.5.2", "jupyterlab-widgets==1.1.11",
         "ipython-genutils==0.2.0"], deps=False
    )
    await piplite.install(
        ["ipywidgets==7.6.5", "ipympl==0.9.3", "seaborn==0.11.2"],
        deps=False
    )
    # ipywidgets 7.x expects Comm.kernel, but the browser kernel's Comm
    # exposes send() without that legacy attribute. Restore the marker.
    from .legacy_ui import install_widget_comm_compat
    install_widget_comm_compat()
    _register_ipympl_backends()
    from .chinese_font import install as install_chinese_font
    await install_chinese_font()
    _COURSE_DEPENDENCIES_READY = True




def _is_tf_constructor(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Sequential"
    return isinstance(func, ast.Attribute) and func.attr == "Sequential"


class _ModelFitTransformer(ast.NodeTransformer):
    def __init__(self, names: set[str], collections: set[str]):
        self.names = names
        self.collections = collections
        self.changed = False

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        func = node.func
        if (isinstance(func, ast.Attribute)
                and func.attr == "fit"
                and isinstance(func.value, ast.Name)
                and func.value.id in self.names) or (
                isinstance(func, ast.Attribute)
                and func.attr == "fit"
                and isinstance(func.value, ast.Subscript)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in self.collections):
            self.changed = True
            return ast.copy_location(ast.Await(node), node)
        return node


def transform_tf_fit(code: str) -> str:
    """Only transform calls on variables known to hold TF.js Keras models."""
    if "Sequential" not in code and (".fit(" not in code or not (_TF_MODEL_NAMES or _TF_MODEL_COLLECTIONS)):
        return code
    if re.search(r"^\s*[%!]", code, re.MULTILINE):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_tf_constructor(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _TF_MODEL_NAMES.add(target.id)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    _TF_MODEL_COLLECTIONS.add(target.value.id)
        elif isinstance(node, ast.AnnAssign) and _is_tf_constructor(node.value):
            if isinstance(node.target, ast.Name):
                _TF_MODEL_NAMES.add(node.target.id)
    if not (_TF_MODEL_NAMES or _TF_MODEL_COLLECTIONS) or ".fit(" not in code:
        return code
    transformer = _ModelFitTransformer(_TF_MODEL_NAMES, _TF_MODEL_COLLECTIONS)
    new_tree = transformer.visit(tree)
    if not transformer.changed:
        return code
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def uses_legacy_scipy(code: str) -> bool:
    """Check references to SciPy image APIs removed before this WASM release."""
    return bool(re.search(
        r"\b(?:(?:scipy\.)?misc\.(?:imread|imresize|imsave)"
        r"|(?:scipy\.)?ndimage\.imread)\b", code
    ))


async def prepare_cell(code: str) -> str:
    if not isinstance(code, str):
        return code
    await initialize_course_dependencies()
    # Resolve hidden imports such as lab_utils_uni -> ipywidgets before
    # executing the original unchanged course cell.
    from .imports_bootstrap import discover, install_missing
    # Imports in lab_coffee_utils.py, autils.py and other local modules can
    # import TensorFlow without the notebook cell mentioning it directly.
    # Discover recursively *before* importing those modules into Python.
    dependencies = discover(code)
    if "tensorflow" in dependencies and "tensorflow" not in sys.modules:
        # The kernel's regular dependency loader runs after this hook.
        # NumPy is needed immediately by the TensorFlow.js bridge.
        from pyodide_js import loadPackagesFromImports
        await loadPackagesFromImports("import numpy")
        from .auto_tfjs.tfjs_compat import install
        await install()
    await install_missing(code)
    # Repair only the removed sklearn APIs used by historical course cells.
    # Load native Pyodide wheels before importing the relevant modules.
    if re.search(r"\bload_boston\b", code):
        from pyodide_js import loadPackagesFromImports
        await loadPackagesFromImports("import sklearn")
        from .legacy_scikit import install_boston
        install_boston()
    if re.search(r"\bfrom\s+sklearn\.linear_model\s+import\b", code):
        from pyodide_js import loadPackagesFromImports
        await loadPackagesFromImports("import sklearn")
        from .legacy_scikit import install_linear_regression
        install_linear_regression()
    if uses_legacy_scipy(code):
        from pyodide_js import loadPackage
        await loadPackage(["scipy", "Pillow"])
        from .legacy_scipy import install as install_scipy
        install_scipy()
    if "tensorflow" not in sys.modules:
        return code
    return transform_tf_fit(code)
