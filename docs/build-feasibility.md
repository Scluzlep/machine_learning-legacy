# Browser-only build feasibility

Inspected 42 substantive Notebooks across seven groups. Of these, **9 import TensorFlow directly**; 33 have no direct TensorFlow import. Counts reflect static import detection, not end-to-end execution.

| Category | Browser/Pyodide feasibility |
|---|---|
| NumPy, Matplotlib, SciPy, scikit-learn, pandas | Generally feasible with Pyodide-compatible distributions; package versions differ from AutoDL |
| Notebook helper `.py` modules, `.npy`, `.h5`, images | Portable as static files; larger datasets increase browser download/storage costs |
| `ipywidgets`, interactive Matplotlib and notebook event handlers | Need explicit compatibility testing / sometimes patching |
| TensorFlow / `tf.keras` (9 notebooks) | **Not executable unchanged** with official native TensorFlow in Pyodide; would need Python API adaptation to TF.js, refactoring to NumPy, or remote Python kernel |
| Original Python 3.8.10 env & CUDA 10.0 | Neither is carried into browser; a JupyterLite build uses a separate modern build Python environment |

Caveats: even a non-TensorFlow Notebook may rely on indirect TensorFlow imports, non-WASM dependencies, interactive UI, notebook outputs, missing files, or significant training memory. 14 of the 42 notebooks use `%matplotlib widget`, and one imports seaborn despite seaborn being absent from the captured installed-package lists. Test execution lab by lab. With Pyodide, Python imports execute on the client and a first-load package download is expected. CUDA/GPU acceleration for original Python TensorFlow is not available.

For the nine TensorFlow labs, TensorFlow.js is a viable separate browser backend, but it does not provide drop-in Python `tensorflow` / `tf.keras` compatibility. A browser-compatible rewrite or a hybrid remote service would be needed.

The old dependency snapshot includes TensorFlow 2.6.0, TensorFlow Estimator 2.15.0, NumPy 1.19.5, scikit-learn 0.24.1 and scipy 1.10.1. The `environment.yml` capture failed and is zero bytes; requirements include nonportable `file:///` references.

Site building and publishing is **highly feasible** as a static asset task; preserving full runtime execution of all Notebooks is **not yet supported**.
