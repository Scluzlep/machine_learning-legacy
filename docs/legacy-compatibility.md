# Legacy browser runtime — compat/pyodide-legacy-course

This branch stages an **independent legacy JupyterLite kernel**. It does not
change or rewrite any original `.ipynb`, and does not modify the `main`
branch. A future dual-kernel site must register separate kernel specifications
and Workers for the modern and legacy Pyodide ABI, rather than mixing wheels.

## Environment comparison

The original environment was recorded in `machine_learning/environment/`:
Python 3.8.10, NumPy 1.19.5, SciPy 1.10.1, scikit-learn 0.24.1, pandas 1.2.0,
Matplotlib 3.4.3, TensorFlow/Keras 2.6.0, h5py 3.1.0,
ipywidgets 7.6.5 and ipympl 0.9.3.

| Package | Legacy browser stack | Notes |
| --- | --- | --- |
| Pyodide | 0.23.4 | matched WASM runtime and `repodata.json` |
| Python | 3.11.2 | **not** original Python 3.8.10 |
| JupyterLite core/kernel | 0.1.3 / 0.1.1 | matching legacy JupyterLab frontend |
| JupyterLab | 3.5.3 | legacy frontend, not the modern main |
| NumPy | 1.24.2 | NumPy 1.x aliases reintroduced at runtime |
| SciPy | 1.9.3 | removed `scipy.misc.imresize` and `scipy.ndimage.imread` restored using Pillow |
| scikit-learn | 1.2.2 | `load_boston` and `normalize=False` adapter |
| pandas | 1.5.3 | same major version as original |
| Matplotlib | 3.5.2 | classic IPython widget backend registration |
| Pillow | 9.1.1 | WASM wheel downloaded on demand |
| h5py | 3.7.0 | WASM wheel downloaded on demand |
| IPython | 8.12.3 | pre-bundled in Piplite index; prevents upstream IPython 9 conflict |
| ipywidgets | 7.6.5 | original widget Python API |
| ipympl | 0.9.3 | original interactive Matplotlib API |
| seaborn | 0.11.2 | compatible Python wheel bundled by build |
| TensorFlow/Keras | TensorFlow.js 4.22.0 with Keras 2.6 subset | not native TensorFlow; unsupported features error |

The packages above are *Pyodide 0.23.4's own* released scientific wheel
versions. They cannot be freely downgraded to native Linux wheels: Emscripten
ABI and Python interpreter builds must match. The original environment contains
`file://` installation references and is not directly transferable to WASM.

## Runtime behavior

- The patched `pyodide_kernel-0.1.1` wheel calls
  `kernel_runtime.prepare_cell` before interpreting a cell.
- The startup hook loads old-compatible compiled WASM scientific wheels
  (NumPy, SciPy, scikit-learn, pandas, Matplotlib) and widget Python wheels.
- NumPy aliases, legacy scikit-learn constructors, and removed SciPy image
  helpers are patched **inside the worker**, not the notebooks.
- TensorFlow.js is transparently exposed through the course's existing
  `tf.keras` bridge. Training is asynchronous internally, with top-level
  `model.fit(...)` adapted by the kernel; the course code stays synchronous.
- The browser may use WebGPU, WebGL, or CPU, depending on availability.
  TensorFlow.js numerical behavior does not equal native TensorFlow 2.6 exactly.

## Build without deploying

The legacy `jupyter_lite_config.py` specifies `PipliteAddon.piplite_urls`\nas a **tuple**, because the CLI option accepted by modern JupyterLite breaks\nthis old addon. A successful static build without `dist/pypi/all.json`\nand the custom kernel wheel is an invalid build.\n\nInstall the pinned `site-requirements.txt` in a separate Python 3.11
builder environment, then execute the build and test commands from the
compat branch's `.github/workflows/pages.yml`. You may manually dispatch a
build on the compat branch; it is prevented from uploading to GitHub Pages.

`tools/ml_kernel/vendor_pyodide.py` hosts the legacy `pyodide.mjs` ESM Worker entry,
`repodata.json`, WASM and the official scientific wheel dependency closure.
Other optional packages retain a Pyodide **0.23.4** CDN fallback.

`tests/test-notebook-integrity.py` checks raw Git blob hashes and verifies
deployed code-cell sources are identical. The Python kernel tests check the
old patched wheel and compatibility shims. Browser tests are still required
to establish end-to-end Notebook execution: **static checks alone do not
prove that all course exercises work in Chromium.**

The legacy kernel bootstraps IPython before cell hooks run. To avoid a 2026
micropip resolver conflict (`ipympl 0.9.3` requires `ipython<9` while
unconstrained installs prefer 9.x), the local Piplite index also bundles
`ipython==8.12.3`, and the cell hook checks the same pin.

## Webpack Worker module loading

JupyterLite Pyodide kernel 0.1.1's published Worker bundle transforms
`await import(pyodideUrl)` into a Webpack require-context module that always
throws `MODULE_NOT_FOUND`. After building `dist/`, run
`python tools/ml_kernel/patch_legacy_worker.py --dist dist`. This strictly
patches a single compiled `loadPyodide` reference back into browser-native
`import(url)`; a missing or ambiguous match **fails the build**. The original
upstream kernel source, wheel contents, and lesson Notebooks are untouched.

## Browser widget dependency isolation

Old `ipywidgets==7.6.5` requests `widgetsnbextension~=3.5.0` from
PyPI; using a late-2026 resolver without a local wheel can fail in Pyodide
(`Can't find a pure Python 3 wheel`). The exact legacy
`widgetsnbextension==3.5.2`, `ipython-genutils==0.2.0`, and
`jupyterlab-widgets==1.1.11` are added to the local Piplite wheel index.
The worker installs these known Python-side extensions with `deps=False`
to avoid accidentally downloading native Jupyter Server and new Tornado
dependencies. The old kernel separately supplies its pinned IPython,
ipykernel and traitlets. Host-side `jupyterlab-pygments` is pinned to
JupyterLab 3-era 0.2.2 to avoid the newer 0.3.x JupyterLab 4 extension.
