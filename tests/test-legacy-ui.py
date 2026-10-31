#!/usr/bin/env python3
"""Pure Python regression for the legacy JupyterLite UI protocol adapters."""
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "tools/ml_kernel/legacy_ui.py"
spec = importlib.util.spec_from_file_location("browser_ml_legacy_ui", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OldRcParams(dict):
    pass


config = OldRcParams({"backend": "module://matplotlib_inline.backend_inline"})
assert not hasattr(config, "_get")
module.install_matplotlib_compat(OldRcParams)
module.install_matplotlib_compat(OldRcParams)  # Idempotent.
assert config._get("backend") == "module://matplotlib_inline.backend_inline"
try:
    config._get("not-present")
except KeyError:
    pass
else:
    raise AssertionError("Missing RcParam must raise KeyError")


class BrowserComm:
    def __init__(self):
        self._closed = False
        self.sent = []

    def send(self, message):
        self.sent.append(message)


module.install_widget_comm_compat(BrowserComm)
module.install_widget_comm_compat(BrowserComm)
comm = BrowserComm()
assert comm.kernel is comm
comm.send({"widget": 3})
assert comm.sent == [{"widget": 3}]
comm._closed = True
assert comm.kernel is None
print("Legacy RcParams._get and ipywidgets7 Comm.kernel bridges: PASS")
