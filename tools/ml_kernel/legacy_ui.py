"""Targeted bridges for the 2022 browser scientific and widget stack.

Only install the missing historical API attributes. These do not replace the
Matplotlib rendering or JupyterLite Comm implementations, and are safe to call
more than once from the zero-edit kernel preprocessor.
"""
from __future__ import annotations


def install_matplotlib_compat(rcparams_cls=None):
    """matplotlib_inline>=0.2 reads RcParams._get, absent in Matplotlib 3.5."""
    if rcparams_cls is None:
        from matplotlib import RcParams
        rcparams_cls = RcParams
    if not hasattr(rcparams_cls, "_get"):
        # Access without any parsing/auto-select, like Matplotlib 3.7+.
        def _get(self, key):
            return dict.__getitem__(self, key)
        rcparams_cls._get = _get
    return rcparams_cls


def install_widget_comm_compat(comm_cls=None):
    """Make JupyterLite's working Comm.send usable from ipywidgets 7.6.

    Widgets 7 treats `comm.kernel is not None` as a 'comm is open' check.
    The lightweight browser Comm has no `kernel` reference, but already
    knows how to dispatch comm_open/comm_msg; return the comm itself as
    its live marker, and None once closed. Do not wrap send or change IDs.
    """
    if comm_cls is None:
        from ipywidgets.widgets import widget
        comm_cls = widget.Comm
    if not isinstance(comm_cls, type):
        raise TypeError("Expected ipywidgets 7 Comm class")
    if not hasattr(comm_cls, "kernel"):
        def _kernel(self):
            if getattr(self, "_closed", False) or getattr(self, "closed", False):
                return None
            return self
        comm_cls.kernel = property(_kernel)
    return comm_cls
