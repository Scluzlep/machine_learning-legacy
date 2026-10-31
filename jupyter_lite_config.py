"""Legacy JupyterLite 0.1.x addon configuration (compat branch only).

PipliteAddon.piplite_urls is a TypedTuple. The modern
`--piplite-wheels .ml-wheel` CLI syntax sets it to a string and silently
disables the addon in JupyterLite 0.1.3. Use a tuple so the patched wheel is
actually indexed and installed by the browser kernel.
"""

c = get_config()  # noqa: F821 - provided by traitlets/JupyterLite
c.PipliteAddon.piplite_urls = (".ml-wheel",)
