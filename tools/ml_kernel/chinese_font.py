"""Configure Matplotlib to use Adobe's authentic Simplified Chinese glyphs.

This is initialized automatically by the custom JupyterLite Pyodide kernel,
before Notebook code runs. No notebook changes, font pip packages or OS fonts.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

_INSTALLED = False
_FONT_NAME = "SourceHanSansSC-Regular.otf"
_FONT_FAMILY = "Source Han Sans SC"
_LABELS = "交互式测试机器学习中文简体数据字体归一化损失"


def _site_font_url() -> str:
    from js import globalThis

    url = urlsplit(str(globalThis.location.href))
    if url.scheme not in ("https", "http"):
        raise RuntimeError("Cannot resolve Pages font path from kernel Worker URL")
    path = url.path
    for marker in (
        "/lab/", "/repl/", "/build/", "/api/", "/kernels/",
        "/notebooks/", "/tree/", "/extensions/",
    ):
        if marker in path:
            path = path.split(marker, 1)[0]
            break
    else:
        path = path.rstrip("/").rsplit("/", 1)[0]
    return f"{url.scheme}://{url.netloc}{path}/assets/fonts/{_FONT_NAME}"


async def install() -> str:
    """Register the SC face once and set the default for all Matplotlib figures."""
    global _INSTALLED
    if _INSTALLED:
        return _FONT_FAMILY
    from matplotlib import font_manager, rcParams
    from matplotlib.ft2font import FT2Font
    from pyodide.http import pyfetch

    path = Path("/home/pyodide/.course-fonts") / _FONT_NAME
    if not path.exists():
        response = await pyfetch(_site_font_url())
        if not response.ok:
            raise RuntimeError(f"Source Han Sans SC asset HTTP {response.status}")
        font_data = await response.bytes()
        if not font_data[:4] == b"OTTO":
            raise RuntimeError("Downloaded SC font is not an OpenType font")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(font_data)
    font = FT2Font(str(path))
    missing = [char for char in _LABELS if ord(char) not in font.get_charmap()]
    if missing:
        raise RuntimeError("SC font missing Chinese glyphs: " + "".join(missing))
    font_manager.fontManager.addfont(str(path))
    detected_family = font_manager.FontProperties(fname=str(path)).get_name()
    if detected_family != _FONT_FAMILY:
        raise RuntimeError(
            f"Wrong regional CJK font: expected {_FONT_FAMILY}, got {detected_family}"
        )

    # Keep original CJK glyphs from the official SC face; DejaVu Sans only
    # serves as the fallback for unrelated non-Chinese symbols.
    sans = [name for name in rcParams["font.sans-serif"]
            if name not in (_FONT_FAMILY, "Source Han Sans", "Noto Sans CJK JP",
                            "Noto Sans CJK TC", "Noto Sans CJK KR")]
    rcParams["font.family"] = ["sans-serif"]
    rcParams["font.sans-serif"] = [_FONT_FAMILY, *sans]
    selected = font_manager.findfont(
        font_manager.FontProperties(family=_FONT_FAMILY),
        fallback_to_default=False,
    )
    if Path(selected).name != _FONT_NAME:
        raise RuntimeError("Matplotlib did not select the official SC regional font")
    _INSTALLED = True
    return detected_family
