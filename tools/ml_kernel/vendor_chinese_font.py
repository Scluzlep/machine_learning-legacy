#!/usr/bin/env python3
"""Vendor Adobe Source Han Sans *Simplified Chinese* for browser Matplotlib.

Use the actual Adobe SC regional OTF, not a pan-CJK or JP/TC fallback.
Pin the upstream source to a known Git blob and preserve the OFL license.
Only generated Pages assets receive the font; course notebooks stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

UPSTREAM_COMMIT = "a4f7cf94edfb9d7ffbdfc4841de276358bd7e0f2"
BASE = f"https://raw.githubusercontent.com/adobe-fonts/source-han-sans/{UPSTREAM_COMMIT}"
NAME = "SourceHanSansSC-Regular.otf"
FONT_URL = f"{BASE}/OTF/SimplifiedChinese/{NAME}"
LICENSE_URL = f"{BASE}/LICENSE.txt"
EXPECTED_BLOB_SHA1 = "8113ad53da18b6a08b133c6c460ae1d818a0f865"
EXPECTED_SIZE = 16_529_832
CHINESE_TEST = "交互式测试机器学习中文简体数据字体归一化损失"


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    font_path = destination / NAME
    if font_path.is_file():
        payload = font_path.read_bytes()
    else:
        with urlopen(FONT_URL, timeout=90) as response:
            payload = response.read()
    git_blob = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    if len(payload) != EXPECTED_SIZE or git_blob != EXPECTED_BLOB_SHA1:
        raise ValueError(
            f"Adobe SC OTF integrity mismatch: size={len(payload)}, sha1={git_blob}"
        )
    # Verify SC face AND character coverage; a Japanese or Traditional face
    # must never be silently accepted as a substitute.
    from fontTools.ttLib import TTFont

    font = TTFont(font_path if font_path.is_file() else __import__("io").BytesIO(payload))
    names = {
        str(n.toUnicode()) for n in font["name"].names
        if n.nameID in (1, 4, 6, 16)
    }
    if not any("Source Han Sans SC" in x or "SourceHanSansSC" in x for x in names):
        raise ValueError(f"Expected Simplified Chinese Source Han Sans face, got: {names}")
    mapping = font.getBestCmap()
    missing = [c for c in CHINESE_TEST if ord(c) not in mapping]
    if missing:
        raise ValueError("Source Han Sans SC missing Chinese glyphs: " + "".join(missing))
    font.close()
    font_path.write_bytes(payload)
    with urlopen(LICENSE_URL, timeout=45) as response:
        license_text = response.read()
    if b"SIL OPEN FONT LICENSE" not in license_text.upper():
        raise ValueError("Source Han Sans OFL license file missing")
    (destination / "OFL-LICENSE.txt").write_bytes(license_text)
    print(f"Bundled Adobe Source Han Sans SC ({len(payload) / 1024**2:.1f} MiB), "
          f"full Simplified Chinese OTF; license included; Git blob verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/assets/fonts"))
    args = parser.parse_args()
    download(args.output)
