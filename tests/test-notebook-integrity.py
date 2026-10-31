#!/usr/bin/env python3
"""Protect all original notebooks, including source cells in deployment copies.

No rewriting of source notebooks is authorized by the compatibility build.
The manifest stores Git blob SHA-1 of each raw source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/notebook-blob-manifest.json"
SOURCE = ROOT / "machine_learning"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify(staged: Path | None = None) -> int:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["notebooks"]
    actual = {p.relative_to(ROOT).as_posix(): p for p in SOURCE.rglob("*.ipynb")}
    if set(actual) != set(expected):
        raise AssertionError(
            f"Notebook set changed: missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}"
        )
    for name, filename in actual.items():
        raw = filename.read_bytes()
        assert blob_sha(raw) == expected[name], f"Original notebook modified: {name}"
        original = json.loads(raw)
        if staged is None:
            continue
        deployed = staged / filename.relative_to(SOURCE)
        assert deployed.is_file(), f"Notebook not deployed: {name}"
        published = json.loads(deployed.read_text(encoding="utf-8"))
        assert len(original["cells"]) == len(published["cells"]), name
        for index, (src, dst) in enumerate(zip(original["cells"], published["cells"])):
            assert src["cell_type"] == dst["cell_type"], (name, index)
            if src["cell_type"] == "code":
                assert src["source"] == dst["source"], (name, index, "code changed")
                assert src.get("outputs") == dst.get("outputs"), (name, index, "outputs changed")
                assert src.get("execution_count") == dst.get("execution_count"), (
                    name, index, "execution count changed"
                )
    print(f"Notebook source integrity PASS ({len(actual)} originals"
          + (", code cells identical in staged copies)" if staged else ")"))
    return len(actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", type=Path)
    verify(parser.parse_args().staged)
