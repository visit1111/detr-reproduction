#!/usr/bin/env python3
"""Download a checkpoint only after validating its SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quarantine(path: Path, digest: str, name_source: Path | None = None) -> Path:
    source = path if name_source is None else name_source
    base = path.with_name(f"{source.name}.invalid-{digest[:12]}")
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{base.name}-{counter}")
        counter += 1
    os.replace(path, candidate)
    return candidate


def download(url: str, destination: Path, sha256: str) -> dict[str, Any]:
    expected = sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")

    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        current = sha256_file(destination)
        if current == expected:
            return {
                "path": str(destination),
                "sha256": current,
                "bytes": destination.stat().st_size,
                "cached": True,
            }
        _quarantine(destination, current)

    partial = destination.with_name(f"{destination.name}.part")
    if partial.exists():
        _quarantine(partial, sha256_file(partial))
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        actual = sha256_file(partial)
        if actual != expected:
            _quarantine(partial, actual, destination)
            raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            _quarantine(partial, sha256_file(partial))
        raise

    return {
        "path": str(destination),
        "sha256": expected,
        "bytes": destination.stat().st_size,
        "cached": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(download(args.url, args.destination, args.sha256), sort_keys=True))


if __name__ == "__main__":
    main()
