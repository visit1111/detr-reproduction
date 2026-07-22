#!/usr/bin/env python3
"""Record a secret-safe manifest for a reproducible run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVE_FLAGS = {"--password", "--token", "--secret", "--api-key", "--key"}


def _sanitize_command(command: list[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        lowered = argument.lower()
        if lowered in SENSITIVE_FLAGS:
            sanitized.append(argument)
            redact_next = True
            continue
        if any(lowered.startswith(f"{flag}=") for flag in SENSITIVE_FLAGS):
            sanitized.append(argument.split("=", 1)[0] + "=<redacted>")
            continue
        sanitized.append(argument)
    return sanitized


def _git(args: list[str]) -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "torchvision", "scipy", "pycocotools", "matplotlib", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(command: list[str], config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    status = _git(["status", "--porcelain"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": _sanitize_command([str(argument) for argument in command]),
        "config": config,
        "output_dir": str(Path(output_dir).absolute()),
        "git": {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(status)},
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": _package_versions(),
    }


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(args.command, config, args.output_dir)
    print(write_manifest(args.output_dir, manifest))


if __name__ == "__main__":
    main()
