#!/usr/bin/env python3
"""Compare DETR COCO metrics with a versioned reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_NAMES = ("AP", "AP50", "AP75")


def _extract_metrics(data: dict[str, Any]) -> dict[str, float]:
    if "test_coco_eval_bbox" in data:
        values = data["test_coco_eval_bbox"]
        if not isinstance(values, list) or len(values) < 3:
            raise ValueError("test_coco_eval_bbox must contain at least three values")
        scale = 100.0 if max(abs(float(value)) for value in values[:3]) <= 1.0 else 1.0
        return {name: float(value) * scale for name, value in zip(METRIC_NAMES, values[:3])}
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and all(name in metrics for name in METRIC_NAMES):
        return {name: float(metrics[name]) for name in METRIC_NAMES}
    raise ValueError("Metrics file does not contain DETR bbox metrics")


def verify(
    actual_path: Path,
    reference_path: Path,
    tolerance: float | None = None,
) -> dict[str, Any]:
    actual_data = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    reference_data = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    actual = _extract_metrics(actual_data)
    reference = _extract_metrics(reference_data)
    allowed = float(reference_data.get("tolerance", 0.2) if tolerance is None else tolerance)
    differences = {name: abs(actual[name] - reference[name]) for name in METRIC_NAMES}
    for name in METRIC_NAMES:
        if differences[name] > allowed + 1e-12:
            raise ValueError(
                f"{name} differs by {differences[name]:.3f}, exceeding tolerance {allowed:.3f}"
            )
    return {
        "passed": True,
        "actual": actual,
        "reference": reference,
        "differences": differences,
        "tolerance": allowed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tolerance", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.actual, args.reference, args.tolerance)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
