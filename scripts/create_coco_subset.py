#!/usr/bin/env python3
"""Create a deterministic image-level subset of a COCO detection dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

try:
    from scripts.verify_dataset import verify_split
except ModuleNotFoundError:  # Support `python scripts/create_coco_subset.py`.
    from verify_dataset import verify_split


def _ensure_link(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise FileExistsError(f"Existing link points elsewhere: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def create_subset(
    source_root: Path,
    target_root: Path,
    fraction: float,
    seed: int,
    *,
    link_images: bool = True,
) -> dict[str, Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")

    source = Path(source_root).resolve(strict=True)
    target = Path(target_root).absolute()
    if source == target.resolve():
        raise ValueError("Source and target roots must differ")

    verify_split(source, "train", check_files=False)
    verify_split(source, "val", check_files=False)
    source_annotation = source / "annotations" / "instances_train2017.json"
    data = json.loads(source_annotation.read_text(encoding="utf-8"))

    image_ids = [image["id"] for image in data["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs in source annotation")

    selected_count = max(1, round(len(image_ids) * fraction))
    selected_ids = set(random.Random(seed).sample(sorted(image_ids), selected_count))
    selected_images = [image for image in data["images"] if image["id"] in selected_ids]
    selected_annotations = [
        annotation for annotation in data["annotations"] if annotation["image_id"] in selected_ids
    ]

    category_ids = {category["id"] for category in data["categories"]}
    invalid_categories = {
        annotation["category_id"] for annotation in selected_annotations
    } - category_ids
    if invalid_categories:
        raise ValueError(f"Unknown category IDs in selected annotations: {sorted(invalid_categories)}")

    target_annotations = target / "annotations"
    target_annotation = target_annotations / "instances_train2017.json"
    if target_annotation.exists() or target_annotation.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing annotation: {target_annotation}")
    target_annotations.mkdir(parents=True, exist_ok=True)

    if link_images:
        _ensure_link(target / "train2017", source / "train2017")
        _ensure_link(target / "val2017", source / "val2017")
        _ensure_link(
            target_annotations / "instances_val2017.json",
            source / "annotations" / "instances_val2017.json",
        )

    subset = {key: value for key, value in data.items() if key not in {"images", "annotations"}}
    subset["images"] = selected_images
    subset["annotations"] = selected_annotations
    temporary = target_annotation.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target_annotation)

    represented = {annotation["category_id"] for annotation in selected_annotations}
    summary = {
        "source_root": str(source),
        "target_root": str(target),
        "fraction": fraction,
        "seed": seed,
        "source_images": len(data["images"]),
        "selected_images": len(selected_images),
        "selected_annotations": len(selected_annotations),
        "categories": len(category_ids),
        "represented_categories": len(represented),
        "missing_categories": sorted(category_ids - represented),
    }
    (target / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-links", action="store_true", help="Used by portable fixture tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_subset(
        args.source_root,
        args.target_root,
        args.fraction,
        args.seed,
        link_images=not args.no_links,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
