#!/usr/bin/env python3
"""Validate a COCO detection split without importing pycocotools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _unique_ids(items: list[dict[str, Any]], label: str) -> set[int]:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label} IDs")
    return set(ids)


def verify_split(
    coco_root: Path,
    split: str,
    expected_images: int | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    root = Path(coco_root).resolve()
    year_split = f"{split}2017"
    annotation_path = root / "annotations" / f"instances_{year_split}.json"
    image_dir = root / year_split
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"Missing or invalid COCO field: {key}")

    image_ids = _unique_ids(data["images"], "image")
    annotation_ids = _unique_ids(data["annotations"], "annotation")
    category_ids = _unique_ids(data["categories"], "category")
    del annotation_ids

    if expected_images is not None and len(image_ids) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(image_ids)}")

    for annotation in data["annotations"]:
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"Annotation references unknown image ID {annotation['image_id']}")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Annotation references unknown category ID {annotation['category_id']}")

    if check_files:
        for image in data["images"]:
            image_path = image_dir / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

    return {
        "root": str(root),
        "split": split,
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
        "files_checked": check_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify_split(args.coco_root, args.split, args.expected_images, args.check_files)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
