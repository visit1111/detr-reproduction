import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_coco_subset import create_subset


FIXTURE = Path(__file__).parent / "fixtures" / "mini_coco"


class CreateCocoSubsetTest(unittest.TestCase):
    def test_selection_is_deterministic_and_filters_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            first_summary = create_subset(FIXTURE, first, 0.5, 42, link_images=False)
            second_summary = create_subset(FIXTURE, second, 0.5, 42, link_images=False)

            first_data = json.loads(
                (first / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
            )
            second_data = json.loads(
                (second / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
            )
            first_ids = [image["id"] for image in first_data["images"]]
            second_ids = [image["id"] for image in second_data["images"]]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(first_summary["selected_images"], 2)
            self.assertEqual(first_summary["selected_annotations"], 2)
            self.assertEqual({ann["image_id"] for ann in first_data["annotations"]}, set(first_ids))
            self.assertEqual(len(first_data["categories"]), 2)

    def test_refuses_to_overwrite_existing_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "subset"
            create_subset(FIXTURE, target, 0.5, 42, link_images=False)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                create_subset(FIXTURE, target, 0.5, 42, link_images=False)

    def test_rejects_invalid_fraction(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                create_subset(FIXTURE, Path(temp) / "subset", 1.0, 42, link_images=False)


if __name__ == "__main__":
    unittest.main()
