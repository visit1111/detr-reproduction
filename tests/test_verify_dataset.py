import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.verify_dataset import verify_split


FIXTURE = Path(__file__).parent / "fixtures" / "mini_coco"


class VerifyDatasetTest(unittest.TestCase):
    def test_valid_split_reports_counts(self):
        summary = verify_split(FIXTURE, "val", expected_images=1, check_files=True)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["annotations"], 1)
        self.assertEqual(summary["categories"], 2)

    def test_rejects_expected_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Expected 2 images"):
            verify_split(FIXTURE, "val", expected_images=2, check_files=False)

    def test_rejects_missing_referenced_image_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "coco"
            shutil.copytree(FIXTURE, root)
            (root / "val2017" / "000000000005.jpg").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "000000000005.jpg"):
                verify_split(root, "val", expected_images=1, check_files=True)

    def test_rejects_duplicate_image_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "coco"
            shutil.copytree(FIXTURE, root)
            annotation = root / "annotations" / "instances_val2017.json"
            data = json.loads(annotation.read_text(encoding="utf-8"))
            data["images"].append(dict(data["images"][0]))
            annotation.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate image IDs"):
                verify_split(root, "val", expected_images=None, check_files=False)


if __name__ == "__main__":
    unittest.main()
