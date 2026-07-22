import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_metrics import verify


class VerifyMetricsTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_accepts_detr_fraction_metrics_within_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual = self._write(root, "actual.json", {"test_coco_eval_bbox": [0.4205, 0.623, 0.443]})
            reference = self._write(
                root,
                "reference.json",
                {"metrics": {"AP": 42.0, "AP50": 62.4, "AP75": 44.2}, "tolerance": 0.2},
            )
            summary = verify(actual, reference, tolerance=None)
            self.assertTrue(summary["passed"])
            self.assertAlmostEqual(summary["actual"]["AP"], 42.05)

    def test_rejects_metric_outside_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual = self._write(root, "actual.json", {"metrics": {"AP": 41.7, "AP50": 62.4, "AP75": 44.2}})
            reference = self._write(
                root,
                "reference.json",
                {"metrics": {"AP": 42.0, "AP50": 62.4, "AP75": 44.2}, "tolerance": 0.2},
            )
            with self.assertRaisesRegex(ValueError, "AP differs"):
                verify(actual, reference, tolerance=None)


if __name__ == "__main__":
    unittest.main()
