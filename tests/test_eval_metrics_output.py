import json
import tempfile
import unittest
from pathlib import Path

from main import write_eval_metrics


class EvalMetricsOutputTest(unittest.TestCase):
    def test_writes_serializable_evaluation_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "eval"
            stats = {
                "loss": 1.25,
                "coco_eval_bbox": [0.42, 0.624, 0.442],
            }
            path = write_eval_metrics(output, stats)
            self.assertEqual(path, output / "eval_metrics.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), stats)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
