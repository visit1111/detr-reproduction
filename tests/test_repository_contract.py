import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    def test_training_config_is_canonical(self):
        config = json.loads(
            (ROOT / "configs" / "detection_10pct_100ep.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["dataset"]["fraction"], 0.1)
        self.assertEqual(config["dataset"]["seed"], 42)
        self.assertEqual(config["training"]["epochs"], 100)
        self.assertEqual(config["training"]["lr_drop"], 80)
        self.assertEqual(config["training"]["batch_size_per_gpu"], 4)
        self.assertEqual(config["training"]["num_workers_per_rank"], 8)
        self.assertEqual(config["training"]["gpu_count"], 4)

    def test_reference_metrics_and_checkpoint_are_exact(self):
        reference = json.loads(
            (ROOT / "results" / "reference" / "official_detr_r50_coco_val2017.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(reference["metrics"], {"AP": 42.0, "AP50": 62.4, "AP75": 44.2})
        self.assertEqual(
            reference["checkpoint"]["sha256"],
            "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc",
        )
        self.assertEqual(
            reference["checkpoint"]["url"],
            "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
        )

    def test_generated_artifacts_are_ignored(self):
        patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        for expected in (".venv-repro/", "artifacts/", "output/", "*.pth", "*.pt"):
            with self.subTest(expected=expected):
                self.assertIn(expected, patterns)


if __name__ == "__main__":
    unittest.main()
