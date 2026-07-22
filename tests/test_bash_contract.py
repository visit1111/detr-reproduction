import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bash_is_usable():
    executable = shutil.which("bash")
    if not executable:
        return False
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0


class BashContractTest(unittest.TestCase):
    def test_entry_point_uses_linux_line_endings(self):
        self.assertNotIn(b"\r\n", (ROOT / "reproduce.sh").read_bytes())
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)

    def test_entry_point_contains_required_training_steps(self):
        text = (ROOT / "reproduce.sh").read_text(encoding="utf-8")
        for required in (
            "set -Eeuo pipefail",
            "--coco-path",
            "--output-dir",
            "--subset-root",
            "--python",
            "--quick-check",
            "--smoke",
            "--skip-install",
            ".venv-repro",
            "detection_10pct_100ep.json",
            "verify_dataset.py",
            "create_coco_subset.py",
            "record_manifest.py",
            "torchrun",
            "--nproc_per_node",
            "checkpoint.pth",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_entry_point_reads_canonical_config(self):
        text = (ROOT / "reproduce.sh").read_text(encoding="utf-8")
        self.assertIn("expected_source_train_images", text)
        self.assertIn("expected_subset_train_images", text)
        self.assertIn("expected_val_images", text)
        self.assertIn("batch_size_per_gpu", text)
        self.assertIn("num_workers_per_rank", text)
        self.assertIn("gpu_count", text)

    @unittest.skipUnless(bash_is_usable(), "Bash is unavailable")
    def test_entry_point_parses_as_bash(self):
        script_text = (ROOT / "reproduce.sh").read_text(encoding="utf-8")
        result = subprocess.run(
            ["bash", "-n"],
            input=script_text.replace("\r\n", "\n").encode("utf-8"),
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
