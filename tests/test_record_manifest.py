import tempfile
import unittest
from pathlib import Path

from scripts.record_manifest import build_manifest, write_manifest


class RecordManifestTest(unittest.TestCase):
    def test_records_revision_and_redacts_sensitive_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            manifest = build_manifest(
                ["python", "main.py", "--token", "secret-value", "--epochs", "1"],
                {"epochs": 1},
                output,
            )
            self.assertEqual(
                manifest["command"],
                ["python", "main.py", "--token", "<redacted>", "--epochs", "1"],
            )
            self.assertEqual(len(manifest["git"]["commit"]), 40)
            self.assertIn("python", manifest)
            self.assertNotIn("environment", manifest)

    def test_writes_manifest_as_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            manifest = build_manifest(["python", "main.py"], {"epochs": 1}, output)
            path = write_manifest(output, manifest)
            self.assertEqual(path, output / "run_manifest.json")
            self.assertIn('"epochs": 1', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
