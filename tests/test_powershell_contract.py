import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PowerShellContractTest(unittest.TestCase):
    def test_entry_point_contains_required_reproduction_steps(self):
        text = (ROOT / "reproduce.ps1").read_text(encoding="utf-8")
        for required in (
            "CocoPath",
            "OutputDir",
            "CheckpointPath",
            "QuickCheck",
            "SkipInstall",
            "Visualize",
            "PythonExecutable",
            ".venv-repro",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "cu124",
            "verify_dataset.py",
            "download_checkpoint.py",
            "record_manifest.py",
            "verify_metrics.py",
            "main.py",
            "$LASTEXITCODE",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_visualization_uses_explicit_paths(self):
        text = (ROOT / "scripts" / "visualize_predictions.py").read_text(encoding="utf-8")
        for required in ("--coco-path", "--checkpoint", "--output-dir", "--threshold"):
            self.assertIn(required, text)
        self.assertNotIn("D:\\\\Users", text)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is unavailable")
    def test_entry_point_parses_as_powershell(self):
        script = ROOT / "reproduce.ps1"
        command = (
            "$errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$null,[ref]$errors) | Out-Null; "
            "if($errors.Count -gt 0){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
