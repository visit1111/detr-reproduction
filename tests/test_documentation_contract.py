import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationContractTest(unittest.TestCase):
    def test_reproduction_guide_documents_both_workflows(self):
        text = (ROOT / "REPRODUCE.md").read_text(encoding="utf-8")
        for required in (
            "reproduce.ps1",
            "-QuickCheck",
            "-Visualize",
            "reproduce.sh",
            "--quick-check",
            "--smoke",
            "--resume",
            "11829",
            "seed 42",
            "AP = 42.0",
            "AP50 = 62.4",
            "AP75 = 44.2",
            "run_manifest.json",
            "eval_metrics.json",
            "checkpoint.pth",
            "git clone",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_readme_links_to_reproduction_guide(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("REPRODUCE.md", text)
        self.assertIn("一键复现", text)

    def test_ci_runs_fast_contract_checks(self):
        text = (ROOT / ".github" / "workflows" / "repro-ci.yml").read_text(encoding="utf-8")
        for required in ("pull_request:", "push:", "python-version", "unittest", "bash -n reproduce.sh"):
            with self.subTest(required=required):
                self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
