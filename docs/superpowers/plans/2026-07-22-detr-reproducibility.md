# DETR Cross-Platform Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a Git-managed DETR repository with a one-command Windows evaluation workflow and a one-command Linux 10%-COCO training workflow.

**Architecture:** Keep official DETR code and history intact except for a small machine-readable evaluation output hook. Add focused standard-library utilities for data validation, deterministic subsetting, checkpoint verification, metric comparison, and run manifests; compose them through PowerShell and Bash entry points. Store every downloaded or generated artifact under ignored directories and commit only code, configuration, documentation, tests, and small reference metrics.

**Tech Stack:** Python 3.10-3.12, PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4, PowerShell 5+, Bash 4+, Git, COCO 2017.

## Global Constraints

- Preserve official upstream commit `29901c51d7fe8712168b8d0d64351170bc0f83e0` in history.
- Do not commit COCO data, credentials, weights, checkpoints, virtual environments, logs, or generated images.
- Windows evaluation must use official checkpoint SHA-256 `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`.
- Linux training must use seed 42, 10% train2017, 100 epochs, `lr_drop=80`, batch size 4 per GPU, 8 workers per rank, and four ranks by default.
- Full val2017 remains 5,000 images.
- New utilities use only the Python standard library unless model or COCO execution actually requires PyTorch/pycocotools.
- Each implementation task ends with tests and one intentional Git commit.

---

### Task 1: Repository safety and canonical experiment configuration

**Files:**
- Modify: `.gitignore`
- Create: `requirements-repro.txt`
- Create: `configs/detection_10pct_100ep.json`
- Create: `results/reference/official_detr_r50_coco_val2017.json`
- Create: `tests/test_repository_contract.py`

**Interfaces:**
- Consumes: no runtime inputs.
- Produces: ignored artifact boundaries, pinned lightweight dependencies, canonical training parameters, and official metric targets.

- [ ] **Step 1: Write repository contract tests**

  Test that the configuration contains fraction `0.1`, seed `42`, epochs `100`, learning-rate drop `80`, batch size `4`, workers `8`, and GPU count `4`. Test that the reference file contains AP `42.0`, AP50 `62.4`, AP75 `44.2`, checkpoint URL, and exact SHA-256. Test `.gitignore` coverage for `.venv-repro/`, `artifacts/`, `output/`, and model files.

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `python -m unittest tests.test_repository_contract -v`

  Expected: failure because the new files do not exist.

- [ ] **Step 3: Add the minimal configuration files**

  Use JSON for machine-readable configuration and reference results. Pin `scipy==1.15.2`, `pycocotools==2.0.8`, `matplotlib==3.10.0`, and `Pillow==11.1.0`; install torch and torchvision separately from the official CUDA wheel index.

- [ ] **Step 4: Verify and commit**

  Run: `python -m unittest tests.test_repository_contract -v`

  Expected: all repository contract tests pass.

  Commit: `chore: define reproducible experiment contract`

### Task 2: Deterministic COCO subsetting and dataset verification

**Files:**
- Create: `scripts/create_coco_subset.py`
- Create: `scripts/verify_dataset.py`
- Create: `tests/fixtures/mini_coco/annotations/instances_train2017.json`
- Create: `tests/fixtures/mini_coco/annotations/instances_val2017.json`
- Create: `tests/test_create_coco_subset.py`
- Create: `tests/test_verify_dataset.py`

**Interfaces:**
- `create_coco_subset.create_subset(source_root: Path, target_root: Path, fraction: float, seed: int) -> dict`
- `verify_dataset.verify_split(coco_root: Path, split: str, expected_images: int | None, check_files: bool) -> dict`
- CLI summaries are JSON on stdout; failures return nonzero.

- [ ] **Step 1: Write failing fixture-based tests**

  Cover deterministic image selection, annotation filtering, category preservation, refusal to overwrite, invalid fractions, missing referenced images, duplicate image IDs, expected-count mismatch, and successful val verification.

- [ ] **Step 2: Run focused tests and confirm failure**

  Run: `python -m unittest tests.test_create_coco_subset tests.test_verify_dataset -v`

  Expected: import failures because the utilities do not exist.

- [ ] **Step 3: Implement minimal utilities**

  Filter annotations at image level, retain COCO metadata and all category definitions, write annotations atomically, and link image directories plus the unchanged validation annotation. Dataset verification checks schema, unique IDs, annotation references, category references, expected counts, and optionally every image file.

- [ ] **Step 4: Verify and commit**

  Run: `python -m unittest tests.test_create_coco_subset tests.test_verify_dataset -v`

  Expected: all subset and dataset tests pass.

  Commit: `feat: add deterministic COCO data preparation`

### Task 3: Checkpoint, metrics, and run-manifest verification

**Files:**
- Create: `scripts/download_checkpoint.py`
- Create: `scripts/verify_metrics.py`
- Create: `scripts/record_manifest.py`
- Create: `tests/test_download_checkpoint.py`
- Create: `tests/test_verify_metrics.py`
- Create: `tests/test_record_manifest.py`

**Interfaces:**
- `download_checkpoint.download(url: str, destination: Path, sha256: str) -> dict`
- `verify_metrics.verify(actual_path: Path, reference_path: Path, tolerance: float) -> dict`
- `record_manifest.build_manifest(command: list[str], config: dict, output_dir: Path) -> dict`

- [ ] **Step 1: Write failing tests**

  Use local `file://` fixtures for successful downloads, cached downloads, and checksum mismatch quarantine. Verify accepted metrics at ±0.2, rejected metrics outside tolerance, DETR `test_coco_eval_bbox` extraction, Git revision capture, platform metadata, and secret-safe command recording.

- [ ] **Step 2: Run focused tests and confirm failure**

  Run: `python -m unittest tests.test_download_checkpoint tests.test_verify_metrics tests.test_record_manifest -v`

  Expected: import failures because the utilities do not exist.

- [ ] **Step 3: Implement minimal utilities**

  Stream downloads into `.part`, hash before atomic replacement, quarantine invalid existing files, compare AP/AP50/AP75 against reference JSON, and collect Git/Python/platform plus optional CUDA metadata without recording environment variables or credentials.

- [ ] **Step 4: Verify and commit**

  Run: `python -m unittest tests.test_download_checkpoint tests.test_verify_metrics tests.test_record_manifest -v`

  Expected: all artifact and manifest tests pass.

  Commit: `feat: add artifact and result verification`

### Task 4: Machine-readable DETR evaluation output

**Files:**
- Modify: `main.py`
- Create: `tests/test_eval_metrics_output.py`

**Interfaces:**
- `main.write_eval_metrics(output_dir: Path, stats: dict) -> Path`
- Full evaluation writes `eval_metrics.json` containing `test_coco_eval_bbox` and other serializable evaluation statistics.

- [ ] **Step 1: Write the failing output test**

  Exercise the helper with a temporary output directory and a representative COCO metric list. Assert UTF-8 JSON, preserved float values, deterministic indentation, and returned path.

- [ ] **Step 2: Run and confirm failure**

  Run: `python -m unittest tests.test_eval_metrics_output -v`

  Expected: failure because `write_eval_metrics` does not exist.

- [ ] **Step 3: Implement the smallest hook**

  Add one helper near argument parsing. In evaluation-only mode, call it on the main process immediately after `evaluate` returns. Do not alter model construction, data loading, losses, or COCO evaluation.

- [ ] **Step 4: Verify upstream compatibility and commit**

  Run: `python -m unittest tests.test_eval_metrics_output test_all.Tester -v`

  Expected: the new test and all eight official core tests pass.

  Commit: `feat: emit machine-readable evaluation metrics`

### Task 5: One-command Windows evaluation

**Files:**
- Create: `reproduce.ps1`
- Create: `scripts/visualize_predictions.py`
- Create: `tests/test_powershell_contract.py`

**Interfaces:**
- `reproduce.ps1 -CocoPath <path> [-OutputDir <path>] [-QuickCheck] [-SkipInstall] [-Visualize]`
- Exit code 0 means environment/data/checkpoint/tests and, unless quick, metrics verification succeeded.

- [ ] **Step 1: Write failing contract tests**

  Check that the PowerShell script exposes the documented parameters, uses `.venv-repro`, installs exact torch/torchvision versions from the CUDA 12.4 index, invokes dataset/checkpoint/metrics verification, and propagates native command failures.

- [ ] **Step 2: Run and confirm failure**

  Run: `python -m unittest tests.test_powershell_contract -v`

  Expected: failure because `reproduce.ps1` does not exist.

- [ ] **Step 3: Implement the entry point and visualization adapter**

  Use strict PowerShell error handling, explicit argument arrays, verified paths, and `Tee-Object` for logs. Visualization accepts explicit checkpoint, COCO, output, threshold, and sample-count arguments and never hardcodes a user path.

- [ ] **Step 4: Parse, quick-check, and commit**

  Run the PowerShell parser, contract tests, and `reproduce.ps1 -CocoPath D:\Users\liuhuayong\Desktop\task\data\coco2017 -QuickCheck -SkipInstall` with the existing environment override.

  Expected: parser succeeds and quick check reports valid val2017 plus a verified or downloadable checkpoint plan without starting the 5,000-image evaluation.

  Commit: `feat: add one-command Windows evaluation`

### Task 6: One-command Linux training

**Files:**
- Create: `reproduce.sh`
- Create: `tests/test_bash_contract.py`

**Interfaces:**
- `reproduce.sh --coco-path <path> [--output-dir <path>] [--quick-check] [--smoke] [--skip-install]`
- Normal mode launches canonical 4-rank, 100-epoch training in the foreground; smoke mode runs one bounded epoch; quick mode performs no GPU training.

- [ ] **Step 1: Write failing contract tests**

  Check strict Bash mode, robust option parsing, canonical config use, refusal to overwrite, dataset verification, subset generation, manifest recording, explicit CUDA device count, torchrun arguments, and propagated exit status.

- [ ] **Step 2: Run and confirm failure**

  Run: `python -m unittest tests.test_bash_contract -v`

  Expected: failure because `reproduce.sh` does not exist.

- [ ] **Step 3: Implement the Linux entry point**

  Resolve repository paths from the script location, use JSON config through Python rather than duplicated constants, create only ignored artifacts, run quick checks without CUDA, and keep the long training process in the foreground for reliable status propagation.

- [ ] **Step 4: Verify and commit**

  Run: `bash -n reproduce.sh`, contract tests, and a server-side `--quick-check --skip-install` using `/data/lhy/data/coco2017`.

  Expected: syntax and contract tests pass; server check validates 118,287 train images, 5,000 val images, 80 categories, and deterministic 11,829-image subset parameters without launching another training job.

  Commit: `feat: add one-command Linux training`

### Task 7: Documentation, CI, and complete verification

**Files:**
- Create: `REPRODUCE.md`
- Create: `.github/workflows/repro-ci.yml`
- Modify: `README.md`
- Create: `tests/test_documentation_contract.py`

**Interfaces:**
- `REPRODUCE.md` is the authoritative user guide.
- CI runs standard-library unit/contract tests and syntax checks without downloading COCO or GPU packages.

- [ ] **Step 1: Write documentation contract tests**

  Require Windows and Linux commands, prerequisites, expected metrics, expected durations, output files, resume instructions, overfitting interpretation, troubleshooting, data/weight exclusions, and Git provenance.

- [ ] **Step 2: Run and confirm failure**

  Run: `python -m unittest tests.test_documentation_contract -v`

  Expected: failure because the reproducibility guide does not exist.

- [ ] **Step 3: Write the guide, README entry, and CI workflow**

  Keep the official README intact below a short project notice. Document exact tested commands and distinguish quick verification from full experiments.

- [ ] **Step 4: Run the complete verification suite**

  Run:

  ```text
  python -m unittest discover -s tests -v
  python -m unittest test_all.Tester -v
  python -m py_compile scripts/*.py
  bash -n reproduce.sh
  PowerShell parser check for reproduce.ps1
  git diff --check
  git status --short
  ```

  Expected: all tests and syntax checks pass, no whitespace errors, and only intended tracked changes remain before the final commit.

- [ ] **Step 5: Commit the verified documentation**

  Commit: `docs: document and verify DETR reproduction`

  After committing, rerun the complete suite and confirm `git status --short` is empty.
