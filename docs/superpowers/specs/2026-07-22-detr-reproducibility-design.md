# DETR Cross-Platform Reproducibility Design

## Context

The project must be managed with Git, runnable by another person with a single entry command, and produce results that can be independently verified. The existing local `detr-main` directory was downloaded without Git metadata. The reproducible repository therefore starts from the complete official `facebookresearch/detr` Git history at commit `29901c51d7fe8712168b8d0d64351170bc0f83e0` and adds all project-specific work on a separate branch.

The two supported execution environments are:

- Windows 11 with PowerShell, one NVIDIA RTX 4060 Laptop GPU, and a local COCO val2017 dataset for official DETR-R50 evaluation.
- Linux with Bash, four NVIDIA RTX 4090 GPUs, and a complete COCO 2017 dataset for deterministic 10% training.

## Goals

1. Preserve official DETR provenance and make every project-specific change visible in Git history.
2. Provide one PowerShell command for Windows evaluation and one Bash command for Linux training.
3. Pin the tested Python and PyTorch dependency versions.
4. Verify the official checkpoint by SHA-256 before loading it.
5. Generate the 10% COCO training subset deterministically with seed 42.
6. Record the Git revision, command-line arguments, environment, dataset summary, and metrics for each run.
7. Provide fast tests that do not require a GPU or the COCO dataset, plus optional end-to-end verification with real data.
8. Keep datasets, downloaded weights, logs, virtual environments, and model checkpoints outside Git history.

## Non-Goals

- Automatically redistributing the COCO dataset.
- Committing official or trained `.pth` model files.
- Reproducing the paper's 300/500-epoch full-data training schedule in the quick verification path.
- Hiding upstream deprecation warnings by making unrelated changes to the DETR architecture.

## Repository Strategy

The repository keeps `origin` pointed at `https://github.com/facebookresearch/detr.git`. Project work is developed on `feat/reproducible-workflows`. When a student-owned GitHub or Gitee repository is available, it is added as a second remote instead of replacing the upstream reference.

Commits are split by responsibility:

1. Reproducibility design and implementation plan.
2. Deterministic data, checkpoint, environment, and manifest utilities with tests.
3. Windows and Linux one-command entry points.
4. Documentation, reference metrics, and final verification evidence.

## User-Facing Entry Points

### Windows evaluation

```powershell
.\reproduce.ps1 -CocoPath D:\datasets\coco2017
```

The script will:

1. Locate Python 3.10-3.12 and create `.venv-repro` when needed.
2. Install PyTorch 2.6.0, torchvision 0.21.0, and the pinned lightweight requirements.
3. Validate that val2017 contains 5,000 images and that `instances_val2017.json` is present.
4. Download `detr-r50-e632da11.pth` into `artifacts/weights` and verify SHA-256 `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`.
5. Evaluate DETR-R50 on COCO val2017.
6. Write machine-readable metrics and a run manifest beneath `artifacts/runs`.
7. Verify that box AP, AP50, and AP75 are within the documented tolerance of 42.0, 62.4, and 44.2.
8. Generate prediction, ground-truth comparison, and metric chart images when visualization is enabled.

An optional `-QuickCheck` switch performs environment, data, checkpoint, and unit-test validation without running all 5,000 validation images.

### Linux training

```bash
bash reproduce.sh --coco-path /data/coco2017 --output-dir /data/output/detr-r50-10pct-100ep
```

The script will:

1. Create `.venv-repro` and install the tested CUDA 12.4 dependency set when needed.
2. Validate all COCO train2017 and val2017 directories and annotations.
3. Generate an image-level 10% train2017 subset containing exactly 11,829 images with seed 42 while retaining all 80 categories.
4. Reuse image directories through symbolic links and leave original annotations unchanged.
5. Refuse to overwrite an output directory containing a checkpoint or log.
6. Launch four distributed ranks with 100 epochs, learning-rate drop at epoch 80, batch size 4 per GPU, and 8 workers per rank.
7. Write the exact command, Git revision, package versions, CUDA/GPU details, subset summary, and output paths to `run_manifest.json`.

An optional `--quick-check` mode validates configuration and creates a tiny temporary annotation fixture without launching a long GPU job. An optional `--smoke` mode runs a bounded one-epoch training check when COCO and a CUDA GPU are available.

## File Structure

```text
REPRODUCE.md
reproduce.ps1
reproduce.sh
requirements-repro.txt
configs/
  detection_10pct_100ep.json
scripts/
  create_coco_subset.py
  download_checkpoint.py
  record_manifest.py
  verify_dataset.py
  verify_metrics.py
  visualize_predictions.py
tests/
  fixtures/
  test_create_coco_subset.py
  test_download_checkpoint.py
  test_record_manifest.py
  test_verify_dataset.py
  test_verify_metrics.py
results/reference/
  official_detr_r50_coco_val2017.json
.github/workflows/repro-ci.yml
```

Each Python utility has one responsibility, accepts explicit paths, returns a nonzero exit code on failure, and emits a concise machine-readable summary where appropriate.

## Configuration and Versioning

`configs/detection_10pct_100ep.json` is the canonical experiment configuration. Shell entry points read it rather than duplicating hyperparameters. Command-line overrides are recorded in the run manifest.

The tested runtime is:

- Python 3.10 on Linux and Python 3.12 on Windows.
- PyTorch 2.6.0 with CUDA 12.4 wheels.
- torchvision 0.21.0.
- scipy 1.15.2.
- pycocotools 2.0.8.

PyTorch and torchvision are installed from the official PyTorch CUDA 12.4 wheel index. Remaining packages are installed from `requirements-repro.txt`.

## Data and Artifact Safety

The `.gitignore` extension excludes:

- `.venv-repro/`
- `artifacts/`
- `data/` and COCO image/annotation archives
- `output/`, training logs, and evaluation caches
- `.pth`, `.pt`, `.onnx`, and generated image files
- Python caches and local editor settings

Scripts never delete source data. Dataset subset creation refuses to replace an existing nonmatching path. Output directories with existing run state are rejected unless the user explicitly selects a new directory.

Credentials, SSH paths, server addresses, and passwords are not written to the repository or run manifests.

## Verification Model

Verification occurs at four levels:

1. **Static:** Python syntax checks, Bash syntax checks, PowerShell parser checks, JSON parsing, and Git cleanliness.
2. **Unit:** deterministic subset selection, COCO reference integrity, checksum rejection, manifest contents, and metric tolerance behavior using tiny fixtures.
3. **Official baseline:** the original eight DETR `test_all.Tester` tests must continue to pass.
4. **End-to-end:** Windows evaluation must produce machine-readable COCO metrics near the official DETR-R50 values; Linux smoke training must advance at least one optimizer step without Traceback, OOM, NCCL failure, or NaN.

The reference metric tolerance is absolute ±0.2 AP points. This catches wrong weights, data splits, and evaluation options while allowing harmless floating-point or library-version variation.

## Failure Handling

- Missing data paths report the exact expected file or directory.
- Checksum mismatches quarantine the partial file and stop before model loading.
- Existing outputs cause an explicit refusal rather than silent overwrite.
- Unsupported Python, unavailable CUDA, or insufficient GPU count is reported before a long run starts.
- Child process exit codes are propagated by both platform entry points.
- Every failure leaves the source dataset and prior runs untouched.

## Acceptance Criteria

The work is accepted when:

1. A fresh clone retains the official DETR history and project commits are clearly separated.
2. All fast repository tests pass on Windows without COCO or a GPU training run.
3. `reproduce.ps1 -QuickCheck` validates the current Windows machine and local val2017 data.
4. The full Windows command reproduces AP 42.0, AP50 62.4, and AP75 44.2 within ±0.2.
5. `reproduce.sh --quick-check` validates the Linux environment and deterministic subset logic.
6. Linux smoke training starts the requested distributed ranks and advances without logged fatal errors.
7. Documentation explains prerequisites, commands, expected duration, outputs, metric interpretation, resuming, and common errors.
8. `git status --short` is clean after tracked changes are committed, with no dataset, credential, weight, log, or checkpoint in Git history.
