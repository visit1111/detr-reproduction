#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash reproduce.sh --coco-path PATH [options]

Required:
  --coco-path PATH       Complete COCO 2017 root.

Options:
  --output-dir PATH      Training output directory.
  --subset-root PATH     Deterministic 10% dataset view.
  --python PATH          Existing Python interpreter to use.
  --resume PATH          Resume from an existing checkpoint.pth.
  --quick-check          Validate environment, data, config, and tests only.
  --smoke                Run one epoch instead of the configured 100 epochs.
  --skip-install         Do not create a venv or install dependencies.
  -h, --help             Show this help.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/configs/detection_10pct_100ep.json"
VENV_DIR="$SCRIPT_DIR/.venv-repro"
COCO_PATH=""
OUTPUT_DIR="$SCRIPT_DIR/artifacts/runs/linux-train-10pct-100ep"
SUBSET_ROOT="$SCRIPT_DIR/artifacts/data/coco2017_train10_seed42"
PYTHON_OVERRIDE=""
RESUME_PATH=""
QUICK_CHECK=0
SMOKE=0
SKIP_INSTALL=0

while (($#)); do
  case "$1" in
    --coco-path) COCO_PATH="${2:?--coco-path requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --subset-root) SUBSET_ROOT="${2:?--subset-root requires a value}"; shift 2 ;;
    --python) PYTHON_OVERRIDE="${2:?--python requires a value}"; shift 2 ;;
    --resume) RESUME_PATH="${2:?--resume requires a value}"; shift 2 ;;
    --quick-check) QUICK_CHECK=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$COCO_PATH" ]]; then
  echo "--coco-path is required." >&2
  usage >&2
  exit 2
fi

cd "$SCRIPT_DIR"

if [[ -n "$PYTHON_OVERRIDE" ]]; then
  PYTHON_BIN="$PYTHON_OVERRIDE"
else
  PYTHON_BIN="$VENV_DIR/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    if ((SKIP_INSTALL)); then
      echo "Missing $VENV_DIR. Remove --skip-install or pass --python." >&2
      exit 1
    fi
    python3 -m venv "$VENV_DIR"
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ((!SKIP_INSTALL)); then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements-repro.txt"
fi

read_config() {
  "$PYTHON_BIN" -c \
    'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8")); [value := value[key] for key in sys.argv[2].split(".")]; print(str(value).lower() if isinstance(value, bool) else value)' \
    "$CONFIG_PATH" "$1"
}

SOURCE_TRAIN_IMAGES="$(read_config dataset.expected_source_train_images)"
SUBSET_TRAIN_IMAGES="$(read_config dataset.expected_subset_train_images)"
VAL_IMAGES="$(read_config dataset.expected_val_images)"
FRACTION="$(read_config dataset.fraction)"
SEED="$(read_config dataset.seed)"
EPOCHS="$(read_config training.epochs)"
LR_DROP="$(read_config training.lr_drop)"
BATCH_SIZE="$(read_config training.batch_size_per_gpu)"
WORKERS="$(read_config training.num_workers_per_rank)"
GPU_COUNT="$(read_config training.gpu_count)"

"$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_dataset.py" \
  --coco-root "$COCO_PATH" --split train --expected-images "$SOURCE_TRAIN_IMAGES"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_dataset.py" \
  --coco-root "$COCO_PATH" --split val --expected-images "$VAL_IMAGES"

"$PYTHON_BIN" -m unittest discover -s "$SCRIPT_DIR/tests" -v

if ((QUICK_CHECK)); then
  "$PYTHON_BIN" -c \
    'import sys, torch; print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | visible GPUs: {torch.cuda.device_count()}")'
  echo "Quick check passed: source=$SOURCE_TRAIN_IMAGES, subset=$SUBSET_TRAIN_IMAGES, val=$VAL_IMAGES, fraction=$FRACTION, seed=$SEED."
  exit 0
fi

SUBSET_ANNOTATION="$SUBSET_ROOT/annotations/instances_train2017.json"
if [[ -e "$SUBSET_ANNOTATION" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_dataset.py" \
    --coco-root "$SUBSET_ROOT" --split train --expected-images "$SUBSET_TRAIN_IMAGES" --check-files
else
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/create_coco_subset.py" \
    --source-root "$COCO_PATH" --target-root "$SUBSET_ROOT" \
    --fraction "$FRACTION" --seed "$SEED"
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_dataset.py" \
    --coco-root "$SUBSET_ROOT" --split train --expected-images "$SUBSET_TRAIN_IMAGES" --check-files
fi

if [[ -n "$RESUME_PATH" ]]; then
  if [[ ! -f "$RESUME_PATH" ]]; then
    echo "Resume checkpoint not found: $RESUME_PATH" >&2
    exit 1
  fi
elif [[ -e "$OUTPUT_DIR/checkpoint.pth" || -e "$OUTPUT_DIR/log.txt" || -e "$OUTPUT_DIR/run_manifest.json" ]]; then
  echo "Refusing to overwrite an existing run in $OUTPUT_DIR. Use --resume or choose a new --output-dir." >&2
  exit 1
fi

if ((SMOKE)); then
  EPOCHS=1
  LR_DROP=1
fi

"$PYTHON_BIN" -c \
  'import sys, torch; required=int(sys.argv[1]); count=torch.cuda.device_count(); print(f"visible CUDA GPUs: {count}"); raise SystemExit(0 if torch.cuda.is_available() and count >= required else f"Need {required} CUDA GPUs, found {count}")' \
  "$GPU_COUNT"

PYTHON_DIR="$(dirname -- "$PYTHON_BIN")"
if [[ -x "$PYTHON_DIR/torchrun" ]]; then
  TORCHRUN=("$PYTHON_DIR/torchrun")
else
  # torchrun is equivalent to python -m torch.distributed.run.
  TORCHRUN=("$PYTHON_BIN" -m torch.distributed.run)
fi

TRAIN_COMMAND=(
  "${TORCHRUN[@]}" --standalone --nproc_per_node "$GPU_COUNT"
  "$SCRIPT_DIR/main.py"
  --coco_path "$SUBSET_ROOT"
  --output_dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --lr_drop "$LR_DROP"
  --batch_size "$BATCH_SIZE"
  --num_workers "$WORKERS"
)
if [[ -n "$RESUME_PATH" ]]; then
  TRAIN_COMMAND+=(--resume "$RESUME_PATH")
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/record_manifest.py" \
  --output-dir "$OUTPUT_DIR" --config "$CONFIG_PATH" "${TRAIN_COMMAND[@]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Starting DETR training in the foreground. Output: $OUTPUT_DIR"
exec "${TRAIN_COMMAND[@]}"
