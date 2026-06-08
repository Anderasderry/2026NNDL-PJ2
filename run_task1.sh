#!/usr/bin/env bash
# Task 1: grid-search ablations, final training, evaluation, and visualization.
#
# Usage (from project root):
#   bash run_task1.sh
#   SKIP_EXISTING=0 bash run_task1.sh          # force re-run all ablations
#   ABLATION_EPOCHS=50 FINAL_EPOCHS=200 bash run_task1.sh
#
# Recommended in tmux:
#   tmux new -s pj2-task1 -d "cd /path/to/PJ2 && bash run_task1.sh"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CIFAR10_DIR="$ROOT/codes/CIFAR10"
OUTPUT_ROOT="$ROOT/outputs/CIFAR10"
mkdir -p logs "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M)"
NUM_WORKERS="${NUM_WORKERS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-50}"
FINAL_EPOCHS="${FINAL_EPOCHS:-200}"
FINAL_WIDTH="${FINAL_WIDTH:-96}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-cifarnet_final}"

# Shared baseline for control-variable ablations.
COMMON=(
  --epochs "$ABLATION_EPOCHS"
  --batch-size 128
  --lr 0.1
  --weight-decay 5e-4
  --width 64
  --dropout 0.5
  --activation relu
  --optimizer sgd
  --loss ce
  --scheduler cosine
  --seed 42
  --num-workers "$NUM_WORKERS"
)

run_ablation() {
  local run_name="$1"
  shift
  local summary="$OUTPUT_ROOT/$run_name/summary.json"

  if [[ "$SKIP_EXISTING" == "1" && -f "$summary" ]]; then
    echo "  -> skipped $run_name (summary.json exists)"
    return 0
  fi

  echo "  -> training $run_name"
  (
    cd "$CIFAR10_DIR"
    python train.py "${COMMON[@]}" "$@" --run-name "$run_name"
  ) 2>&1 | tee "$ROOT/logs/task1_${run_name}_${TS}.log"
}

echo "Project root     : $ROOT"
echo "Log timestamp    : $TS"
echo "Num workers      : $NUM_WORKERS"
echo "Skip existing    : $SKIP_EXISTING"
echo "Ablation epochs  : $ABLATION_EPOCHS"
echo "Final run        : $FINAL_RUN_NAME (width=$FINAL_WIDTH, epochs=$FINAL_EPOCHS)"
echo "========================================"
echo "Step 1/4: Task 1 grid search (9 runs)"
echo "========================================"

run_ablation cifarnet
run_ablation width32 --width 32
run_ablation width96 --width 96
run_ablation loss_label_smooth --loss label_smooth
run_ablation wd_1e4 --weight-decay 1e-4
run_ablation wd_1e3 --weight-decay 1e-3
run_ablation act_gelu --activation gelu
run_ablation act_leaky_relu --activation leaky_relu
run_ablation optim_adamw --optimizer adamw --lr 3e-4

echo "========================================"
echo "Step 2/4: Task 1 final training"
echo "========================================"

FINAL_SUMMARY="$OUTPUT_ROOT/$FINAL_RUN_NAME/summary.json"
if [[ "$SKIP_EXISTING" == "1" && -f "$FINAL_SUMMARY" ]]; then
  echo "  -> skipped $FINAL_RUN_NAME (summary.json exists)"
else
  echo "  -> training $FINAL_RUN_NAME"
  (
    cd "$CIFAR10_DIR"
    python train.py \
      --epochs "$FINAL_EPOCHS" \
      --batch-size 128 \
      --lr 0.1 \
      --weight-decay 5e-4 \
      --width "$FINAL_WIDTH" \
      --dropout 0.5 \
      --activation relu \
      --optimizer sgd \
      --loss ce \
      --scheduler cosine \
      --seed 42 \
      --num-workers "$NUM_WORKERS" \
      --run-name "$FINAL_RUN_NAME"
  ) 2>&1 | tee "$ROOT/logs/task1_${FINAL_RUN_NAME}_${TS}.log"
fi

echo "========================================"
echo "Step 3/4: Task 1 evaluation"
echo "========================================"

(
  cd "$CIFAR10_DIR"
  python evaluate.py --checkpoint "../../outputs/CIFAR10/$FINAL_RUN_NAME/best_model.pt" --num-workers "$NUM_WORKERS"
) 2>&1 | tee "$ROOT/logs/task1_eval_${FINAL_RUN_NAME}_${TS}.log"

echo "========================================"
echo "Step 4/4: Task 1 visualization"
echo "========================================"

(
  cd "$CIFAR10_DIR"
  python visualize.py --run-name "$FINAL_RUN_NAME" --skip-landscape --num-workers "$NUM_WORKERS"
) 2>&1 | tee "$ROOT/logs/task1_vis_${FINAL_RUN_NAME}_${TS}.log"

echo "========================================"
echo "Writing experiments_report.json"
echo "========================================"

python - <<'PY'
import json
import os
import time

root = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(root, "outputs", "CIFAR10")

experiments = [
    ("cifarnet", "baseline", "Main model: CIFARNet baseline (width=64, ReLU, CE, SGD)"),
    ("width32", "width", "Ablation (a): base width=32"),
    ("width96", "width", "Ablation (a): base width=96"),
    ("loss_label_smooth", "loss", "Ablation (b): label smoothing (0.1)"),
    ("wd_1e4", "loss", "Ablation (b): lighter L2 regularization (weight_decay=1e-4)"),
    ("wd_1e3", "loss", "Ablation (b): stronger L2 regularization (weight_decay=1e-3)"),
    ("act_gelu", "activation", "Ablation (c): GELU activation"),
    ("act_leaky_relu", "activation", "Ablation (c): LeakyReLU activation"),
    ("optim_adamw", "optimizer", "Optimizer ablation: AdamW (lr=3e-4)"),
]

rows = []
for run_name, group, description in experiments:
    summary_path = os.path.join(output_root, run_name, "summary.json")
    row = {
        "run_name": run_name,
        "group": group,
        "description": description,
        "status": "done" if os.path.isfile(summary_path) else "missing",
    }
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        row.update({
            "best_test_acc": summary.get("best_test_acc"),
            "best_test_error": summary.get("best_test_error"),
            "parameters": summary.get("parameters"),
            "train_time_sec": summary.get("train_time_sec"),
        })
    rows.append(row)

report = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "group": "all",
    "experiments": rows,
}
report_path = os.path.join(output_root, "experiments_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Report saved: {report_path}")
PY

echo "========================================"
echo "TASK 1 DONE"
echo "Ablations        : $OUTPUT_ROOT/"
echo "Final model      : $OUTPUT_ROOT/$FINAL_RUN_NAME/"
echo "Report           : $OUTPUT_ROOT/experiments_report.json"
echo "Logs             : $ROOT/logs/task1_*_${TS}.log"
echo "========================================"
