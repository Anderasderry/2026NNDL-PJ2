#!/usr/bin/env bash
# Task 2: VGG-A vs VGG-A+BN comparison, loss/gradient landscape, and report figures.
#
# Usage (from project root):
#   bash run_task2.sh
#   EPOCHS=2 N_ITEMS=1024 bash run_task2.sh    # quick debug
#   SKIP_TRAIN=1 bash run_task2.sh             # only replot from existing JSON
#
# Recommended in tmux:
#   tmux new -s pj2-task2 -d "cd /path/to/PJ2 && bash run_task2.sh"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VGG_DIR="$ROOT/codes/VGG_BatchNorm"
OUTPUT_ROOT="$ROOT/outputs/VGG_BatchNorm"
mkdir -p logs "$OUTPUT_ROOT/figures" "$OUTPUT_ROOT/models"

TS="$(date +%Y%m%d_%H%M)"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
N_ITEMS="${N_ITEMS:--1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "Project root     : $ROOT"
echo "Log timestamp    : $TS"
echo "Num workers      : $NUM_WORKERS"
echo "Epochs           : $EPOCHS"
echo "Batch size       : $BATCH_SIZE"
echo "Comparison lr    : $LR"
echo "N items          : $N_ITEMS"
echo "Skip train       : $SKIP_TRAIN"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "========================================"
  echo "Step 1/2: Task 2 training + landscape"
  echo "========================================"

  TRAIN_ARGS=(
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --learning-rates 1e-3 2e-3 1e-4 5e-4
    --plot-stride 50
    --fill-alpha 0.15
  )
  if [[ "$N_ITEMS" != "-1" ]]; then
    TRAIN_ARGS+=(--n-items "$N_ITEMS")
  fi

  (
    cd "$VGG_DIR"
    python VGG_Loss_Landscape.py "${TRAIN_ARGS[@]}"
  ) 2>&1 | tee "$ROOT/logs/task2_train_${TS}.log"
else
  echo "========================================"
  echo "Step 1/2: skipped (SKIP_TRAIN=1)"
  echo "========================================"
fi

echo "========================================"
echo "Step 2/2: Task 2 visualization (report figures)"
echo "========================================"

(
  cd "$VGG_DIR"
  python VGG_Loss_Landscape.py --replot-comparison \
    --loss-ylim 0 2.5 \
    --predictiveness-ylim 0 1 \
    --plot-stride 50 \
    --fill-alpha 0.15
) 2>&1 | tee "$ROOT/logs/task2_replot_${TS}.log"

echo "========================================"
echo "TASK 2 DONE"
echo "Outputs          : $OUTPUT_ROOT/"
echo "Figures          : $OUTPUT_ROOT/figures/"
echo "Models           : $OUTPUT_ROOT/models/"
echo "Logs             : $ROOT/logs/task2_*_${TS}.log"
echo "========================================"
