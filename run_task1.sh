set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CIFAR10_DIR="$ROOT/codes/CIFAR10"
OUTPUT_ROOT="$ROOT/outputs/CIFAR10"
mkdir -p logs "$OUTPUT_ROOT"

TS="$(date +%Y%m%d_%H%M)"
NUM_WORKERS="${NUM_WORKERS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GRID_SEARCH_EPOCHS="${GRID_SEARCH_EPOCHS:-50}"

# Grid search shared baseline (control variables; one dimension changes per run).
COMMON=(
  --epochs "$GRID_SEARCH_EPOCHS"
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

run_grid_search() {
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
echo "Grid search ep.  : $GRID_SEARCH_EPOCHS"
echo "========================================"
echo "Step 1/2: Task 1 grid search (9 runs)"
echo "========================================"

run_grid_search cifarnet
run_grid_search width32 --width 32
run_grid_search width96 --width 96
run_grid_search loss_label_smooth --loss label_smooth
run_grid_search wd_1e4 --weight-decay 1e-4
run_grid_search wd_1e3 --weight-decay 1e-3
run_grid_search act_gelu --activation gelu
run_grid_search act_leaky_relu --activation leaky_relu
run_grid_search optim_adamw --optimizer adamw --lr 3e-4

echo "========================================"
echo "Step 2/2: Writing experiments_report.json"
echo "========================================"

python - <<'PY'
import json
import os
import time

root = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(root, "outputs", "CIFAR10")

experiments = [
    ("cifarnet", "baseline", "Main model: CIFARNet baseline (width=64, ReLU, CE, SGD)"),
    ("width32", "width", "grid search (a): base width=32"),
    ("width96", "width", "grid search (a): base width=96"),
    ("loss_label_smooth", "loss", "grid search (b): label smoothing (0.1)"),
    ("wd_1e4", "loss", "grid search (b): lighter L2 regularization (weight_decay=1e-4)"),
    ("wd_1e3", "loss", "grid search (b): stronger L2 regularization (weight_decay=1e-3)"),
    ("act_gelu", "activation", "grid search (c): GELU activation"),
    ("act_leaky_relu", "activation", "grid search (c): LeakyReLU activation"),
    ("optim_adamw", "optimizer", "grid search: AdamW optimizer (lr=3e-4)"),
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
echo "TASK 1 GRID SEARCH DONE"
echo "Results          : $OUTPUT_ROOT/"
echo "Report           : $OUTPUT_ROOT/experiments_report.json"
echo "Logs             : $ROOT/logs/task1_*_${TS}.log"
echo ""
echo "Next: pick the best config from the report, then run 200-epoch training, e.g.:"
echo "  cd codes/CIFAR10 && python train.py --epochs 200 --width 96 --run-name cifarnet_final --num-workers $NUM_WORKERS"
echo "========================================"
