#!/usr/bin/env bash
# One-shot runtime-efficiency sweep for all 8 methods.
#
#   cd <repo root> && bash bench_efficiency/run_all.sh
#
# Each method is benched ONCE. Throughput does not depend on the protocol, so
# few-shot / full-shot / zero-shot numbers are derived in aggregate.py by
# multiplying the measured throughput by the counted workload.
#
# Expect ~40-70 min on a single modern GPU, dominated by model downloads on the
# first run. Every step is independent -- re-run just the ones that failed.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCH=bench_efficiency
RESULTS="$BENCH/results"
mkdir -p "$RESULTS/logs"

DATA_ROOT="${DATA_ROOT:-hf_cache}"
# Two test folders of clearly different size, for Time-LLM / CALF slope timing.
TEST_SMALL="${TEST_SMALL:-$DATA_ROOT/test/D1NAMO}"
TEST_LARGE="${TEST_LARGE:-$DATA_ROOT/test/HUPA-UCM}"

step() {
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> $name"
  echo "=================================================================="
  if "$@" 2>&1 | tee "$RESULTS/logs/${name}.log"; then
    echo "--- $name OK"
  else
    echo "!!! $name FAILED (see $RESULTS/logs/${name}.log); continuing"
  fi
}

# ---------------------------------------------------------------------------
# 0. Data. The per-method hf_cache copies are stale/duplicated; build one
#    canonical cache at the repo root and point everything at it.
# ---------------------------------------------------------------------------
if [ ! -d "$DATA_ROOT/test" ]; then
  step prepare_dataset python prepare_dataset.py \
    --hf_name byluuu/gluco-tsfm-benchmark \
    --output_dir "$DATA_ROOT" \
    --create_mixed
fi

# ---------------------------------------------------------------------------
# 1. Workload: how many windows each protocol actually processes (CPU, ~1 min)
# ---------------------------------------------------------------------------
step count_windows python "$BENCH/count_windows.py" \
  --data-root "$DATA_ROOT" \
  --context-steps 144 --horizon-steps 6 \
  --strides 1 10 12 240 \
  --out "$RESULTS/workload.json"

# Window counts for the two Time-LLM / CALF test folders, read back out.
read -r W_SMALL W_LARGE <<<"$(python - "$RESULTS/workload.json" "$(basename "$TEST_SMALL")" "$(basename "$TEST_LARGE")" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
per = payload["conventions"]["multiwindow"]["test"]["per_dataset"]
print(per.get(sys.argv[2], {}).get("stride_1", 0), per.get(sys.argv[3], {}).get("stride_1", 0))
PY
)"
echo "test-window counts: small=$W_SMALL large=$W_LARGE"

# ---------------------------------------------------------------------------
# 2. LSTM (TensorFlow). Needs Keras 2: pip install tf_keras
# ---------------------------------------------------------------------------
step lstm_throughput python "$BENCH/bench_lstm.py" --mode throughput

# Real few-shot run to early stopping -- the only way to know the LSTM's true
# epoch count. Few-shot data is small (stride 240), so this is affordable.
step lstm_fewshot_real python "$BENCH/bench_lstm.py" \
  --mode fewshot-real \
  --train-csv-dir "$DATA_ROOT/train/mixed" \
  --train-stride 240

# ---------------------------------------------------------------------------
# 3. The four methods on the shared multiwindow harness
# ---------------------------------------------------------------------------
for M in gpformer timesfm timer moirai; do
  step "${M}_train" python "$BENCH/bench_multiwindow.py" \
    --method "$M" --data-root "$DATA_ROOT" --steps 20 60
done

# Zero-shot inference (no training loop to time)
for M in timesfm timer moirai; do
  step "${M}_zeroshot" python "$BENCH/bench_multiwindow.py" \
    --method "$M" --zeroshot --data-root "$DATA_ROOT"
done

# ---------------------------------------------------------------------------
# 4. Chronos-2
# ---------------------------------------------------------------------------
step chronos_zeroshot python "$BENCH/bench_chronos.py" --protocol zeroshot
step chronos_fullshot python "$BENCH/bench_chronos.py" --protocol fullshot --steps 20 60

# ---------------------------------------------------------------------------
# 5. Time-LLM and CALF
# ---------------------------------------------------------------------------
for M in timellm calf; do
  step "${M}" python "$BENCH/bench_informer.py" \
    --method "$M" \
    --train-root "$DATA_ROOT/train/mixed" \
    --test-small "$TEST_SMALL" --test-large "$TEST_LARGE" \
    --test-windows "$W_SMALL" "$W_LARGE"
done

# ---------------------------------------------------------------------------
# 6. Final table
# ---------------------------------------------------------------------------
step aggregate python "$BENCH/aggregate.py"

echo ""
echo "Done. Table: $RESULTS/efficiency_table.md"
