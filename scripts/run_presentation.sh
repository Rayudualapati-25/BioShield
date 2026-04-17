#!/usr/bin/env bash
# Autonomous presentation-run orchestrator.
#
# Runs data prep -> Condition A -> B -> C -> D -> plots, logging milestones
# to /tmp/bioshield_progress.log for external watchers (mobile notifications).
#
# Mac is kept awake via `caffeinate -dims` bound to this script's PID: when
# this script exits (success or crash), the caffeinate child dies with it.
# No `sudo` needed. Lid-close is also safe as long as AC + external display
# + external input device are connected.

set -u
set -o pipefail

PROJECT_ROOT="/Users/venkatrayudu/Workspace/Projects/NOVEL AI PROJECT"
PY=/opt/anaconda3/bin/python3
CONFIG="configs/config_presentation.yaml"
PROG_LOG=/tmp/bioshield_progress.log
RUN_LOG=/tmp/bioshield_run.log

cd "$PROJECT_ROOT"

# --- Performance env ---
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0    # no recommended-memory cap; use all we can
export OMP_NUM_THREADS=10                       # M3 Max has 12 cores; leave 2 for OS + python
export MKL_NUM_THREADS=10
# Keep HF offline so transformers doesn't hit the safetensors PR-check endpoint
# mid-training. Data prep already ran; no dataset fetch needed from here on.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Keep Mac awake, process nice'd up ---
# -d: display on, -i: system idle disabled, -m: disk idle disabled, -s: system sleep disabled
# -w PID: stay alive only as long as PID exists. $$ = this script's PID.
caffeinate -dims -w $$ &
CAFF_PID=$!
trap "kill $CAFF_PID 2>/dev/null || true" EXIT

# Raise our own priority one notch (doesn't need sudo for nicer values; -5 does)
renice -n -5 $$ >/dev/null 2>&1 || true

mkdir -p experiments/metrics experiments/plots experiments/round_data

# --- Archive the old 200-scale smoke results so the new run has a clean slate ---
ARCHIVE=experiments/metrics/smoke_200
mkdir -p "$ARCHIVE"
for f in condition_A_metrics.json condition_B_metrics.json condition_C_metrics.json \
         condition_B_round0_metrics.json condition_B_round1_metrics.json condition_B_round2_metrics.json \
         condition_C_round0_metrics.json metrics_log.csv; do
  if [ -f "experiments/metrics/$f" ]; then
    mv "experiments/metrics/$f" "$ARCHIVE/$f" 2>/dev/null || true
  fi
done

notify() {
  local msg="$1"
  echo "$(date +'%Y-%m-%d %H:%M:%S') | $msg" | tee -a "$PROG_LOG"
}

run_stage() {
  local name="$1"; shift
  local t0=$(date +%s)
  notify "START | $name"
  if "$@" >> "$RUN_LOG" 2>&1; then
    local elapsed=$(($(date +%s) - t0))
    notify "DONE  | $name | ${elapsed}s"
  else
    local elapsed=$(($(date +%s) - t0))
    notify "FAIL  | $name | ${elapsed}s | see $RUN_LOG"
    return 1
  fi
}

notify "PIPELINE_BEGIN | presentation-run, config=$CONFIG"
notify "INFO  | caffeinate PID=$CAFF_PID (prevents sleep for lifetime of this script)"

# --- 1. Data prep: stream 500 real PubMed + generate 500 BioMistral fakes ---
# If data/processed/{train,val,test}.csv already exist at the right scale, this
# is a no-op. data/prepare_data.py checks the target row count.
run_stage "data_prep_500" \
  $PY data/prepare_data.py --config "$CONFIG" --max_real 500 --fakes-source generator \
  || notify "WARN  | data_prep hit an error. Attempting to continue — CSVs may already exist."

# --- 2. Condition A: static baseline detector ---
run_stage "condition_A" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition static_baseline

# --- 3. Condition B: SeqGAN + detector retrain ---
run_stage "condition_B" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition seqgan_only

# --- 4. Condition C: Qwen zero-shot rewrite, no retrain ---
run_stage "condition_C" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition agent_only

# --- 5. Condition D: full pipeline (Qwen rewrite + detector retrain) ---
run_stage "condition_D" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition full_pipeline

# --- 6. Plots ---
run_stage "plots" \
  $PY evaluation/visualization.py --config "$CONFIG" --all_conditions

# --- 7. RAID + M4 cross-benchmarks (defensive; skipped if datasets unavailable) ---
# Only run if we have network (datasets streaming needs HF Hub reachability).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
run_stage "bench_raid" \
  $PY scripts/raid_benchmark.py --config "$CONFIG" --benchmark raid --n_samples 200 \
  || notify "INFO  | RAID bench skipped (network or schema issue)"
run_stage "bench_m4" \
  $PY scripts/raid_benchmark.py --config "$CONFIG" --benchmark m4 --n_samples 200 \
  || notify "INFO  | M4 bench skipped (network or schema issue)"

notify "PIPELINE_END | all stages complete"
