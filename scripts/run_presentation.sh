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
# NOTE: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set only around training stages
# (via the `go_offline` / `go_online` helpers below). Data prep and benchmarks
# need the hub reachable to stream datasets.
go_offline() { export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; }
go_online()  { unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; }
go_online

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

# --- 1. Data prep: stream 1500 real PubMed + generate 1500 BioMistral fakes ---
# Needs HF hub reachable to stream PubMed abstracts. load_generator_fakes()
# flips HF_HUB_OFFLINE internally for the BioMistral load, so global online is fine.
go_online
run_stage "data_prep_1500" \
  $PY data/prepare_data.py --config "$CONFIG" --max_real 1500 --fakes-source generator \
  || notify "WARN  | data_prep hit an error. Attempting to continue — CSVs may already exist."

# --- 2-5. Training stages: offline-safe (models already in HF cache). ---
go_offline

run_stage "condition_A" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition static_baseline

run_stage "condition_B" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition seqgan_only

run_stage "condition_C" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition agent_only

run_stage "condition_D" \
  $PY training/adversarial_loop.py --config "$CONFIG" --condition full_pipeline

# --- 6. Plots (local only, offline fine) ---
run_stage "plots" \
  $PY evaluation/visualization.py --config "$CONFIG" --all_conditions

# --- 7. RAID + M4 cross-benchmarks: needs network to stream datasets ---
go_online
run_stage "bench_raid" \
  $PY scripts/raid_benchmark.py --config "$CONFIG" --benchmark raid --n_samples 200 \
  || notify "INFO  | RAID bench skipped (network or schema issue)"
run_stage "bench_m4" \
  $PY scripts/raid_benchmark.py --config "$CONFIG" --benchmark m4 --n_samples 200 \
  || notify "INFO  | M4 bench skipped (network or schema issue)"

notify "PIPELINE_END | all stages complete"
