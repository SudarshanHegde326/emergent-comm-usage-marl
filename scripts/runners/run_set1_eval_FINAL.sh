#!/usr/bin/env bash
# =============================================================================
#  SET 1 — EVAL (FIXED). Runs the offline analyzer on EXISTING checkpoints,
#  strictly 2 at a time (one per GPU), waiting for each pair to finish before
#  starting the next, and only aggregating after ALL are done.
#
#  One analyzer call per checkpoint -> one CSV row with: win rate, MINE,
#  sparsity, message-ablation, and noise/dropout/quantisation(bandwidth) sweeps.
#  Scope: FC/GNN/ATTN, EPO+STANDARD, seeds 0,1,2 = 18 runs. Eval-only (HPC).
# =============================================================================
set -uo pipefail
cd ~/msc_marl
export PYTHONPATH=$(pwd)
mkdir -p ~/msc_marl/artifacts/logs artifacts/metrics
TRAINER=scripts/trainers/14_mappo_smacv2_matched.py
MASTER=~/msc_marl/artifacts/logs/SET1_EVAL_MASTER.log
echo "================ SET 1 EVAL START: $(date) ================" | tee "$MASTER"

# run ONE analyzer job in the FOREGROUND (blocks until it finishes)
analyze_one () {                       # args: gpu mode comm folder seed
  local gpu=$1 mode=$2 comm=$3 folder=$4 seed=$5
  local ckpt="artifacts/checkpoints/$folder/seed$seed/latest.pt"
  local out="artifacts/metrics/${mode}_${comm}_seed${seed}.csv"
  if [ ! -f "$ckpt" ]; then
    echo "  SKIP (missing checkpoint): $ckpt" | tee -a "$MASTER"; return
  fi
  echo "  START $mode $comm seed$seed -> GPU $gpu ($(date +%H:%M:%S))" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/analysis/analyze_checkpoint.py \
    --trainer "$TRAINER" --checkpoint "$ckpt" --episodes 64 \
    --out "$out" \
    > ~/msc_marl/artifacts/logs/set1_${mode}_${comm}_seed${seed}.log 2>&1
  if [ -f "$out" ]; then
    echo "  DONE  $mode $comm seed$seed -> $out ($(date +%H:%M:%S))" | tee -a "$MASTER"
  else
    echo "  FAIL  $mode $comm seed$seed -> no CSV; see artifacts/logs/set1_${mode}_${comm}_seed${seed}.log" | tee -a "$MASTER"
  fi
}

JOBS=(
  "epo fc par_epo_fc_max"
  "epo gnn par_epo_gnn"
  "epo attn par_epo_attn"
  "standard fc std_fc_max"
  "standard gnn std_gnn"
  "standard attn std_attn"
)

# Build the flat list of (mode comm folder seed), then run 2 at a time.
PENDING=()
for job in "${JOBS[@]}"; do
  set -- $job; mode=$1; comm=$2; folder=$3
  for seed in 0 1 2; do
    PENDING+=("$mode $comm $folder $seed")
  done
done

i=0
while [ $i -lt ${#PENDING[@]} ]; do
  # GPU 0 job (foreground-backgrounded), GPU 1 job (background), then wait both
  set -- ${PENDING[$i]};      analyze_one 0 "$1" "$2" "$3" "$4" &
  p0=$!
  j=$(( i + 1 ))
  if [ $j -lt ${#PENDING[@]} ]; then
    set -- ${PENDING[$j]};    analyze_one 1 "$1" "$2" "$3" "$4" &
    p1=$!
  else
    p1=""
  fi
  wait $p0
  [ -n "$p1" ] && wait $p1
  i=$(( i + 2 ))
done

echo "================ AGGREGATING: $(date) ================" | tee -a "$MASTER"
if ls artifacts/metrics/*.csv >/dev/null 2>&1; then
  python scripts/analysis/aggregate_effects.py --glob "artifacts/metrics/*.csv" --out artifacts/metrics/summary.csv \
    2>&1 | tee -a "$MASTER"
  echo "Combined table -> artifacts/metrics/summary.csv" | tee -a "$MASTER"
else
  echo "  No CSVs produced — every analyzer run failed. Check a log:" | tee -a "$MASTER"
  echo "    tail -30 ~/msc_marl/artifacts/logs/set1_epo_fc_seed0.log" | tee -a "$MASTER"
fi

echo "================ SET 1 DONE: $(date) ================" | tee -a "$MASTER"
ls -la artifacts/metrics/*.csv 2>&1 | tee -a "$MASTER"
