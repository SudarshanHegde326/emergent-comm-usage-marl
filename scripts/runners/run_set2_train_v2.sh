#!/usr/bin/env bash
# =============================================================================
#  SET 2 — TRAIN (run once, leave ~3 days). Parallel trainer, 2 GPUs.
#
#  Trained-in channel constraints on FC, matched to FC baseline settings
#  (hidden 128, msg-dim 64, lr 3e-4, ent 0.01, pool max, aux 0.2):
#
#    Noise      EPO+STD  FC  sigma 0.5, 1.0      seed 0     2M   (4 runs)
#    Bandwidth  EPO+STD  FC  quant 2, 4, 8 bits  seed 0     2M   (6 runs)
#    IB/KL      EPO+STD  FC  beta 0.1, 0.001     seeds 0,1  10M  (8 runs)
#
#  Scheduling: same-duration runs paired so neither GPU idles — all 2M runs
#  first (2 at a time), then the 10M IB runs (seeds 0,1 in parallel per beta,
#  per mode). Adding standard IB doubles Block B vs the EPO-only version.
#
#  Each run: fresh folder (rm -rf) + NO --resume -> re-running REPLACES old
#  Set-2 results. Set-2 uses its OWN artifacts/checkpoints/set2_* folders, so it does
#  NOT touch Set-1 results (artifacts/metrics/*.csv) or the baselines.
#  Runs to FULL target (no time limit), then VERIFIED.
# =============================================================================
set -uo pipefail
cd ~/msc_marl
export PYTHONPATH=$(pwd)
export WANDB_MODE=online
mkdir -p ~/msc_marl/logs
ENVS=8
HP="--hidden 128 --msg-dim 64 --lr-actor 3e-4 --ent-coef 0.01 --pool-type max --aux-coef 0.2"
TRAINER=scripts/trainers/15_mappo_smacv2_parallel.py
MASTER=~/msc_marl/artifacts/logs/SET2_TRAIN_MASTER.log
echo "================ SET 2 TRAIN START: $(date) ================" | tee "$MASTER"
echo "shared settings: $HP" | tee -a "$MASTER"

# ---- verify one run reached its target (target in transitions) --------------
verify_run () {                        # args: label out_dir target seed
  local label="$1" out="$2" target="$3" seed="$4"
  local f="$out/seed$seed/final.pt"
  if [ ! -f "$f" ]; then
    echo "  [VERIFY] $label seed$seed: MISSING final.pt -> FAILED" | tee -a "$MASTER"; return
  fi
  local steps
  steps=$(python -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['steps'])" 2>/dev/null)
  if [ -z "${steps:-}" ]; then
    echo "  [VERIFY] $label seed$seed: cannot read steps -> FAILED" | tee -a "$MASTER"; return
  fi
  local trans=$(( steps * ENVS ))
  if [ "$trans" -ge "$target" ]; then
    echo "  [VERIFY] $label seed$seed: OK ($trans / $target)" | tee -a "$MASTER"
  else
    echo "  [VERIFY] $label seed$seed: ONLY $trans / $target -> FAILED" | tee -a "$MASTER"
  fi
}

# ---- launch one run on a given GPU (backgrounded); uses global $MODE ---------
launch () {                            # args: gpu out logtag seed steps extra_flags...
  local gpu="$1" out="$2" tag="$3" seed="$4" steps="$5"; shift 5
  CUDA_VISIBLE_DEVICES=$gpu nohup python "$TRAINER" \
    --comm fc --mode "$MODE" --seed "$seed" \
    --num-envs $ENVS --total-steps "$steps" $HP "$@" \
    --eval-interval 10000 --ckpt-interval 100000 \
    --out-dir "$out" \
    > ~/msc_marl/artifacts/logs/${tag}_seed${seed}.log 2>&1 &
  echo "  $tag seed$seed -> GPU $gpu, PID $! ($out, $steps steps, flags: $*)" | tee -a "$MASTER"
}

# =========================================================================
#  BLOCK A — all 2M runs (noise + bandwidth), seed 0, 2 at a time
#  entry: "out_dir|logtag|MODE|extra_flags"
# =========================================================================
STEPS2M=2000000
A_JOBS=(
  "artifacts/checkpoints/set2_noise_epo_fc_s0.5|noise_epo_s0.5|epo|--noise-std 0.5"
  "artifacts/checkpoints/set2_noise_epo_fc_s1.0|noise_epo_s1.0|epo|--noise-std 1.0"
  "artifacts/checkpoints/set2_noise_std_fc_s0.5|noise_std_s0.5|standard|--noise-std 0.5"
  "artifacts/checkpoints/set2_noise_std_fc_s1.0|noise_std_s1.0|standard|--noise-std 1.0"
  "artifacts/checkpoints/set2_bw_epo_fc_q2|bw_epo_q2|epo|--quant-bits 2"
  "artifacts/checkpoints/set2_bw_epo_fc_q4|bw_epo_q4|epo|--quant-bits 4"
  "artifacts/checkpoints/set2_bw_epo_fc_q8|bw_epo_q8|epo|--quant-bits 8"
  "artifacts/checkpoints/set2_bw_std_fc_q2|bw_std_q2|standard|--quant-bits 2"
  "artifacts/checkpoints/set2_bw_std_fc_q4|bw_std_q4|standard|--quant-bits 4"
  "artifacts/checkpoints/set2_bw_std_fc_q8|bw_std_q8|standard|--quant-bits 8"
)
echo "========== BLOCK A: 2M noise+bandwidth ($(date)) ==========" | tee -a "$MASTER"
i=0
for job in "${A_JOBS[@]}"; do
  IFS='|' read -r OUT TAG MODE FLAGS <<< "$job"
  gpu=$(( i % 2 ))
  rm -rf "$OUT"
  launch "$gpu" "$OUT" "$TAG" 0 "$STEPS2M" $FLAGS
  i=$(( i + 1 ))
  [ $(( i % 2 )) -eq 0 ] && wait
done
wait
for job in "${A_JOBS[@]}"; do
  IFS='|' read -r OUT TAG MODE FLAGS <<< "$job"
  verify_run "$TAG" "$OUT" "$STEPS2M" 0
done

# =========================================================================
#  BLOCK B — IB 10M, EPO + STANDARD, beta 0.1 & 0.001, seeds 0 and 1
#  seeds 0 & 1 in parallel (one per GPU) per (mode, beta)
# =========================================================================
STEPS10M=10000000
echo "========== BLOCK B: 10M IB (EPO + STANDARD) ($(date)) ==========" | tee -a "$MASTER"
for MODE in epo standard; do
  for BETA in 0.1 0.001; do
    OUT="artifacts/checkpoints/set2_ib_${MODE}_fc_b${BETA}"
    TAG="ib_${MODE}_b${BETA}"
    rm -rf "$OUT"
    launch 0 "$OUT" "$TAG" 0 "$STEPS10M" --kl-coef "$BETA"
    launch 1 "$OUT" "$TAG" 1 "$STEPS10M" --kl-coef "$BETA"
    wait
    verify_run "$TAG" "$OUT" "$STEPS10M" 0
    verify_run "$TAG" "$OUT" "$STEPS10M" 1
  done
done

echo "================ SET 2 FINISHED: $(date) ================" | tee -a "$MASTER"
ls -la artifacts/checkpoints/set2_*/seed*/final.pt 2>&1 | tee -a "$MASTER"
echo ">>> Read this file: $MASTER  (any 'FAILED' = re-run only that one)" | tee -a "$MASTER"
