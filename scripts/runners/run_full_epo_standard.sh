#!/usr/bin/env bash
# =============================================================================
#  FULL FRESH RUN — all 8 conditions, 3 seeds each. Training only.
#
#    EPO      : none / fc / gnn / attn   -> 10M, 3 seeds each
#    STANDARD : none / fc / gnn / attn   -> 10M, 3 seeds each
#
#  Per-condition flags:
#    no-comm        -> --aux-coef 0        (aux OFF)
#    fc / attn / gnn-> --aux-coef 0.2      (aux ON, stronger)
#    fc also        -> --pool-type max
#
#  2 GPUs: seeds 0 & 1 run in parallel (one per GPU); seed 2 runs after on GPU 0.
#  Hyperparameters pinned on the command line. IB (--kl-coef) is NOT here.
#  Each stage: fresh folder, NO --resume, runs to FULL 10M, then VERIFIED
#  (all 3 seeds) before the next stage. Failures logged; chain keeps going.
#
#  NOTE: DELETES/overwrites the old par_epo_* and std_* folders on purpose.
#  Rename them first if you want to keep them.
# =============================================================================
set -uo pipefail
cd ~/msc_marl
export PYTHONPATH=$(pwd)
export WANDB_MODE=online
mkdir -p ~/msc_marl/artifacts/logs

STEPS=10000000
ENVS=8
HP="--hidden 128 --msg-dim 64 --lr-actor 3e-4 --ent-coef 0.01"
MASTER=~/msc_marl/artifacts/logs/FULL_RUN_MASTER.log
echo "================ FULL RUN START: $(date) ================" | tee "$MASTER"
echo "hyperparameters: $HP" | tee -a "$MASTER"

# ---- verify a finished run truly reached STEPS (all 3 seeds) ----------------
verify_done () {                       # args: label  out_dir
  local label="$1" out="$2" ok=1
  for SEED in 0 1 2; do
    local f="$out/seed$SEED/final.pt"
    if [ ! -f "$f" ]; then
      echo "  [VERIFY] $label seed$SEED: MISSING final.pt -> FAILED" | tee -a "$MASTER"; ok=0; continue
    fi
    local steps
    steps=$(python -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['steps'])" 2>/dev/null)
    if [ -z "${steps:-}" ]; then
      echo "  [VERIFY] $label seed$SEED: cannot read steps -> FAILED" | tee -a "$MASTER"; ok=0; continue
    fi
    local trans=$(( steps * ENVS ))
    if [ "$trans" -ge "$STEPS" ]; then
      echo "  [VERIFY] $label seed$SEED: OK ($trans transitions)" | tee -a "$MASTER"
    else
      echo "  [VERIFY] $label seed$SEED: ONLY $trans / $STEPS -> FAILED" | tee -a "$MASTER"; ok=0
    fi
  done
  return $(( 1 - ok ))
}

# ---- launch one seed (helper) -----------------------------------------------
launch_seed () {                       # args: gpu mode comm out tag seed pool aux
  local gpu="$1" mode="$2" comm="$3" out="$4" tag="$5" seed="$6" pool="$7" aux="$8"
  CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/trainers/15_mappo_smacv2_parallel.py \
    --comm "$comm" $pool $aux --mode "$mode" --seed "$seed" \
    --num-envs $ENVS --total-steps $STEPS $HP \
    --eval-interval 10000 --ckpt-interval 100000 \
    --out-dir "$out" \
    > ~/msc_marl/artifacts/logs/${tag}_${comm}_seed${seed}.log 2>&1 &
  echo "  $comm seed$seed -> GPU $gpu, PID $!" | tee -a "$MASTER"
}

# ---- launch one condition (3 seeds), wait for FULL completion ---------------
run_stage () {                         # args: mode  comm  out_dir  logtag
  local mode="$1" comm="$2" out="$3" tag="$4" pool=""
  local aux="--aux-coef 0.2"                     # fc / attn / gnn
  [ "$comm" = "fc" ]   && pool="--pool-type max"
  [ "$comm" = "none" ] && aux="--aux-coef 0"     # no-comm: aux OFF
  echo "========== [$tag] $comm ($mode) START: $(date) ==========" | tee -a "$MASTER"
  rm -rf "$out"                        # clean slate: no stale checkpoint, no --resume

  # seeds 0 and 1 in parallel (one per GPU)
  launch_seed 0 "$mode" "$comm" "$out" "$tag" 0 "$pool" "$aux"
  launch_seed 1 "$mode" "$comm" "$out" "$tag" 1 "$pool" "$aux"
  wait                                 # both finish

  # seed 2 alone on GPU 0
  launch_seed 0 "$mode" "$comm" "$out" "$tag" 2 "$pool" "$aux"
  wait                                 # seed 2 finishes

  verify_done "[$tag] $comm" "$out"
  echo "========== [$tag] $comm ($mode) END: $(date) ==========" | tee -a "$MASTER"
}

# ============================= STAGE 1: EPO x4 ===============================
run_stage epo  none  artifacts/checkpoints/par_epo_none    EPO
run_stage epo  fc    artifacts/checkpoints/par_epo_fc_max  EPO
run_stage epo  gnn   artifacts/checkpoints/par_epo_gnn     EPO
run_stage epo  attn  artifacts/checkpoints/par_epo_attn    EPO

# =========================== STAGE 2: STANDARD x4 ===========================
run_stage standard none  artifacts/checkpoints/std_none    STD
run_stage standard fc    artifacts/checkpoints/std_fc_max  STD
run_stage standard gnn   artifacts/checkpoints/std_gnn     STD
run_stage standard attn  artifacts/checkpoints/std_attn    STD

# =============================== FINAL REPORT ================================
echo "================ FULL RUN FINISHED: $(date) ================" | tee -a "$MASTER"
echo "Checkpoints written:" | tee -a "$MASTER"
ls -la artifacts/checkpoints/par_epo_*/seed*/final.pt artifacts/checkpoints/std_*/seed*/final.pt 2>&1 | tee -a "$MASTER"
echo "" | tee -a "$MASTER"
echo ">>> Read this file when done: $MASTER" | tee -a "$MASTER"
echo ">>> Any line containing 'FAILED' = that run did NOT reach 10M; re-run only that one." | tee -a "$MASTER"