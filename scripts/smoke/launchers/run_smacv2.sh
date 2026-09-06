#!/usr/bin/env bash
# =============================================================================
# SMACv2 runner - replaces run_3seeds_nocomm_smacv2.sh,
# run_3seeds_gnn_smacv2.sh and run_6seeds_fc_attn_smacv2.sh.
#
# WHY THOSE THREE WERE DELETED
# ----------------------------
# They passed --num-heads and --num-layers, which the trainer's argparse did
# not define. Every attention and GNN seed died in under a second with
# "unrecognized arguments", and because the loop used
# `|| { echo FAILED; continue; }` with no `set -e`, the script still printed
# "Run complete". This script pre-flights every flag before launching anything
# and aborts loudly on the first real failure.
#
# USAGE
# -----
#   ./run_smacv2.sh                                  # standard, all 4 conditions, 3 seeds
#   MODE=epo ./run_smacv2.sh                         # the EPO condition
#   MODE=epo COMMS="none attn" SEEDS="0 1 2" STEPS=2000000 ./run_smacv2.sh
#   DRY_RUN=1 ./run_smacv2.sh                        # validate flags, launch nothing
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- configuration (override from the environment) ----
MODE="${MODE:-standard}"                 # standard | epo
COMMS="${COMMS:-none fc gnn attn}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-2000000}"
MAP="${MAP:-10gen_protoss}"
HIDDEN="${HIDDEN:-64}"
MSG_DIM="${MSG_DIM:-32}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-200000}" # steps between resumable checkpoints
ATTEMPTS="${ATTEMPTS:-1}"                # >1 = auto-resubmit --resume on timeout
COMM_MEMORY="${COMM_MEMORY:-on}"         # on | off  (off = ablation)
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

TRAINER="$REPO_ROOT/scripts/trainers/14_mappo_smacv2_matched.py"
[ -f "$TRAINER" ] || { echo "ERROR: trainer not found: $TRAINER" >&2; exit 1; }

# ---- find python: prefer active venv, else repo venv, else system ----
if [ -n "${VIRTUAL_ENV:-}" ]; then PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
else PYTHON_EXEC="$(command -v python3 || command -v python)"; fi
[ -x "$PYTHON_EXEC" ] || { echo "ERROR: no python found. Activate your venv." >&2; exit 1; }

case "$MODE" in standard|epo) ;; *) echo "ERROR: MODE must be standard or epo" >&2; exit 1;; esac

MEMORY_FLAG="--comm-to-memory"
[ "$COMM_MEMORY" = "off" ] && MEMORY_FLAG="--no-comm-to-memory"

LOG_DIR="$REPO_ROOT/artifacts/logs/smacv2_${MODE}"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo " repo        : $REPO_ROOT"
echo " python      : $PYTHON_EXEC"
echo " mode        : $MODE   (epo => prob_obs_enemy=0.0, action_mask=False, 6v5)"
echo " map         : $MAP"
echo " conditions  : $COMMS"
echo " seeds       : $SEEDS"
echo " steps/run   : $STEPS"
echo " comm->memory: $COMM_MEMORY"
echo " started     : $(date)"
echo "=========================================================="

# ---- PRE-FLIGHT: prove every flag parses before burning HPC time ----
echo "--> pre-flight: validating arguments for every condition"
for COMM in $COMMS; do
    "$PYTHON_EXEC" - "$TRAINER" "$COMM" "$MODE" "$MAP" "$MSG_DIM" \
                     "$NUM_HEADS" "$NUM_LAYERS" "$HIDDEN" "$MEMORY_FLAG" <<'PY'
import importlib.util, sys
trainer, comm, mode, mp, msg, heads, layers, hidden, mem = sys.argv[1:10]
spec = importlib.util.spec_from_file_location("t", trainer)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.build_args(["--comm", comm, "--mode", mode, "--map-name", mp,
              "--msg-dim", msg, "--num-heads", heads, "--num-layers", layers,
              "--hidden", hidden, mem, "--no-wandb", "--seed", "0"])
print(f"    ok: --comm {comm}")
PY
done
echo "--> pre-flight passed"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing launched."; exit 0
fi

FAILED=()
for COMM in $COMMS; do
  for SEED in $SEEDS; do
    OUT_DIR="$REPO_ROOT/artifacts/checkpoints/smacv2_${MODE}_${COMM}"
    if [ -f "$OUT_DIR/seed${SEED}/final.pt" ]; then
        echo "==> SKIP  $COMM seed $SEED (final.pt exists)"
        continue
    fi
    LOG="$LOG_DIR/${COMM}_seed${SEED}.log"
    echo "==> RUN   $COMM seed $SEED   (log: $LOG)"

    # --resume makes each run continue from <out>/seed<seed>/latest.pt if it
    # exists. The trainer checkpoints every --ckpt-interval steps, so a
    # wall-clock timeout loses at most that many steps. ATTEMPTS>1 adds an
    # auto-resubmit loop: if the job dies (timeout, SC2 drop) before reaching
    # the target, it is relaunched --resume and picks up where it stopped.
    set +e
    ATTEMPT=1
    STATUS=0
    while [ "$ATTEMPT" -le "$ATTEMPTS" ]; do
        [ "$ATTEMPT" -gt 1 ] && echo "    (auto-resume attempt $ATTEMPT/$ATTEMPTS)"
        "$PYTHON_EXEC" "$TRAINER" \
            --comm "$COMM" \
            --mode "$MODE" \
            --map-name "$MAP" \
            --seed "$SEED" \
            --total-steps "$STEPS" \
            --hidden "$HIDDEN" \
            --msg-dim "$MSG_DIM" \
            --num-heads "$NUM_HEADS" \
            --num-layers "$NUM_LAYERS" \
            --eval-episodes "$EVAL_EPISODES" \
            --eval-interval "$EVAL_INTERVAL" \
            --ckpt-interval "$CKPT_INTERVAL" \
            --out-dir "$OUT_DIR" \
            --resume \
            $MEMORY_FLAG $EXTRA_ARGS 2>&1 | tee -a "$LOG"
        STATUS=${PIPESTATUS[0]}
        # Success, OR a completed run (final.pt present) -> stop retrying.
        if [ "$STATUS" -eq 0 ] || [ -f "$OUT_DIR/seed${SEED}/final.pt" ]; then
            break
        fi
        echo "    run exited $STATUS before completion; will resume."
        ATTEMPT=$((ATTEMPT + 1))
        sleep 3
    done
    set -e

    if [ "$STATUS" -ne 0 ] && [ ! -f "$OUT_DIR/seed${SEED}/final.pt" ]; then
        echo "!!! FAILED: $COMM seed $SEED (exit $STATUS after $ATTEMPTS attempts) -- see $LOG"
        FAILED+=("$COMM/seed$SEED")
    fi
  done
done

echo "=========================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo " ALL RUNS COMPLETED SUCCESSFULLY  $(date)"
    exit 0
else
    echo " ${#FAILED[@]} RUN(S) FAILED: ${FAILED[*]}"
    echo " finished $(date)"
    exit 1
fi
