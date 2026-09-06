#!/usr/bin/env bash
# ============================================================
# HPC Audit Script — Phases 4, 5, 6
# Run from the repo root on your Linux HPC:
#   bash scripts/hpc_audit_phases4to6.sh 2>&1 | tee hpc_audit_output.log
# ============================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PASS_COUNT=0
FAIL_COUNT=0

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }
pass()   { echo "[PASS] $1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail()   { echo "[FAIL] $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }

# ============================================================
banner "PHASE 4 — SMACv2 ENV CONSTRUCTION"
# ============================================================

echo "Running scripts/test_smacv2_standard.py ..."
if python scripts/smoke/test_smacv2_standard.py; then
    pass "Phase 4: env construction + 3-episode rollout"
else
    fail "Phase 4: env construction failed (see traceback above)"
    echo
    echo ">>> Phase 4 FAILED — stopping here. Fix the env before re-running. <<<"
    exit 1
fi

# ============================================================
banner "PHASE 5 — SMOKE ALL 4 TRAINERS (2000 steps each)"
# ============================================================

SMOKE_SEED=999
COMMON_ARGS="--seed $SMOKE_SEED --total-steps 2000 --rollout-len 256 --minibatch-size 64 --ppo-epochs 2 --hidden 64 --no-wandb"

run_smoke() {
    local label="$1"
    local script="$2"
    local extra_args="${3:-}"
    local ckpt_path="$4"

    echo
    echo "--- $label ---"
    local log_file="/tmp/smoke_${label}.log"

    if python "$script" $COMMON_ARGS $extra_args > "$log_file" 2>&1; then
        local status="OK"
    else
        local status="CRASHED"
    fi

    echo "Exit status: $status"
    echo "Last 5 lines of output:"
    tail -5 "$log_file"

    if [ "$status" = "CRASHED" ]; then
        echo
        echo ">>> FULL TRACEBACK <<<"
        cat "$log_file"
        fail "$label smoke: CRASHED"
        return 1
    fi

    # Check action_valid_frac == 1.0000
    local avf
    avf=$(grep "action_valid_frac" "$log_file" | tail -1 | grep -oP "action_valid_frac=\K[0-9.]+" || true)
    if [ -z "$avf" ]; then
        echo "  WARN: action_valid_frac not found in log output"
    elif python -c "import sys; v=float('$avf'); sys.exit(0 if abs(v-1.0)<1e-6 else 1)" 2>/dev/null; then
        echo "  action_valid_frac = $avf  [OK == 1.0000]"
    else
        echo "  action_valid_frac = $avf  [FAIL — expected 1.0000]"
        fail "$label action_valid_frac != 1.0"
    fi

    # Check ep_return is finite (no NaN/inf in last line)
    local last_ret
    last_ret=$(grep "ep_ret=" "$log_file" | tail -1 | grep -oP "ep_ret=\K[-0-9.]+" || true)
    if [ -z "$last_ret" ]; then
        echo "  WARN: ep_ret not found in log (may be no completed episode in 2000 steps)"
    elif python -c "import sys, math; v=float('$last_ret'); sys.exit(0 if math.isfinite(v) else 1)" 2>/dev/null; then
        echo "  ep_ret = $last_ret  [finite OK]"
    else
        echo "  ep_ret = $last_ret  [FAIL — NaN or Inf]"
        fail "$label ep_ret is not finite"
    fi

    # Check checkpoint was saved
    if [ -f "$ckpt_path" ]; then
        echo "  checkpoint saved: $ckpt_path  [OK]"
    else
        echo "  checkpoint NOT found at $ckpt_path  [FAIL]"
        fail "$label checkpoint not saved"
        return 1
    fi

    pass "$label smoke: completed without crash"
    return 0
}

# GNN extra: also check per_layer_L2
run_smoke_gnn() {
    local label="12_gnncomm"
    local script="scripts/trainers/12_gnncomm_mappo_smacv2.py"
    local extra_args="--num-layers 2"
    local ckpt_path="artifacts/checkpoints/smacv2_gnncomm/seed${SMOKE_SEED}/final.pt"
    local log_file="/tmp/smoke_${label}.log"

    echo
    echo "--- $label ---"
    if python "$script" $COMMON_ARGS $extra_args > "$log_file" 2>&1; then
        local status="OK"
    else
        local status="CRASHED"
    fi

    echo "Exit status: $status"
    echo "Last 5 lines of output:"
    tail -5 "$log_file"

    if [ "$status" = "CRASHED" ]; then
        echo; echo ">>> FULL TRACEBACK <<<"; cat "$log_file"
        fail "$label smoke: CRASHED"
        return 1
    fi

    # per_layer_L2 check — 3 finite numbers
    local l2_line
    l2_line=$(grep "per_layer_L2" "$log_file" | tail -1 || true)
    if [ -z "$l2_line" ]; then
        echo "  WARN: per_layer_L2 not found in log"
    else
        echo "  $l2_line"
        python -c "
import re, math, sys
line = '''$l2_line'''
nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
floats = [float(x) for x in nums]
if len(floats) < 2:
    print('  per_layer_L2: only', len(floats), 'numbers found (want >=2)')
    sys.exit(1)
bad = [v for v in floats if not math.isfinite(v)]
if bad:
    print('  per_layer_L2: non-finite values:', bad)
    sys.exit(1)
print('  per_layer_L2:', floats, '[', len(floats), 'finite numbers OK]')
" 2>/dev/null || echo "  per_layer_L2 parse check failed"
    fi

    # action_valid_frac
    local avf
    avf=$(grep "action_valid_frac" "$log_file" | tail -1 | grep -oP "action_valid_frac=\K[0-9.]+" || true)
    if [ -n "$avf" ]; then
        python -c "import sys; v=float('$avf'); sys.exit(0 if abs(v-1.0)<1e-6 else 1)" 2>/dev/null \
            && echo "  action_valid_frac = $avf  [OK]" \
            || echo "  action_valid_frac = $avf  [FAIL]"
    else
        echo "  WARN: action_valid_frac not found in log"
    fi

    # checkpoint
    if [ -f "$ckpt_path" ]; then
        echo "  checkpoint saved: $ckpt_path  [OK]"
    else
        echo "  checkpoint NOT found at $ckpt_path  [FAIL]"
        fail "$label checkpoint not saved"; return 1
    fi

    pass "$label smoke: completed without crash"
}

run_smoke "09_nocomm"   "scripts/trainers/09_nocomm_mappo_smacv2.py"   "" \
    "artifacts/checkpoints/smacv2_nocomm/seed${SMOKE_SEED}/final.pt"

run_smoke "10_fccomm"   "scripts/trainers/10_fccomm_mappo_smacv2.py"   "" \
    "artifacts/checkpoints/smacv2_fccomm/seed${SMOKE_SEED}/final.pt"

run_smoke "11_attncomm" "scripts/trainers/11_attncomm_mappo_smacv2.py" "" \
    "artifacts/checkpoints/smacv2_attncomm/seed${SMOKE_SEED}/final.pt"

run_smoke_gnn

# ============================================================
banner "PHASE 6 — CHECKPOINT AUDIT"
# ============================================================

echo "Running scripts/audit_smacv2_checkpoints.py ..."
if python scripts/smoke/audit_smacv2_checkpoints.py; then
    pass "Phase 6: all checkpoints OK"
else
    fail "Phase 6: one or more checkpoints BAD (see above)"
fi

# ============================================================
banner "FINAL SUMMARY"
# ============================================================

echo
echo "| Phase | Description                        | Result |"
echo "|-------|------------------------------------|--------|"
echo "| 4     | SMACv2 env construction             | see above |"
echo "| 5     | 4-trainer 2000-step smokes          | see above |"
echo "| 6     | Checkpoint field audit              | see above |"
echo
echo "PASS count : $PASS_COUNT"
echo "FAIL count : $FAIL_COUNT"
echo

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ">>> ALL PHASES PASSED — code is ready for full multi-seed runs. <<<"
    exit 0
else
    echo ">>> $FAIL_COUNT FAILURE(S) — investigate items marked [FAIL] above. <<<"
    exit 1
fi
