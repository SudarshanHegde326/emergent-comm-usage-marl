#!/usr/bin/env bash
# Day 41 — Weekend W6 status snapshot.
#
# Runs check_w6_progress.sh + audit_smacv2_checkpoints.py, stores
# both outputs to a dated file under experiments/. Safe to run any
# weekend day.
#
# Usage (from repo root):
#     bash scripts/weekend_w6_snapshot.sh

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../../src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../repo/src/envs/smacv2_env.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/../../../repo" && pwd )"
else
    echo "ERROR: cannot locate repo root"
    exit 1
fi

# Locate the checker + auditor wherever they live (scripts/ or a subfolder).
find_script() {
    # $1 = filename to find under $REPO_ROOT/scripts
    local hit
    hit="$(find "$REPO_ROOT/scripts" -name "$1" -type f 2>/dev/null | head -1)"
    echo "$hit"
}
CHECKER="$(find_script check_w6_progress.sh)"
AUDITOR="$(find_script audit_smacv2_checkpoints.py)"
if [ -z "$CHECKER" ]; then
    echo "ERROR: check_w6_progress.sh not found under $REPO_ROOT/scripts"
    exit 1
fi
if [ -z "$AUDITOR" ]; then
    echo "ERROR: audit_smacv2_checkpoints.py not found under $REPO_ROOT/scripts"
    exit 1
fi
echo "[snapshot] checker = $CHECKER"
echo "[snapshot] auditor = $AUDITOR"

STAMP="$(date +%F_%H%M)"
OUTDIR="$REPO_ROOT/experiments/weekend_snapshots"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/w6_snapshot_${STAMP}.txt"

{
    echo "============================================================"
    echo "W6 weekend snapshot @ $(date)"
    echo "============================================================"
    echo
    echo "----- check_w6_progress.sh -----"
    bash "$CHECKER"
    echo
    echo "----- audit_smacv2_checkpoints.py -----"
    python "$AUDITOR"
    echo
    echo "============================================================"
    echo "End of snapshot"
    echo "============================================================"
} | tee "$OUTFILE"

echo
echo "==> snapshot saved to $OUTFILE"
echo "    Compare against yesterday's snapshot in $OUTDIR/"