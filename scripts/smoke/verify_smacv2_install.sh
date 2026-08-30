#!/usr/bin/env bash
# Day 35 — verify SMACv2 is installable + runnable on this machine.
#
# Five checks, each a possible failure point. Each prints PASS / WARN /
# FAIL with a one-line diagnostic; the overall exit code is the logical
# AND of the hard checks (WARN does not fail the build).
#
# Outputs a markdown-formatted summary suitable for piping into
# `docs/smacv2_install_verification.md`.
#
# Usage (from repo root):
#     bash scripts/verify_smacv2_install.sh 2>&1 | tee docs/smacv2_install_verification.md
#
# NOTE ON VERSIONS: this script intentionally does NOT hardcode a
# StarCraft II version or a map-download URL, because the required
# version changes between SMACv2 releases. Where a value must be
# confirmed at source, the script tells you exactly what to check
# rather than guessing.

set -uo pipefail

echo "# SMACv2 Install Verification — $(date '+%Y-%m-%d %H:%M')"
echo
echo "Host: \`$(hostname)\`"
echo
echo "Python: \`$(python --version 2>&1)\`"
echo
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "venv: \`active -> $VIRTUAL_ENV\` (pip will install into the venv)"
    IN_VENV=1
else
    echo "venv: \`not active\` (pip will target system Python)"
    IN_VENV=0
fi
echo
echo "---"
echo

OVERALL=0

# ---------------------------------------------------------------
# Check 1 — pip install smacv2
# ---------------------------------------------------------------
echo "## Check 1 — pip install"
echo
if pip show smacv2 >/dev/null 2>&1; then
    echo "PASS — smacv2 already installed: \`$(pip show smacv2 | grep -i '^version')\`"
else
    if [ "$IN_VENV" -eq 1 ]; then
        PIP_CMD="pip install smacv2"
    else
        PIP_CMD="pip install --break-system-packages smacv2"
    fi
    echo "Running: \`$PIP_CMD\`"
    if $PIP_CMD >/tmp/smacv2_pip.log 2>&1; then
        echo "PASS — pip install smacv2 succeeded."
        pip show smacv2 | grep -i '^version'
    else
        echo "FAIL — pip install smacv2 failed. Last 10 lines of pip log:"
        echo
        echo '```'
        tail -10 /tmp/smacv2_pip.log
        echo '```'
        OVERALL=1
    fi
fi
echo

# ---------------------------------------------------------------
# Check 2 — StarCraft II base game accessible
# ---------------------------------------------------------------
echo "## Check 2 — StarCraft II base game"
echo
SC2DIR=""
if [ -n "${SC2PATH:-}" ] && [ -d "$SC2PATH" ]; then
    SC2DIR="$SC2PATH"
elif [ -d "$HOME/StarCraftII" ]; then
    SC2DIR="$HOME/StarCraftII"
    echo "NOTE — \$SC2PATH not set. Add to ~/.bashrc:"
    echo "    export SC2PATH=\$HOME/StarCraftII"
    echo
fi

if [ -z "$SC2DIR" ]; then
    echo "FAIL — StarCraftII not found at \$SC2PATH or \$HOME/StarCraftII."
    echo
    echo "SMACv2 wraps StarCraft II; the SC2 base game must be installed first."
    echo "Confirm the CURRENT required SC2 version from the SMACv2 GitHub README"
    echo "(https://github.com/oxwhirl/smacv2) — do NOT assume an old version."
    echo "Typical Linux install (CHECK the version + EULA URL at source first):"
    echo "    # version below is a PLACEHOLDER — confirm the one SMACv2 wants"
    echo "    wget <SC2_LINUX_ZIP_URL_FROM_BLIZZARD>"
    echo "    unzip -P iagreetotheeula <downloaded.zip> -d \$HOME/"
    echo "    export SC2PATH=\$HOME/StarCraftII"
    OVERALL=1
elif [ -d "$SC2DIR/Versions" ]; then
    echo "PASS — StarCraftII found at \`$SC2DIR\` and Versions/ present."
else
    echo "FAIL — \`$SC2DIR\` exists but has no Versions/ subdirectory."
    echo "       This usually means an incomplete or corrupted SC2 download."
    OVERALL=1
fi
echo

# ---------------------------------------------------------------
# Check 3 — SMAC maps present
# ---------------------------------------------------------------
echo "## Check 3 — SMAC map pack"
echo
if [ -n "$SC2DIR" ] && [ -d "$SC2DIR/Maps" ]; then
    if ls "$SC2DIR/Maps" 2>/dev/null | grep -qiE "smac|gen_"; then
        echo "PASS — a SMAC-style map directory is present under \`$SC2DIR/Maps\`."
    else
        echo "WARN — \`$SC2DIR/Maps\` exists but no SMAC maps detected."
        echo "       SMACv2 procedural maps are a separate download. If Check 4"
        echo "       fails with 'map not found', install the SMAC_Maps pack into"
        echo "       \`$SC2DIR/Maps/\` per the SMACv2 README."
    fi
else
    echo "WARN — could not find a Maps/ directory to check."
    echo "       If Check 4 fails with 'map not found', the map pack is missing."
fi
echo

# ---------------------------------------------------------------
# Check 4 — Python import
# ---------------------------------------------------------------
echo "## Check 4 — Python import"
echo
if python -c "from smacv2.env import StarCraft2Env" 2>/tmp/smacv2_import.log; then
    echo "PASS — \`from smacv2.env import StarCraft2Env\` succeeded."
else
    echo "FAIL — import failed:"
    echo
    echo '```'
    cat /tmp/smacv2_import.log
    echo '```'
    OVERALL=1
fi
echo

# ---------------------------------------------------------------
# Check 5 — short random-agent rollout on a v2 map
# ---------------------------------------------------------------
echo "## Check 5 — random-agent rollout (protoss_5_vs_5)"
echo
python - <<'PY' >/tmp/smacv2_rollout.log 2>&1
import sys
try:
    from smacv2.env import StarCraft2Env
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
except Exception as e:
    print(f"  IMPORT NOTE: {type(e).__name__}: {e}")
    # Fall back to the bare env import if the wrapper path differs in
    # this SMACv2 version; the constructor below will reveal the shape.
    StarCraftCapabilityEnvWrapper = None
    try:
        from smacv2.env import StarCraft2Env
    except Exception as e2:
        print(f"  EXCEPTION (import): {type(e2).__name__}: {e2}")
        sys.exit(1)

import random

def run_rollout(env, n_agents):
    env.reset()
    n_actions = env.get_total_actions() if hasattr(env, "get_total_actions") else None
    print(f"  n_agents = {n_agents}")
    if n_actions is not None:
        print(f"  n_actions = {n_actions}")
    reward = 0.0
    for step in range(50):
        avail = [env.get_avail_agent_actions(a) for a in range(n_agents)]
        actions = []
        for a in range(n_agents):
            valid = [i for i, v in enumerate(avail[a]) if v == 1]
            actions.append(random.choice(valid) if valid else 0)
        reward, done, info = env.step(actions)
        if done:
            print(f"  episode done at step {step+1}, last reward = {reward:.3f}")
            break
    else:
        print(f"  50 steps completed, last reward = {reward:.3f}")
    env.close()

try:
    # SMACv2 procedural config for protoss_5_vs_5. The exact keys can vary
    # by SMACv2 version — if this raises, confirm the constructor in the
    # SMACv2 README / Ellis 2023 and adjust capability_config accordingly.
    distribution_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot"],
            "weights": [0.5, 0.5],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }

    if StarCraftCapabilityEnvWrapper is not None:
        env = StarCraftCapabilityEnvWrapper(
            capability_config=distribution_config,
            map_name="10gen_protoss",
            debug=False,
            conic_fov=False,
        )
        n_agents = env.env.n_agents if hasattr(env, "env") else env.n_agents
        run_rollout(env, n_agents)
    else:
        # Last-resort bare construction; likely to fail on real SMACv2,
        # which is itself the diagnostic signal that a capability_config
        # is required.
        env = StarCraft2Env(map_name="protoss_5_vs_5")
        run_rollout(env, env.n_agents)

    print("  STATUS: rollout completed without crash.")
    sys.exit(0)
except Exception as e:
    print(f"  EXCEPTION: {type(e).__name__}: {e}")
    print("  -> If this mentions capability_config / map_name / distribution,")
    print("     confirm the exact StarCraft2Env constructor in the SMACv2 README")
    print("     (the procedural-generation config differs from SMACv1).")
    sys.exit(1)
PY
RC=$?
if [ "$RC" -eq 0 ]; then
    echo "PASS — random-agent rollout completed."
    echo
    echo '```'
    cat /tmp/smacv2_rollout.log
    echo '```'
else
    echo "FAIL — random-agent rollout did not complete."
    echo
    echo '```'
    cat /tmp/smacv2_rollout.log
    echo '```'
    OVERALL=1
fi
echo

# ---------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------
echo "---"
echo
echo "## Overall verdict"
echo
if [ "$OVERALL" -eq 0 ]; then
    echo "**PASS** — SMACv2 install is viable on this machine."
    echo
    echo "Next steps if W6 SMACv2 path is chosen Monday:"
    echo "    - Wire SMACv2 env into a MAPPO trainer (Day 36 morning)."
    echo "    - Smoke-test no-comm condition (Day 36 afternoon)."
    echo "    - 3-seed sweep starts Day 37."
else
    echo "**FAIL** — at least one hard check failed. SMACv2 install is NOT"
    echo "viable without additional setup (see specific failures above)."
    echo
    echo "Implication: the W6 SMACv2 path requires >= 1 day of install"
    echo "debugging BEFORE any MAPPO integration begins. Bring this finding"
    echo "to Monday's Helal conversation — a failed install is a strong"
    echo "signal for the Ch.4 pivot."
fi
echo
exit $OVERALL
