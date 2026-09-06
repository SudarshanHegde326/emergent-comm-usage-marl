#!/usr/bin/env bash
set -uo pipefail

echo "=========================================================="
echo "SMACv2 Full Install — $(date '+%Y-%m-%d %H:%M')"
echo "=========================================================="
echo "Host: $(hostname)"
echo "Python: $(python --version 2>&1)"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "venv: active -> $VIRTUAL_ENV"
    IN_VENV=1
else
    echo "venv: not active (targeting system Python)"
    IN_VENV=0
fi
echo

OVERALL=0
SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
SKIP_SC2="${SKIP_SC2:-0}"
ALLOW_SC2_DOWNLOAD="${ALLOW_SC2_DOWNLOAD:-0}"

echo "[STEP 1/6] pip install smacv2"
if pip show smacv2 >/dev/null 2>&1; then
    VER=$(pip show smacv2 | awk '/^Version/ {print $2}')
    echo "  PASS — smacv2 $VER already installed."
else
    if [ "$IN_VENV" -eq 1 ]; then
        PIP_CMD="pip install smacv2"
    else
        PIP_CMD="pip install --break-system-packages smacv2"
    fi
    echo "  running: $PIP_CMD"
    if $PIP_CMD >/tmp/smacv2_pip.log 2>&1; then
        VER=$(pip show smacv2 | awk '/^Version/ {print $2}')
        echo "  PASS — installed smacv2 $VER."
    else
        echo "  FAIL — pip install failed. Last 10 lines:"
        tail -10 /tmp/smacv2_pip.log | sed 's/^/    /'
        OVERALL=1
    fi
fi
echo

echo "[STEP 2/6] StarCraft II binary"
if [ "$SKIP_SC2" = "1" ]; then
    echo "  SKIP — SKIP_SC2=1; assuming SC2 installed elsewhere."
elif [ -d "$SC2PATH/Versions" ]; then
    echo "  PASS — SC2 found at $SC2PATH (Versions/ present)."
    ls "$SC2PATH/Versions" | head -3 | sed 's/^/    /'
elif [ -d "$SC2PATH" ]; then
    echo "  FAIL — $SC2PATH exists but has no Versions/ (incomplete install)."
    OVERALL=1
else
    echo "  not found at $SC2PATH"
    if [ "$ALLOW_SC2_DOWNLOAD" != "1" ]; then
        echo "  STOP — auto-download is OFF (safety)."
        OVERALL=1
    fi
fi
echo

echo "[STEP 3/6] SMACv2 SC2 maps"
SC2_MAPS_DIR="$SC2PATH/Maps"
SMAC_MAPS_DIR="$SC2_MAPS_DIR/SMAC_Maps"
if [ -d "$SMAC_MAPS_DIR" ] && [ "$(ls -A "$SMAC_MAPS_DIR" 2>/dev/null | wc -l)" -gt 5 ]; then
    echo "  PASS — SMAC_Maps present:"
    ls "$SMAC_MAPS_DIR" | head -5 | sed 's/^/    /'
else
    echo "  WARN — SMAC_Maps not found under $SMAC_MAPS_DIR."
fi
echo

echo "[STEP 4/6] Python import"
if python -c "from smacv2.env import StarCraftCapabilityEnvWrapper" 2>/tmp/smacv2_import.log; then
    echo "  PASS — StarCraftCapabilityEnvWrapper imports cleanly."
else
    echo "  FAIL — import error:"
    cat /tmp/smacv2_import.log | sed 's/^/    /'
    OVERALL=1
fi
echo

echo "[STEP 5/6] Standard-mode env (10gen_protoss, conic_fov=False)"
python - <<'INNER_PY' 2>&1 | sed 's/^/    /'
try:
    from smacv2.env import StarCraftCapabilityEnvWrapper
    import random
    cfg = {
        "n_units": 5, "n_enemies": 5,
        "team_gen": {"dist_type": "weighted_teams",
                     "unit_types": ["stalker", "zealot", "colossus"],
                     "weights": [0.45, 0.45, 0.1],
                     "exception_unit_types": ["colossus"], "observe": True},
        "start_positions": {"dist_type": "surrounded_and_reflect",
                            "p": 0.5, "map_x": 32, "map_y": 32},
    }
    env = StarCraftCapabilityEnvWrapper(capability_config=cfg,
                                        map_name="10gen_protoss",
                                        debug=False, conic_fov=False, seed=0)
    env.reset()
    info = env.get_env_info()
    n = info["n_agents"]
    print(f"PASS — n_agents={n}, n_actions={info['n_actions']}, obs_dim={info['obs_shape']}, state_dim={info['state_shape']}")
    for _ in range(10):
        avail = [env.get_avail_agent_actions(a) for a in range(n)]
        acts = [random.choice([i for i,v in enumerate(avail[a]) if v]) for a in range(n)]
        env.step(acts)
    env.close()
    print("       10-step random rollout completed.")
except Exception as e:
    print(f"FAIL — {type(e).__name__}: {e}")
INNER_PY
echo

echo "[STEP 6/6] Restricted/EPO-mode env (conic_fov=True)"
python - <<'INNER_PY' 2>&1 | sed 's/^/    /'
try:
    from smacv2.env import StarCraftCapabilityEnvWrapper
    import random
    cfg = {
        "n_units": 5, "n_enemies": 5,
        "team_gen": {"dist_type": "weighted_teams",
                     "unit_types": ["stalker", "zealot", "colossus"],
                     "weights": [0.45, 0.45, 0.1],
                     "exception_unit_types": ["colossus"], "observe": True},
        "start_positions": {"dist_type": "surrounded_and_reflect",
                            "p": 0.5, "map_x": 32, "map_y": 32},
    }
    env = StarCraftCapabilityEnvWrapper(capability_config=cfg,
                                        map_name="10gen_protoss",
                                        debug=False, conic_fov=True, seed=0)
    env.reset()
    info = env.get_env_info()
    n = info["n_agents"]
    print(f"PASS — restricted mode loaded: obs_dim={info['obs_shape']}, state_dim={info['state_shape']}")
    for _ in range(10):
        avail = [env.get_avail_agent_actions(a) for a in range(n)]
        acts = [random.choice([i for i,v in enumerate(avail[a]) if v]) for a in range(n)]
        env.step(acts)
    env.close()
    print("       10-step random rollout completed in restricted mode.")
except Exception as e:
    print(f"FAIL — {type(e).__name__}: {e}")
INNER_PY
echo

echo "=========================================================="
if [ "$OVERALL" -eq 0 ]; then
    echo "RESULT: SMACv2 install steps PASSED."
else
    echo "RESULT: install INCOMPLETE — see failures above."
fi
echo "=========================================================="
exit $OVERALL
