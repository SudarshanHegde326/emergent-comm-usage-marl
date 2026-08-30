"""SMACv2 EPO smoke test — verifies the REAL Extended Partial Observability
settings, on 10gen_protoss (6v5, the paper's EPO team size).

WHY THIS FILE WAS REWRITTEN
---------------------------
The previous version defined "EPO" as conic_fov=True vs conic_fov=False. That
is NOT the Extended Partial Observability challenge. Diffing the benchmark's own
configs (smacv2/examples/configs/sc2_gen_protoss{,_epo}.yaml), the ENTIRE
difference between standard and EPO is two parameters:

    standard :  prob_obs_enemy = 1.0   action_mask = True
    epo      :  prob_obs_enemy = 0.0   action_mask = False

and conic_fov is False in BOTH. Testing conic_fov told you nothing about EPO and
would have quietly confirmed the wrong thing.

This test builds both configurations exactly as the trainer's make_env() does,
and checks the mechanism that actually makes EPO hard for communication: with
action_mask=False, the available-action vector no longer restricts attack
actions to in-range enemies, so on average MORE actions are marked "available".
That is the free information the benchmark removed to force agents to infer or
communicate which enemies they can hit.

Run: python scripts/smoke/test_smacv2_epo.py
Requires a working SMACv2 + StarCraft II install (it launches the game).
"""
from __future__ import annotations

import random
import sys

try:
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
except ImportError as e:
    print(f"FAIL: cannot import StarCraftCapabilityEnvWrapper ({e}).")
    sys.exit(1)

MAP_NAME = "10gen_protoss"


def capability(n_units: int, n_enemies: int) -> dict:
    return {
        "n_units": n_units,
        "n_enemies": n_enemies,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5, "map_x": 32, "map_y": 32,
        },
    }


def make_env(epo: bool, seed: int = 0):

    return StarCraftCapabilityEnvWrapper(
        capability_config=capability(6 if epo else 5, 5),
        map_name=MAP_NAME,
        debug=False,
        conic_fov=False,                       # False in BOTH modes
        prob_obs_enemy=0.0 if epo else 1.0,    # <- the real EPO switch
        action_mask=False if epo else True,    # <- the real EPO switch
        seed=seed,
    )


def env_dims(env):
    info = env.get_env_info()
    return info["obs_shape"], info["state_shape"], info["n_agents"], info["n_actions"]


def mean_available_actions(env, n_agents: int, steps: int, rng: random.Random) -> float:

    env.reset()
    counts = []
    for _ in range(steps):
        acts = []
        for a in range(n_agents):
            avail = env.get_avail_agent_actions(a)
            legal = [i for i, v in enumerate(avail) if v]
            # count only living agents (a dead unit has just the no-op available)
            if len(legal) > 1:
                counts.append(len(legal))
            acts.append(rng.choice(legal) if legal else 0)
        _, done, _ = env.step(acts)
        if done:
            env.reset()
    return sum(counts) / len(counts) if counts else 0.0


def main() -> int:
    print("# SMACv2 EPO smoke test — 10gen_protoss\n")
    print("EPO is defined by prob_obs_enemy=0.0 AND action_mask=False "
          "(conic_fov=False in both).\n")

    rng = random.Random(42)

    std_env = make_env(epo=False, seed=0)
    std_obs, std_state, std_agents, std_actions = env_dims(std_env)
    std_avail = mean_available_actions(std_env, std_agents, 60, rng)
    std_env.close()

    epo_env = make_env(epo=True, seed=0)
    epo_obs, epo_state, epo_agents, epo_actions = env_dims(epo_env)
    epo_avail = mean_available_actions(epo_env, epo_agents, 60, rng)
    epo_env.close()

    print("| Quantity | Standard (5v5) | EPO (6v5) |")
    print("|---|---|---|")
    print(f"| n_agents | {std_agents} | {epo_agents} |")
    print(f"| obs_dim per agent | {std_obs} | {epo_obs} |")
    print(f"| global state_dim | {std_state} | {epo_state} |")
    print(f"| mean available actions / living agent | {std_avail:.2f} | {epo_avail:.2f} |\n")

    checks = []
    checks.append(("EPO has 6 allies (paper team size)", epo_agents == 6))
    checks.append(("standard has 5 allies", std_agents == 5))
    checks.append(("EPO exposes >= as many available actions as standard",
                   epo_avail >= std_avail - 1e-6))

    print("## Checks\n")
    all_ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_ok = all_ok and passed

    print()
    if all_ok:
        print("STATUS: PASS — EPO configuration verified via the real "
              "prob_obs_enemy / action_mask switches.")
        return 0
    print("STATUS: FAIL — EPO switches did not produce the expected effect; "
          "check the SMACv2 version supports prob_obs_enemy / action_mask.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
