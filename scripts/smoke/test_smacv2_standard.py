from __future__ import annotations
import random
import sys

try:
    from smacv2.env import StarCraftCapabilityEnvWrapper
except ImportError as e:
    print(f"FAIL: cannot import StarCraftCapabilityEnvWrapper ({e}).")
    sys.exit(1)

PROTOSS_5V5_CONFIG = {
    "n_units": 5, "n_enemies": 5,
    "team_gen": {
        "dist_type": "weighted_teams", "unit_types": ["stalker", "zealot", "colossus"],
        "weights": [0.45, 0.45, 0.1], "exception_unit_types": ["colossus"], "observe": True,
    },
    "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "map_x": 32, "map_y": 32},
}
MAP_NAME = "10gen_protoss"

def make_standard_env(seed: int = 0):
    return StarCraftCapabilityEnvWrapper(
        capability_config=PROTOSS_5V5_CONFIG, map_name=MAP_NAME,
        debug=False, conic_fov=False, seed=seed,
    )

def main():
    print("# SMACv2 Standard-Mode Test — 10gen_protoss (5v5)\n")
    try:
        env = make_standard_env(seed=42)
    except Exception as e:
        print(f"FAIL — env construction raised {type(e).__name__}: {e}")
        sys.exit(1)

    env_info = env.get_env_info()
    n_agents, n_actions = env_info["n_agents"], env_info["n_actions"]
    obs_dim, state_dim = env_info["obs_shape"], env_info["state_shape"]
    episode_limit = env_info["episode_limit"]

    print(f"- n_agents       = {n_agents}")
    print(f"- n_actions      = {n_actions} (max action dim)")
    print(f"- per-agent obs  = {obs_dim}")
    print(f"- global state   = {state_dim}  (for centralised critic)")
    print(f"- episode_limit  = {episode_limit} steps\n---\n")

    episode_returns, episode_lens, episode_wins = [], [], []
    rng = random.Random(42)
    total_steps = 0

    for ep in range(3):
        env.reset()
        ep_return, ep_len, won = 0.0, 0, False
        for _step in range(episode_limit):
            avail = [env.get_avail_agent_actions(a) for a in range(n_agents)]
            actions = [rng.choice([i for i, v in enumerate(avail[a]) if v]) if any(avail[a]) else 0 for a in range(n_agents)]
            
            step_out = env.step(actions)
            reward, done, info = step_out
            ep_return += reward
            ep_len += 1
            total_steps += 1
            if done:
                won = bool(info.get("battle_won", False))
                break
        episode_returns.append(ep_return)
        episode_lens.append(ep_len)
        episode_wins.append(won)
        print(f"## Episode {ep + 1}\n- length      = {ep_len} steps\n- return      = {ep_return:+.3f}\n- battle_won  = {won}\n")

    env.close()
    n_ep = len(episode_returns)
    print(f"---\n## Summary\n- total_steps         = {total_steps}")
    print(f"- mean episode return = {sum(episode_returns) / n_ep:+.3f}")
    print(f"- mean episode length = {sum(episode_lens) / n_ep:.1f} steps")
    print(f"- win rate (random)   = {sum(episode_wins)}/{n_ep} = {100 * sum(episode_wins) / n_ep:.0f}%\n")
    print("STATUS: PASS — rollout completed without crash.")

if __name__ == "__main__":
    main()
