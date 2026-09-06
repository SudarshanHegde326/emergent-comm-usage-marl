"""Load a trained AttentionCommPipeline checkpoint and compute statistics
about the attention weights it produces during evaluation rollouts.

WHAT THIS GIVES YOU (for Chapter 5)
-----------------------------------
- Mean attention entropy per head (low = focused, high = diffuse)
- Mean attention matrix (N, N) averaged across heads (heatmap-ready)
- Per-head attention matrices (N, N) so we can show head specialisation
- Mean episode return at evaluation time (a sanity check)

The output is a JSON file plus a NumPy .npz with the raw matrices for
downstream plotting.

USAGE
-----
    python analyze_attention_weights.py \
        --checkpoint artifacts/checkpoints/attncomm_mappo/seed0/final.pt \
        --eval-episodes 20 \
        --out artifacts/checkpoints/attncomm_mappo/seed0/attn_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# --- path setup (same trick as the trainer) ---------------------------------
THIS_FILE = Path(__file__).resolve()
candidate_roots = [THIS_FILE.parent.parent.parent, THIS_FILE.parent.parent]
REPO_ROOT = None
for root in candidate_roots:
    if (root / "src" / "comms" / "attention_comm.py").exists():
        REPO_ROOT = root
        break
    if (root / "repo" / "src" / "comms" / "attention_comm.py").exists():
        REPO_ROOT = root / "repo"
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root.")
sys.path.insert(0, str(REPO_ROOT))

from src.comms.attention_comm import AttentionCommPipeline                # noqa: E402



# ============================================================================
# [FIX] Central env config. Set once from CLI in main(); make_env reads it.
# Previously n_agents was HARDCODED to 5 and could only be changed by editing
# this file by hand (which is how a 3-agent analysis of a 6-agent model got
# produced). Never hand-edit config again -- pass flags.
# ============================================================================
ENV_CFG = {
    "n_agents": 6,
    "agent_neighbors": 2,
    "landmark_neighbors": 2,
    "max_cycles": 50,
    "num_landmarks": 4,
    "num_obstacles": 0,
}

def _apply_env_cfg_from_args(args):
    """Fill ENV_CFG from parsed CLI args and SHOUT the config."""
    for k in ENV_CFG:
        if hasattr(args, k) and getattr(args, k) is not None:
            ENV_CFG[k] = getattr(args, k)
    print(f"[analysis/env] {ENV_CFG}", flush=True)
    return ENV_CFG


def make_env(seed: int, n_agents: int = None, agent_neighbors: int = None,
             landmark_neighbors: int = None, max_cycles: int = None,
             num_landmarks: int = None, num_obstacles: int = None):
    """[FIX] Reads ENV_CFG unless explicitly overridden. MUST match the
    environment the checkpoint was TRAINED in, or results are meaningless."""
    n_agents           = ENV_CFG["n_agents"]           if n_agents is None else n_agents
    agent_neighbors    = ENV_CFG["agent_neighbors"]    if agent_neighbors is None else agent_neighbors
    landmark_neighbors = ENV_CFG["landmark_neighbors"] if landmark_neighbors is None else landmark_neighbors
    max_cycles         = ENV_CFG["max_cycles"]         if max_cycles is None else max_cycles
    num_landmarks      = ENV_CFG["num_landmarks"]      if num_landmarks is None else num_landmarks
    num_obstacles      = ENV_CFG["num_obstacles"]      if num_obstacles is None else num_obstacles

    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if agent_neighbors < 0 else agent_neighbors
    _ln = None if landmark_neighbors < 0 else landmark_neighbors
    env = planC_parallel_env(
        N=n_agents, num_landmarks=num_landmarks, num_obstacles=num_obstacles,
        local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=max_cycles, continuous_actions=False
    )
    env.reset(seed=seed)
    return env


def load_pipeline(checkpoint_path: Path, device: torch.device):
    """Reconstruct the pipeline from saved hyperparameters in the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    env = make_env(0)
    obs, _ = env.reset(seed=0)
    agents = list(env.agents)
    obs_dim = obs[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n
    env.close() if hasattr(env, "close") else None

    pipe = AttentionCommPipeline(
        obs_dim=obs_dim,
        msg_dim=args["msg_dim"],
        action_dim=action_dim,
        num_heads=args["num_heads"],
        hidden=args["hidden"],
        dropout=args.get("attn_dropout", 0.0),
    ).to(device)
    pipe.load_state_dict(ckpt["pipeline"])
    pipe.eval()
    return pipe, args, obs_dim, action_dim


def rollout_collect(
    pipe: AttentionCommPipeline,
    n_episodes: int,
    seed: int,
    device: torch.device,
):
    """Run `n_episodes` greedy eval episodes and stack attention weights.

    Returns:
        all_weights: ndarray (T, H, N, N)
        returns: list[float] of per-episode returns
    """
    env = make_env(seed)
    weights_list = []
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        agents = list(env.agents)
        N = len(agents)
        ep_ret = 0.0
        done = False
        while not done:
            obs_arr = np.stack([obs[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, _, weights = pipe(obs_t)
                # Greedy (argmax) for eval
                actions = logits.squeeze(0).argmax(dim=-1)
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            ep_ret += float(np.mean(list(reward_dict.values())))
            done = any(term_dict.values()) or any(trunc_dict.values())
            weights_list.append(weights.squeeze(0).cpu().numpy())  # (H, N, N)
        returns.append(ep_ret)

    all_weights = np.stack(weights_list, axis=0)                      # (T, H, N, N)
    return all_weights, returns


def attention_entropy(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-(timestep, head, receiver) entropy. Returns (T, H, N)."""
    w = np.clip(weights, eps, 1.0)
    return -(w * np.log(w)).sum(axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # [FIX] env config as real CLI args -- never hand-edit / sed this file again.
    # n_agents MUST match the agent count the checkpoint was TRAINED with.
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--agent-neighbors", type=int, default=2)
    parser.add_argument("--landmark-neighbors", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSON path. .npz alongside.")
    # [FIX] the two flags the earlier patch was missing
    parser.add_argument("--num-landmarks", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=0)
    args = parser.parse_args()
    _apply_env_cfg_from_args(args)   # [FIX] set env config + print it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[analyze] device={device} checkpoint={args.checkpoint}")

    pipe, train_args, obs_dim, action_dim = load_pipeline(args.checkpoint, device)
    print(f"[analyze] msg_dim={train_args['msg_dim']} num_heads={train_args['num_heads']}")

    weights, returns = rollout_collect(pipe, args.eval_episodes, args.seed, device)
    T, H, N, _ = weights.shape
    print(f"[analyze] collected {T} steps over {len(returns)} episodes")

    # Statistics
    ent = attention_entropy(weights)                                  # (T, H, N)
    mean_ent_per_head = ent.mean(axis=(0, 2))                         # (H,)
    mean_matrix_overall = weights.mean(axis=(0, 1))                   # (N, N) — averaged over time + heads
    mean_matrix_per_head = weights.mean(axis=0)                       # (H, N, N)

    uniform_ent_ref = float(np.log(max(N - 1, 1)))                    # uniform over non-self

    stats = {
        "checkpoint": str(args.checkpoint),
        "msg_dim": int(train_args["msg_dim"]),
        "num_heads": int(train_args["num_heads"]),
        "num_agents": int(N),
        "eval_episodes": int(args.eval_episodes),
        "eval_steps_total": int(T),
        "mean_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "mean_attn_entropy_overall": float(ent.mean()),
        "mean_attn_entropy_per_head": [float(x) for x in mean_ent_per_head],
        "uniform_attn_entropy_reference": uniform_ent_ref,
        "mean_attn_matrix_overall": mean_matrix_overall.tolist(),
        "interpretation_hint": (
            "mean_attn_matrix_overall[i][j] = how much agent i attended to "
            "agent j on average. Diagonal should be 0 (self-mask). Rows sum to 1."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"[analyze] wrote JSON -> {args.out}")

    npz_path = args.out.with_suffix(".npz")
    np.savez(
        npz_path,
        mean_matrix_overall=mean_matrix_overall,
        mean_matrix_per_head=mean_matrix_per_head,
        entropy_per_step=ent,
        returns=np.array(returns),
    )
    print(f"[analyze] wrote NPZ  -> {npz_path}")

    # Pretty print to terminal
    print("\n--- summary ---")
    print(f"  mean return       : {stats['mean_episode_return']:+.2f} ± {stats['std_episode_return']:.2f}")
    print(f"  mean attn entropy : {stats['mean_attn_entropy_overall']:.3f} (uniform ref = {uniform_ent_ref:.3f})")
    print(f"  per-head entropy  : {['%.3f' % x for x in mean_ent_per_head]}")
    print(f"  mean matrix (rows=receiver, cols=sender):")
    for row in mean_matrix_overall:
        print("    " + "  ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()