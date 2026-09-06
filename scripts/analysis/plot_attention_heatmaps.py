"""Plot attention heatmaps for a trained AttentionCommPipeline checkpoint.

Produces a figure with:
  - Top-left: mean attention matrix averaged over all heads (the "headline" matrix)
  - Right of it: one heatmap per head (head specialisation view)
  - Bottom-left: per-head attention entropy (bar chart)
  - Bottom-right: per-receiver mean entropy across heads (bar chart)

Rows of every heatmap = RECEIVER agent (who is attending).
Columns of every heatmap = SENDER agent (who is being attended to).
Self-mask makes the diagonal zero by construction.

USAGE
-----
    python plot_attention_heatmaps.py \
        --checkpoint artifacts/checkpoints/attncomm_mappo/seed0/final.pt \
        --eval-episodes 20 \
        --out figures/day11_attn_heatmap_seed0.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- path setup -------------------------------------------------------------
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

from src.comms.attention_comm import AttentionCommPipeline  # noqa: E402



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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    env = make_env(0)
    obs, _ = env.reset(seed=0)
    agents = list(env.agents)
    obs_dim = obs[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n
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
    return pipe, args


def rollout_collect(pipe, n_episodes, seed, device):
    env = make_env(seed)
    weights_list = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        agents = list(env.agents)
        done = False
        while not done:
            obs_arr = np.stack([obs[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, _, weights = pipe(obs_t)
                actions = logits.squeeze(0).argmax(dim=-1)
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            done = any(term_dict.values()) or any(trunc_dict.values())
            weights_list.append(weights.squeeze(0).cpu().numpy())
    return np.stack(weights_list, axis=0)   # (T, H, N, N)


def attention_entropy(w: np.ndarray, eps=1e-12) -> np.ndarray:
    w = np.clip(w, eps, 1.0)
    return -(w * np.log(w)).sum(axis=-1)


def plot(weights: np.ndarray, out_path: Path, checkpoint_label: str):
    T, H, N, _ = weights.shape
    mean_overall = weights.mean(axis=(0, 1))                # (N, N)
    mean_per_head = weights.mean(axis=0)                    # (H, N, N)
    ent = attention_entropy(weights)                        # (T, H, N)
    ent_per_head = ent.mean(axis=(0, 2))                    # (H,)
    ent_per_receiver = ent.mean(axis=(0, 1))                # (N,)

    # Layout: top row = overall heatmap + per-head heatmaps
    #         bottom row = two bar charts
    fig = plt.figure(figsize=(3 * (H + 1) + 0.5, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, H + 1, height_ratios=[3, 2])

    # --- Top-left: overall mean attention matrix ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mean_overall, vmin=0.0, vmax=mean_overall.max(), cmap="viridis")
    ax.set_title("mean attention\n(avg over heads)", fontsize=10)
    ax.set_xlabel("sender j")
    ax.set_ylabel("receiver i")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mean_overall[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mean_overall[i, j] < 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- Top-row remaining: one heatmap per head ---------------------------
    for h in range(H):
        ax = fig.add_subplot(gs[0, 1 + h])
        m = mean_per_head[h]
        im = ax.imshow(m, vmin=0.0, vmax=max(m.max(), 1e-3), cmap="viridis")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xlabel("sender j")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{m[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if m[i, j] < 0.5 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # --- Bottom-left: entropy per head -------------------------------------
    ax = fig.add_subplot(gs[1, 0:max(1, (H + 1) // 2)])
    ax.bar(range(H), ent_per_head)
    ax.axhline(np.log(max(N - 1, 1)), linestyle="--",
               label=f"uniform ref = log({N - 1}) = {np.log(max(N - 1, 1)):.2f}")
    ax.set_xticks(range(H))
    ax.set_xlabel("head")
    ax.set_ylabel("mean entropy")
    ax.set_title("attention entropy per head")
    ax.legend(loc="upper right", fontsize=8)

    # --- Bottom-right: entropy per receiver --------------------------------
    ax = fig.add_subplot(gs[1, max(1, (H + 1) // 2):])
    ax.bar(range(N), ent_per_receiver, color="C2")
    ax.axhline(np.log(max(N - 1, 1)), linestyle="--", color="gray")
    ax.set_xticks(range(N))
    ax.set_xlabel("receiver agent i")
    ax.set_ylabel("mean entropy")
    ax.set_title("attention entropy per receiver")

    fig.suptitle(f"Attention diagnostics — {checkpoint_label}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    # [FIX] env config as REAL flags. Must match the training config.
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--agent-neighbors", type=int, default=2)
    parser.add_argument("--landmark-neighbors", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--num-landmarks", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=0)
    args = parser.parse_args()
    _apply_env_cfg_from_args(args)   # [FIX] set env config + print it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe, train_args = load_pipeline(args.checkpoint, device)
    label = f"msg={train_args['msg_dim']}, heads={train_args['num_heads']}"
    weights = rollout_collect(pipe, args.eval_episodes, args.seed, device)
    plot(weights, args.out, label)


if __name__ == "__main__":
    main()