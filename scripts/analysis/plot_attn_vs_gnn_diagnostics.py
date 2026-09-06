"""2-panel diagnostic figure for the attention-comm vs GNN-comm
deep comparison.

Left panel:  attention entropy per head per receiver, averaged
             across the 3 attention-comm seeds + 20 eval episodes.
Right panel: per-layer L2 norm trajectory, averaged across the 3
             GNN-comm seeds + 20 eval episodes.

Both panels share the same eval-episode count, eval-seed offset,
and (B, N, D) shape conventions, so the comparison is fair.

USAGE
-----
    python scripts/plot_attn_vs_gnn_diagnostics.py \\
        --attn-checkpoints artifacts/checkpoints/attncomm_official/seed{0,1,2}/final.pt \\
        --gnn-checkpoints  artifacts/checkpoints/gnncomm_mappo/seed{0,1,2}/final.pt \\
        --eval-episodes 20 \\
        --out figures/attn_vs_gnn_diagnostics.png \\
        --out-pdf figures/attn_vs_gnn_diagnostics.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def load_attn_pipeline(checkpoint_path: Path, device: torch.device):
    from src.comms.attention_comm import AttentionCommPipeline
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
    return pipe


def load_gnn_pipeline(checkpoint_path: Path, device: torch.device):
    from src.comms.gnn_comm import GNNCommPipeline
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    env = make_env(0)
    obs, _ = env.reset(seed=0)
    agents = list(env.agents)
    obs_dim = obs[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n
    pipe = GNNCommPipeline(
        obs_dim=obs_dim,
        msg_dim=args["msg_dim"],
        action_dim=action_dim,
        num_layers=args["num_layers"],
        hidden=args["hidden"],
        channel=None,
    ).to(device)
    pipe.load_state_dict(ckpt["pipeline"])
    pipe.eval()
    return pipe


# ---------------------------------------------------------------------------
# Diagnostic collection
# ---------------------------------------------------------------------------


def collect_attn_entropy(pipe, n_episodes: int, seed_offset: int,
                         device: torch.device, eps: float = 1e-12):
    """Per-step per-head per-receiver entropy of attention weights,
    averaged across (T, H, N) over n_episodes eval rollouts.
    Returns (per_head_entropy: (H,), per_receiver_entropy: (N,))."""
    env = make_env(seed_offset)
    all_w = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        agents = list(env.agents)
        done = False
        while not done:
            obs_arr = np.stack([obs[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, _, weights = pipe(obs_t)        # weights: (1, H, N, N)
                actions = logits.squeeze(0).argmax(dim=-1)
            all_w.append(weights.squeeze(0).cpu().numpy())  # (H, N, N)
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs, _, term, trunc, _ = env.step(action_dict)
            done = any(term.values()) or any(trunc.values())
    w = np.stack(all_w, axis=0)                              # (T, H, N, N)
    w = np.clip(w, eps, 1.0)
    ent = -(w * np.log(w)).sum(axis=-1)                       # (T, H, N)
    return ent.mean(axis=(0, 2)), ent.mean(axis=(0, 1))      # (H,), (N,)


def collect_per_layer_l2(pipe, n_episodes: int, seed_offset: int,
                         device: torch.device):
    """Per-layer L2 norm trajectory averaged across (T, N) over
    n_episodes eval rollouts.
    Returns: ndarray of shape (L+1,) for L = pipe.num_layers."""
    env = make_env(seed_offset)
    L = pipe.num_layers
    layer_l2_sum = np.zeros(L + 1, dtype=np.float64)
    count = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        agents = list(env.agents)
        done = False
        while not done:
            obs_arr = np.stack([obs[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, _, _, per_layer = pipe(obs_t)
                actions = logits.squeeze(0).argmax(dim=-1)
            for l_idx, h in enumerate(per_layer):
                layer_l2_sum[l_idx] += h.squeeze(0).norm(dim=-1).mean().item()
            count += 1
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs, _, term, trunc, _ = env.step(action_dict)
            done = any(term.values()) or any(trunc.values())
    return layer_l2_sum / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn-checkpoints", nargs="+", required=True, type=Path)
    p.add_argument("--gnn-checkpoints", nargs="+", required=True, type=Path)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=12345)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--out-pdf", default=None, type=Path)
    # [FIX] env config as REAL flags. Must match the training config.
    p.add_argument("--n-agents", type=int, default=6)
    p.add_argument("--agent-neighbors", type=int, default=2)
    p.add_argument("--landmark-neighbors", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=50)
    p.add_argument("--num-landmarks", type=int, default=4)
    p.add_argument("--num-obstacles", type=int, default=0)
    args = p.parse_args()
    _apply_env_cfg_from_args(args)   # [FIX] set env config + print it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}")

    # --- attention entropy across the 3 attn checkpoints ----------------
    attn_per_head_list = []
    attn_per_recv_list = []
    for ck in args.attn_checkpoints:
        if not ck.exists():
            print(f"  SKIP missing attn checkpoint: {ck}")
            continue
        pipe = load_attn_pipeline(ck, device)
        per_h, per_r = collect_attn_entropy(pipe, args.eval_episodes,
                                            args.seed_offset, device)
        print(f"  attn {ck.name}: per-head ent = {per_h}, per-recv ent = {per_r}")
        attn_per_head_list.append(per_h)
        attn_per_recv_list.append(per_r)
    if not attn_per_head_list:
        raise SystemExit("No attention checkpoints loaded.")
    attn_per_head = np.stack(attn_per_head_list, axis=0).mean(axis=0)  # (H,)
    attn_per_recv = np.stack(attn_per_recv_list, axis=0).mean(axis=0)  # (N,)
    uniform_ent_ref = float(np.log(max(attn_per_recv.shape[0] - 1, 1)))

    # --- per-layer L2 across the 3 GNN checkpoints ---------------------
    gnn_l2_list = []
    for ck in args.gnn_checkpoints:
        if not ck.exists():
            print(f"  SKIP missing gnn checkpoint: {ck}")
            continue
        pipe = load_gnn_pipeline(ck, device)
        l2 = collect_per_layer_l2(pipe, args.eval_episodes,
                                  args.seed_offset, device)
        print(f"  gnn  {ck.name}: per-layer L2 = {l2}")
        gnn_l2_list.append(l2)
    if not gnn_l2_list:
        raise SystemExit("No GNN checkpoints loaded.")
    gnn_per_layer = np.stack(gnn_l2_list, axis=0).mean(axis=0)         # (L+1,)
    gnn_per_layer_std = np.stack(gnn_l2_list, axis=0).std(axis=0, ddof=1)

    # --- plot ---------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: attention entropy per head (cross-seed mean)
    H = attn_per_head.shape[0]
    axes[0].bar(np.arange(H), attn_per_head, color="#009E73", alpha=0.85,
                edgecolor="black", linewidth=0.7)
    axes[0].axhline(uniform_ent_ref, linestyle="--", color="gray",
                    label=f"uniform ref = log({attn_per_recv.shape[0] - 1}) = "
                          f"{uniform_ent_ref:.2f}")
    axes[0].set_xticks(np.arange(H))
    axes[0].set_xlabel("attention head")
    axes[0].set_ylabel("mean attention entropy (nats)")
    axes[0].set_title("Attention-comm — per-head entropy\n"
                      "(lower = more selective)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)

    # Right: per-layer L2 trajectory (cross-seed mean ± std)
    L = gnn_per_layer.shape[0]
    axes[1].errorbar(np.arange(L), gnn_per_layer,
                     yerr=gnn_per_layer_std, capsize=6,
                     marker="o", markersize=8, color="#D55E00",
                     markerfacecolor="white",
                     markeredgecolor="#D55E00", markeredgewidth=1.5)
    axes[1].set_xticks(np.arange(L))
    axes[1].set_xticklabels([f"h^({i})" for i in range(L)])
    axes[1].set_xlabel("aggregator layer")
    axes[1].set_ylabel("mean per-layer L2 norm")
    axes[1].set_title("GNN-comm — per-layer L2 trajectory\n"
                      "(collapsing = over-smoothing)", fontsize=11)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)

    fig.suptitle(
        "Attention vs GNN — channel-content diagnostics (Simple Spread, N=3, 3 seeds × 20 eval episodes)",
        fontsize=11, y=1.02,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"[diag] saved -> {args.out}")
    if args.out_pdf is not None:
        plt.savefig(args.out_pdf)
        print(f"[diag] saved -> {args.out_pdf}")

    # Console summary
    print("\n--- summary ---")
    print(f"  attention per-head entropy (cross-seed mean): "
          f"{attn_per_head.tolist()}")
    print(f"  attention uniform-(N-1) reference:           {uniform_ent_ref:.3f}")
    print(f"  GNN per-layer L2 trajectory (cross-seed mean): "
          f"{gnn_per_layer.tolist()}")
    print(f"  GNN per-layer L2 trajectory (cross-seed std):  "
          f"{gnn_per_layer_std.tolist()}")


if __name__ == "__main__":
    main()