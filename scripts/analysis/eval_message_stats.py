"""
Diagnose FC-comm messages from a saved checkpoint.
File: scripts/eval_message_stats.py
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.comms.fc_comm import FCCommPipeline

def evaluate_channel_health(norm: float, std: float) -> str:
    """Evaluates message tensor distributions for representation collapse anomalies."""
    if norm < 1e-4 and std < 1e-4:
        return "DEA — The communication channel has collapsed completely to zero vectors."
    if std < 0.02:
        return "SUSPICIOU— Low message variance detected. Agents may be broadcasting uniform white noise."
    if norm > 40.0:
        return "SUSPICIOUS— Unbounded large magnitudes detected. Risk of exploding gradients."
    return "ALIVE Layer Status: HEALTHY Protocols Emerging! "

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/fccomm_mappo/seed0/final.pt", type=Path)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--n-agents", type=int, default=6)   # [FIX] was 5; project uses 6
    ap.add_argument("--agent-neighbors", type=int, default=2, help="-1 = see all (easy)")
    ap.add_argument("--landmark-neighbors", type=int, default=2, help="-1 = see all (easy)")
    ap.add_argument("--num-landmarks", type=int, default=4)
    ap.add_argument("--num-obstacles", type=int, default=2)
    ap.add_argument("--max-cycles", type=int, default=50)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"[error] Target network checkpoint not located at: {args.checkpoint}")
        return

    # Extract state dictionary layers safely
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[eval] Loading structural parameters from: {args.checkpoint}")

    # Initialize environment configuration matrix
    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if args.agent_neighbors < 0 else args.agent_neighbors
    _ln = None if args.landmark_neighbors < 0 else args.landmark_neighbors
    env = planC_parallel_env(
        N=args.n_agents, num_landmarks=args.num_landmarks, num_obstacles=args.num_obstacles,
        local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=args.max_cycles, render_mode=None,
    )
    obs_dict, _ = env.reset(seed=0)
    agents = list(env.agents)
    N = len(agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    # Reconstruct policy pipeline layers matching checkpoint geometry
    pipeline = FCCommPipeline(obs_dim=obs_dim, msg_dim=8, action_dim=action_dim)
    pipeline.load_state_dict(ckpt["pipeline"])
    pipeline.eval()

    all_msgs = []
    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=ep)
        while env.agents:
            arr = np.stack([obs_dict[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                _, msgs, _ = pipeline(obs_t)

            all_msgs.append(msgs.squeeze(0).cpu().numpy())
            action_dict = {a: env.action_space(a).sample() for a in env.agents}
            obs_dict, _, _, _, _ = env.step(action_dict)

    all_msgs = np.stack(all_msgs, axis=0) # Shape tracking: (Timesteps, Agents, Msg_dim)
    T, _, D = all_msgs.shape

    print(f"[eval] Extracted {T} sequential communication intervals across evaluation episodes.")
    print("\n=== Inter-Agent Message Distributions ===")
    print(f"{'Agent ID':<10} {'Mean Value':>12} {'Std Dev':>12} {'Mean L2 Norm':>14}")
    for i in range(N):
        m = all_msgs[:, i, :]
        l2_norm = np.linalg.norm(m, axis=1).mean()
        print(f"agent_{i:<5} {m.mean():>12.4f} {m.std():>12.4f} {l2_norm:>14.4f}")

    # Evaluate structural distinctiveness across communication lines
    print("\n=== Pairwise Protocol Cosine Similarity ===")
    msgs_norm = all_msgs / (np.linalg.norm(all_msgs, axis=-1, keepdims=True) + 1e-8)
    for i in range(N):
        for j in range(i + 1, N):
            cos_sim = np.mean(np.sum(msgs_norm[:, i] * msgs_norm[:, j], axis=-1))
            print(f"  agent_{i} <-> agent_{j} similarity: {cos_sim:+.4f}")

    overall_norm = float(np.linalg.norm(all_msgs.reshape(-1, D), axis=1).mean())
    overall_std = float(all_msgs.std())
    print("\n=== Channel Diagnostic Verdict ===")
    print(evaluate_channel_health(overall_norm, overall_std))

if __name__ == "__main__":
    main()