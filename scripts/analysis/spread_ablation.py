"""Message Ablation for Simple Spread (FC / Attention / GNN)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.envs.simple_spread_planC import planC_parallel_env
from src.comms.fc_comm import FCCommPipeline
from src.comms.attention_comm import AttentionCommPipeline
from src.comms.gnn_comm import GNNCommPipeline


def make_env(n_agents=5, num_landmarks=4, agent_neighbors=1, landmark_neighbors=1, num_obstacles=0, max_cycles=50, seed=0):
    env = planC_parallel_env(
        N=n_agents,
        num_landmarks=num_landmarks,
        num_obstacles=num_obstacles,
        local_ratio=0.5,
        num_agent_neighbors=agent_neighbors,
        num_landmark_neighbors=landmark_neighbors,
        max_cycles=max_cycles,
        render_mode=None,
    )
    env.reset(seed=seed)
    return env


def build_pipeline(comm_type, obs_dim, action_dim, msg_dim=8, hidden=128, num_heads=2, num_layers=2):
    if comm_type == "fc":
        return FCCommPipeline(obs_dim, msg_dim, action_dim, agg_mode="mean", hidden=hidden)
    if comm_type == "attn":
        return AttentionCommPipeline(obs_dim, msg_dim, action_dim, num_heads=num_heads, hidden=hidden, dropout=0.0)
    if comm_type == "gnn":
        return GNNCommPipeline(obs_dim, msg_dim, action_dim, num_layers=num_layers, hidden=hidden)
    raise ValueError(comm_type)


def forward_with_optional_zero(pipeline, obs_t, comm_type, zero_messages=False):
    """obs_t: (1, N, obs_dim)"""
    if comm_type == "fc":
        # FC: logits, msgs, agg
        if not zero_messages:
            logits, msgs, agg = pipeline(obs_t)
            return logits
        # Ablation: zero the aggregated message contribution
        h = pipeline.feat(obs_t)
        msgs = pipeline.msg_head(obs_t)
        agg = torch.zeros_like(msgs)
        z = pipeline.norm(h + torch.tanh(pipeline.transform(agg)))
        logits = pipeline.actor(z)
        return logits

    if comm_type == "attn":
        logits, msgs, agg, weights = pipeline(obs_t)
        if zero_messages:
            zeros = torch.zeros_like(agg)
            logits = pipeline.actor(obs_t, zeros)
        return logits

    if comm_type == "gnn":
        logits, msgs, msgs_post, agg, per_layer = pipeline(obs_t)
        if zero_messages:
            zeros = torch.zeros_like(agg)
            logits = pipeline.actor(obs_t, zeros)
        return logits

    raise ValueError(comm_type)


def evaluate(pipeline, env, agents, device, comm_type, episodes=30, zero_messages=False):
    returns = []
    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=1000 + ep)
        ep_ret = 0.0
        while env.agents:
            arr = np.stack([obs_dict[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits = forward_with_optional_zero(pipeline, obs_t, comm_type, zero_messages=zero_messages)
                logits = logits.squeeze(0)
                dist = Categorical(logits=logits)
                actions = dist.sample()

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            ep_ret += float(np.mean(list(reward_dict.values())))
            if any(term_dict.values()) or any(trunc_dict.values()):
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--comm-type", type=str, required=True, choices=["fc", "attn", "gnn"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--n-agents", type=int, default=5)
    p.add_argument("--num-landmarks", type=int, default=4)
    p.add_argument("--agent-neighbors", type=int, default=1)
    p.add_argument("--landmark-neighbors", type=int, default=1)
    p.add_argument("--num-obstacles", type=int, default=0)
    p.add_argument("--msg-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    env = make_env(
        n_agents=args.n_agents,
        num_landmarks=args.num_landmarks,
        agent_neighbors=args.agent_neighbors,
        landmark_neighbors=args.landmark_neighbors,
        num_obstacles=args.num_obstacles,
        seed=0,
    )
    obs_dict, _ = env.reset(seed=0)
    agents = list(env.agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    pipeline = build_pipeline(
        args.comm_type, obs_dim, action_dim,
        msg_dim=args.msg_dim, hidden=args.hidden,
        num_heads=args.num_heads, num_layers=args.num_layers,
    ).to(device)

    state = ckpt["pipeline"] if "pipeline" in ckpt else ckpt
    pipeline.load_state_dict(state)
    pipeline.eval()

    print(f"\n=== Ablation: {args.comm_type} | checkpoint={args.checkpoint} ===")
    mean_n, std_n = evaluate(pipeline, env, agents, device, args.comm_type, args.episodes, zero_messages=False)
    print(f"Normal (with messages) : {mean_n:.2f} ± {std_n:.2f}")

    mean_z, std_z = evaluate(pipeline, env, agents, device, args.comm_type, args.episodes, zero_messages=True)
    print(f"Ablation (messages=0)  : {mean_z:.2f} ± {std_z:.2f}")
    print(f"Performance drop       : {mean_n - mean_z:.2f}")


if __name__ == "__main__":
    main()