"""Stable MINE for Simple Spread communication."""
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
from src.comms.channels import LowRankChannel


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)


def make_env(n_agents=5, num_landmarks=4, agent_neighbors=1, landmark_neighbors=1, num_obstacles=0, seed=0):
    env = planC_parallel_env(
        N=n_agents, num_landmarks=num_landmarks, num_obstacles=num_obstacles,
        local_ratio=0.5, num_agent_neighbors=agent_neighbors,
        num_landmark_neighbors=landmark_neighbors, max_cycles=50, render_mode=None,
    )
    env.reset(seed=seed)
    return env


def build_pipeline(comm_type, obs_dim, action_dim, state_dict, msg_dim=8, hidden=128, num_heads=2, num_layers=2):
    channel = None
    if comm_type == "gnn":
        # If checkpoint has low-rank bandwidth weights, rebuild matching channel
        if any(k.startswith("channel.down") for k in state_dict.keys()):
            # infer bandwidth from weight shape
            bw = state_dict["channel.down.weight"].shape[0]
            channel = LowRankChannel(msg_dim=msg_dim, bandwidth=int(bw))
            print(f"[mine] detected LowRankChannel bandwidth={bw}")
        return GNNCommPipeline(obs_dim, msg_dim, action_dim, num_layers=num_layers, hidden=hidden, channel=channel)
    if comm_type == "fc":
        return FCCommPipeline(obs_dim, msg_dim, action_dim, agg_mode="mean", hidden=hidden)
    if comm_type == "attn":
        return AttentionCommPipeline(obs_dim, msg_dim, action_dim, num_heads=num_heads, hidden=hidden)
    raise ValueError(comm_type)


def extract_msgs(pipeline, obs_t, comm_type):
    out = pipeline(obs_t)
    msgs = out[1]
    return msgs.squeeze(0)


@torch.no_grad()
def collect_pairs(pipeline, env, agents, device, comm_type, episodes=30):
    xs, ys = [], []
    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=3000 + ep)
        while env.agents:
            arr = np.stack([obs_dict[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
            msgs = extract_msgs(pipeline, obs_t, comm_type)
            for i in range(len(agents)):
                xs.append(obs_t[0, i].cpu())
                ys.append(msgs[i].cpu())
            logits = pipeline(obs_t)[0].squeeze(0)
            actions = Categorical(logits=logits).sample()
            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs_dict, _, term, trunc, _ = env.step(action_dict)
            if any(term.values()) or any(trunc.values()):
                break
    return torch.stack(xs), torch.stack(ys)


def estimate_mi(x, y, steps=3000, batch_size=256, lr=1e-4):
    """MINE with EMA of e^T for stability (Belghazi et al.)."""
    device = x.device
    mine = MINE(x.shape[-1], y.shape[-1]).to(device)
    opt = torch.optim.Adam(mine.parameters(), lr=lr)
    n = x.shape[0]
    ma_et = 1.0
    ma_rate = 0.01
    history = []

    for t in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x_b, y_b = x[idx], y[idx]
        y_shuff = y[torch.randperm(n, device=device)[:batch_size]]

        t_joint = mine(x_b, y_b)
        t_marg = mine(x_b, y_shuff)

        et = torch.exp(t_marg.detach()).mean()
        ma_et = (1 - ma_rate) * ma_et + ma_rate * et.item()
        # biased but stable gradient for marginal term
        loss = -(t_joint.mean() - (1.0 / ma_et) * torch.exp(t_marg).mean())

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mine.parameters(), 1.0)
        opt.step()

        if (t + 1) % 300 == 0:
            with torch.no_grad():
                mi = (t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)).item()
            history.append(mi)
            print(f"  mine_step={t+1:4d}  MI_est={mi:.4f}")

    # use median of second half for robustness
    if len(history) >= 4:
        final_mi = float(np.median(history[len(history)//2:]))
    else:
        final_mi = float(np.median(history)) if history else float("nan")
    return final_mi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--comm-type", required=True, choices=["fc", "attn", "gnn"])
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
    p.add_argument("--mine-steps", type=int, default=3000)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["pipeline"] if "pipeline" in ckpt else ckpt

    env = make_env(args.n_agents, args.num_landmarks, args.agent_neighbors,
                   args.landmark_neighbors, args.num_obstacles, seed=0)
    obs_dict, _ = env.reset(seed=0)
    agents = list(env.agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    pipeline = build_pipeline(args.comm_type, obs_dim, action_dim, state,
                              msg_dim=args.msg_dim, hidden=args.hidden,
                              num_heads=args.num_heads, num_layers=args.num_layers).to(device)
    pipeline.load_state_dict(state, strict=False)
    pipeline.eval()

    print(f"\n=== Stable MINE: {args.comm_type} ===")
    x, y = collect_pairs(pipeline, env, agents, device, args.comm_type, episodes=args.episodes)
    x, y = x.to(device), y.to(device)
    print(f"pairs={x.shape[0]} obs_dim={x.shape[1]} msg_dim={y.shape[1]}")
    mi = estimate_mi(x, y, steps=args.mine_steps)
    print(f"\nFinal stable MI estimate I(obs; msg) ≈ {mi:.4f} nats")


if __name__ == "__main__":
    main()