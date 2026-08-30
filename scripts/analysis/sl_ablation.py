"""Message Ablation for Simple Speaker-Listener."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# path setup
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.envs.sl_env import (
    SL_AGENT_ORDER, SL_MAX_ACTION_DIM, SL_N_AGENTS, SL_PADDED_OBS_DIM,
    action_mask, make_sl_env, pad_obs_to_max,
)
from src.comms.attention_comm import MessageHead, MultiHeadAttentionAggregator, CommActor
from src.comms.fc_comm import FCAggregator
from src.comms.gnn_comm import GNNAggregator


# -------------------- Pipelines --------------------

class SLPipelineNone(nn.Module):
    def __init__(self, padded_obs_dim, action_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(padded_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, obs_NB):
        logits = self.actor(obs_NB)
        return logits, None, None


class SLPipelineFC(nn.Module):
    def __init__(self, padded_obs_dim, msg_dim, action_dim, hidden=128):
        super().__init__()
        self.msg_head = MessageHead(padded_obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = FCAggregator(mode="mean")
        self.actor = CommActor(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    def forward(self, obs_NB):
        msgs = self.msg_head(obs_NB)
        agg = self.agg(msgs)
        logits = self.actor(obs_NB, agg)
        return logits, msgs, agg


class SLPipelineAttn(nn.Module):
    def __init__(self, padded_obs_dim, msg_dim, action_dim, num_heads=2, hidden=128, attn_dropout=0.0):
        super().__init__()
        self.msg_head = MessageHead(padded_obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = MultiHeadAttentionAggregator(msg_dim, num_heads=num_heads, dropout=attn_dropout)
        self.actor = CommActor(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    def forward(self, obs_NB):
        msgs = self.msg_head(obs_NB)
        agg, _ = self.agg(msgs)
        logits = self.actor(obs_NB, agg)
        return logits, msgs, agg


class SLPipelineGNN(nn.Module):
    def __init__(self, padded_obs_dim, msg_dim, action_dim, num_layers=2, hidden=128):
        super().__init__()
        self.msg_head = MessageHead(padded_obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = GNNAggregator(msg_dim, num_layers=num_layers, hidden=msg_dim)
        self.actor = CommActor(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    def forward(self, obs_NB):
        msgs = self.msg_head(obs_NB)
        agg, _ = self.agg(msgs)
        logits = self.actor(obs_NB, agg)
        return logits, msgs, agg


def build_pipeline(comm_type, padded_obs_dim, action_dim, msg_dim=8, hidden=128):
    if comm_type == "none":
        return SLPipelineNone(padded_obs_dim, action_dim, hidden=hidden)
    if comm_type == "fc":
        return SLPipelineFC(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    if comm_type == "attn":
        return SLPipelineAttn(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    if comm_type == "gnn":
        return SLPipelineGNN(padded_obs_dim, msg_dim, action_dim, hidden=hidden)
    raise ValueError(f"unknown comm_type {comm_type}")


# -------------------- Evaluation --------------------

def evaluate(pipeline, env, device, episodes=50, zero_messages=False):
    returns = []
    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=ep)
        ep_ret = 0.0
        while env.agents:
            obs_t = pad_obs_to_max(obs_dict, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, msgs, agg = pipeline(obs_t)

                if zero_messages and agg is not None:
                    zeros = torch.zeros_like(agg)
                    logits = pipeline.actor(obs_t, zeros)

                # Apply action mask
                mask = action_mask(device=device).unsqueeze(0)
                logits = logits + mask   # mask contains 0 for valid and -inf for invalid

                dist = Categorical(logits=logits)
                actions = dist.sample()

            action_dict = {a: int(actions[0, i].item()) for i, a in enumerate(SL_AGENT_ORDER)}
            obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
            ep_ret += sum(rewards.values()) / max(len(rewards), 1)
            if all(terms.values()) or all(truncs.values()):
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--comm-type", type=str, required=True, choices=["none", "fc", "attn", "gnn"])
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    pipeline = build_pipeline(args.comm_type, SL_PADDED_OBS_DIM, SL_MAX_ACTION_DIM).to(device)
    pipeline.load_state_dict(ckpt["pipeline"])
    pipeline.eval()

    env = make_sl_env()

    print(f"\n=== Evaluating {args.comm_type} ===")
    mean_normal, std_normal = evaluate(pipeline, env, device, args.episodes, zero_messages=False)
    print(f"Normal (with messages) : {mean_normal:.2f} ± {std_normal:.2f}")

    if args.comm_type != "none":
        mean_zero, std_zero = evaluate(pipeline, env, device, args.episodes, zero_messages=True)
        print(f"Ablation (messages=0)  : {mean_zero:.2f} ± {std_zero:.2f}")
        print(f"Performance drop       : {mean_normal - mean_zero:.2f}")
    else:
        print("No-comm has no messages to ablate.")


if __name__ == "__main__":
    main()