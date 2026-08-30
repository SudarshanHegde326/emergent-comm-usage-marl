
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    raise RuntimeError("Could not locate the repo root.")
sys.path.insert(0, str(REPO_ROOT))

from src.utils.seeding import set_seed                                    
from src.comms.attention_comm import (                                     
    MessageHead,
    MultiHeadAttentionAggregator,
    CommActor,
)
from src.comms.fc_comm import FCAggregator  
from src.comms.gnn_comm import GNNAggregator
from src.envs.sl_env import (                                              
    SL_AGENT_ORDER,
    SL_MAX_ACTION_DIM,
    SL_N_AGENTS,
    SL_PADDED_OBS_DIM,
    action_mask,
    make_sl_env,
    pad_obs_to_max,
)

# ----------------------------- SL pipeline(s) ------------------------------
class SLPipelineNone(nn.Module):
    
    def __init__(self, padded_obs_dim, action_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(padded_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs_NB):
        # obs_NB: (B, N, padded_obs_dim)
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

    def __init__(self, padded_obs_dim, msg_dim, action_dim,
                 num_heads=2, hidden=128, attn_dropout=0.0):
        super().__init__()
        self.msg_head = MessageHead(padded_obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = MultiHeadAttentionAggregator(msg_dim, num_heads=num_heads,
                                                dropout=attn_dropout)
        self.actor = CommActor(padded_obs_dim, msg_dim, action_dim, hidden=hidden)

    def forward(self, obs_NB):
        msgs = self.msg_head(obs_NB)
        agg, _weights = self.agg(msgs)
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
        agg, _per_layer = self.agg(msgs)       
        logits = self.actor(obs_NB, agg)       
        return logits, msgs, agg
    
def build_pipeline(args, padded_obs_dim, action_dim):
    if args.comm_type == "none":
        return SLPipelineNone(padded_obs_dim, action_dim, hidden=args.hidden)
    if args.comm_type == "fc":
        return SLPipelineFC(padded_obs_dim, args.msg_dim, action_dim,
                            hidden=args.hidden)
    if args.comm_type == "attn":
        return SLPipelineAttn(padded_obs_dim, args.msg_dim, action_dim,
                              num_heads=args.num_heads, hidden=args.hidden,
                              attn_dropout=args.attn_dropout)
    if args.comm_type == "gnn":
        return SLPipelineGNN(padded_obs_dim, args.msg_dim, action_dim,
                             num_layers=args.num_layers, hidden=args.hidden)
    raise ValueError(f"unknown comm_type {args.comm_type!r}")


# ----------------------------- Critic (unchanged shape, larger input) -------

class CentralisedCritic(nn.Module):
    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------- Rollout buffer -------------------------------

class RolloutBuffer:
    def __init__(self, T, N, padded_obs_dim, action_dim, device):
        self.T, self.N = T, N
        self.device = device
        self.obs = torch.zeros(T, N, padded_obs_dim, device=device)
        self.global_obs = torch.zeros(T, N * padded_obs_dim, device=device)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, device=device)
        self.rewards = torch.zeros(T, device=device)
        self.dones = torch.zeros(T, device=device)
        self.ptr = 0

    def add(self, obs, global_obs, actions, logprobs, value, reward, done):
        i = self.ptr
        self.obs[i] = obs
        self.global_obs[i] = global_obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.ptr += 1

    def is_full(self): return self.ptr >= self.T
    def reset(self): self.ptr = 0


# ----------------------------- GAE (unchanged) ------------------------------

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


# ----------------------------- Training loop -------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[sl/train] device={device} seed={args.seed} comm={args.comm_type}"
    )

    if args.no_wandb:
        wandb_run = None
    else:
        import wandb
        wandb_run = wandb.init(
            project="msc-marl-comm",
            name=f"sl_{args.comm_type}_seed{args.seed}",
            config=vars(args),
            tags=["w3", "sl", args.comm_type],
        )

    env = make_sl_env(seed=args.seed)
    obs_dict, _ = env.reset(seed=args.seed)
    N = SL_N_AGENTS
    padded_obs_dim = SL_PADDED_OBS_DIM
    action_dim = SL_MAX_ACTION_DIM

    pipeline = build_pipeline(args, padded_obs_dim, action_dim).to(device)
    critic = CentralisedCritic(N * padded_obs_dim, hidden=args.hidden).to(device)
    optim_a = torch.optim.Adam(pipeline.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    buf = RolloutBuffer(args.rollout_len, N, padded_obs_dim, action_dim, device)

    # Pre-build the action mask once on the right device
    mask = action_mask(device=device)   # (N, action_dim)

    total_env_steps = 0
    ep_ret = 0.0
    ep_rets: list[float] = []
    invalid_action_count = 0
    sampled_action_count = 0
    t0 = time.time()

    while total_env_steps < args.total_steps:
        buf.reset()
        while not buf.is_full():
            obs_t = pad_obs_to_max(obs_dict, device=device)    # (N, padded_obs_dim)
            global_obs = obs_t.flatten()                        # (N*padded_obs_dim,)

            with torch.no_grad():
                logits, _, _ = pipeline(obs_t.unsqueeze(0))     # (1, N, action_dim)
                logits = logits.squeeze(0)                       # (N, action_dim)
                logits_masked = logits + mask                    # broadcasts (N, action_dim)
                dist = Categorical(logits=logits_masked)
                actions = dist.sample()                          # (N,)
                logprobs = dist.log_prob(actions)
                value = critic(global_obs.unsqueeze(0)).squeeze(0)


            sampled_action_count += N
            for i, agent in enumerate(SL_AGENT_ORDER):
                from src.envs.sl_env import SL_ACTION_DIMS
                if actions[i].item() >= SL_ACTION_DIMS[agent]:
                    invalid_action_count += 1

            action_dict = {agent: int(actions[i].item())
                           for i, agent in enumerate(SL_AGENT_ORDER)}
            next_obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            team_reward = float(np.mean(list(reward_dict.values())))
            terminated = float(any(term_dict.values()))
            truncated = float(any(trunc_dict.values()))
            done = float(terminated or truncated)  # episode boundary (reset/accounting)

            buf.add(obs_t, global_obs, actions, logprobs, value, team_reward, terminated)

            ep_ret += team_reward
            obs_dict = next_obs_dict
            total_env_steps += 1  # real env transitions

            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs_dict, _ = env.reset()

        with torch.no_grad():
            last_obs_t = pad_obs_to_max(obs_dict, device=device)
            last_value = critic(last_obs_t.flatten().unsqueeze(0)).squeeze(0)
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last_value,
                               args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_per = adv.unsqueeze(1).expand(-1, N)
        ret_per = ret.unsqueeze(1).expand(-1, N)

        flat_obs = buf.obs.reshape(-1, padded_obs_dim)
        flat_global = buf.global_obs
        flat_actions = buf.actions.reshape(-1)
        flat_old_lp = buf.logprobs.reshape(-1)
        flat_adv = adv_per.reshape(-1)
        flat_ret = ret_per.reshape(-1)

        B_total = flat_obs.shape[0]
        idx = torch.arange(B_total, device=device)
        for _ in range(args.ppo_epochs):
            perm = idx[torch.randperm(B_total, device=device)]
            for start in range(0, B_total, args.minibatch_size):
                mb = perm[start:start + args.minibatch_size]
                t_idx = mb // N
                a_idx = mb % N
                unique_t, inverse_t = torch.unique(t_idx, return_inverse=True)
                obs_at_t = buf.obs[unique_t]
                logits_full, _, _ = pipeline(obs_at_t)             # (U, N, action_dim)
                # Build a (U, N, action_dim) mask by broadcasting
                logits_full_masked = logits_full + mask.unsqueeze(0)
                logits_mb = logits_full_masked[inverse_t, a_idx]
                dist = Categorical(logits=logits_mb)
                new_lp = dist.log_prob(flat_actions[mb])
                ent = dist.entropy().mean()
                ratio = (new_lp - flat_old_lp[mb]).exp()
                s1 = ratio * flat_adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * flat_adv[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_pred = critic(flat_global[t_idx])
                value_loss = F.mse_loss(value_pred, flat_ret[mb])
                loss = policy_loss + 0.5 * value_loss - args.ent_coef * ent

                optim_a.zero_grad(set_to_none=True)
                optim_c.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optim_a.step()
                optim_c.step()

        # logging
        mean_ret = float(np.mean(ep_rets[-20:])) if ep_rets else float("nan")
        fps = int(total_env_steps / (time.time() - t0 + 1e-9))
        valid_frac = 1.0 if sampled_action_count == 0 else \
            1.0 - (invalid_action_count / sampled_action_count)

        print(
            f"comm={args.comm_type:>4s}  steps={total_env_steps:>8d}  "
            f"mean_ret={mean_ret:7.2f}  fps={fps}  "
            f"action_valid_frac={valid_frac:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log({
                "step": total_env_steps,
                "mean_return_last20": mean_ret,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": ent.item(),
                "fps": fps,
                "action_valid_frac": valid_frac,
            })

    save_dir = REPO_ROOT / "experiments" / f"sl_{args.comm_type}" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pipeline": pipeline.state_dict(),
            "critic": critic.state_dict(),
            "args": vars(args),
            "action_valid_frac": valid_frac,
        },
        save_dir / "final.pt",
    )
    print(f"[sl/train] saved -> {save_dir}/final.pt")
    if wandb_run is not None:
        wandb_run.finish()


# ----------------------------- CLI -------------------------------------------


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--comm-type", required=True, choices=["none", "fc", "attn", "gnn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout-len", type=int, default=512)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--msg-dim", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--num-layers", type=int, default=2)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.comm_type == "attn" and (args.msg_dim % args.num_heads != 0):
        raise SystemExit(
            f"--msg-dim ({args.msg_dim}) must be divisible by --num-heads "
            f"({args.num_heads}) for attention pipeline."
        )
    train(args)