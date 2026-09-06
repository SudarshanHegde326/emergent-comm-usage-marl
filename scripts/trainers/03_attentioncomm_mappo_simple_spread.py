"""MAPPO with multi-head Attention communication on PettingZoo Simple Spread.
File: scripts/03_attentioncomm_mappo_simple_spread.py

"""

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

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.seeding import set_seed
from src.comms.attention_comm import AttentionCommPipeline

class CentralisedCritic(nn.Module):
    """Centralised value function V(global state) matching FC-Comm baseline exactly."""
    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)

class RolloutBuffer:
    """Stores on-policy transition batches alongside attention routing histories."""
    def __init__(self, T: int, N: int, obs_dim: int, msg_dim: int, num_heads: int, device: torch.device) -> None:
        self.T, self.N = T, N
        self.device = device
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.global_obs = torch.zeros(T, N * obs_dim, device=device)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, device=device)
        self.rewards = torch.zeros(T, device=device)
        self.dones = torch.zeros(T, device=device)
        self.msgs = torch.zeros(T, N, msg_dim, device=device)
        self.attn_weights = torch.zeros(T, num_heads, N, N, device=device) # ATTN DELTA
        self.ptr = 0

    def add(self, obs, global_obs, actions, logprobs, value, reward, done, msgs, attn_weights):
        i = self.ptr
        self.obs[i] = obs
        self.global_obs[i] = global_obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.msgs[i] = msgs
        self.attn_weights[i] = attn_weights
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.T

    def reset(self) -> None:
        self.ptr = 0

def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns

def make_env(seed: int, args):
    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if args.agent_neighbors < 0 else args.agent_neighbors
    _ln = None if args.landmark_neighbors < 0 else args.landmark_neighbors
    env = planC_parallel_env(
        N=args.n_agents, num_landmarks=args.num_landmarks, num_obstacles=args.num_obstacles, local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=args.max_cycles, render_mode=None,
    )
    env.reset(seed=seed)
    return env

def obs_dict_to_tensor(obs_dict, agents, device):
    arr = np.stack([obs_dict[a] for a in agents], axis=0)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)

def attention_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes mean Shannon entropy of communication weights across active channels."""
    w = weights.clamp(min=eps)
    ent = -(w * w.log()).sum(dim=-1)
    return ent.mean()

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[attn-comm/train] device={device}, seed={args.seed}, msg_dim={args.msg_dim}, num_heads={args.num_heads}")

    if args.no_wandb:
        wandb_run = None
    else:
        import wandb
        wandb_run = wandb.init(
            project="msc-marl-comm",
            name=f"attncomm_simple-spread_seed{args.seed}_msg{args.msg_dim}_h{args.num_heads}",
            config=vars(args),
            tags=["w2", "attn-comm", "integration"],
        )

    env = make_env(args.seed, args)
    obs_dict, _ = env.reset(seed=args.seed)
    agents = list(env.agents)
    N = len(agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    pipeline = AttentionCommPipeline(
        obs_dim=obs_dim, msg_dim=args.msg_dim, action_dim=action_dim,
        num_heads=args.num_heads, hidden=args.hidden, dropout=args.attn_dropout
    ).to(device)
    critic = CentralisedCritic(N * obs_dim, hidden=args.hidden).to(device)

    optim_a = torch.optim.Adam(pipeline.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    buf = RolloutBuffer(args.rollout_len, N, obs_dim, args.msg_dim, args.num_heads, device)
    total_env_steps = 0
    episode_return = 0.0
    episode_returns: list[float] = []
    t0 = time.time()

    while total_env_steps < args.total_steps:
        buf.reset()
        while not buf.is_full():
            obs_t = obs_dict_to_tensor(obs_dict, agents, device)
            global_obs = obs_t.flatten()

            with torch.no_grad():
                logits, msgs, agg, weights = pipeline(obs_t.unsqueeze(0)) # 4-TUPLE UNPACK
                logits = logits.squeeze(0)
                msgs = msgs.squeeze(0)
                weights_step = weights.squeeze(0)

                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                value = critic(global_obs.unsqueeze(0)).squeeze(0)

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            next_obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            team_reward = float(np.mean(list(reward_dict.values())))
            terminated = float(any(term_dict.values()))
            truncated = float(any(trunc_dict.values()))
            done = float(terminated or truncated)  # episode boundary (reset/accounting)

            buf.add(obs_t, global_obs, actions, logprobs, value, team_reward, terminated, msgs, weights_step)
            episode_return += team_reward
            obs_dict = next_obs_dict
            total_env_steps += 1  

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs_dict, _ = env.reset()

        with torch.no_grad():
            last_obs_t = obs_dict_to_tensor(obs_dict, agents, device)
            last_global = last_obs_t.flatten()
            last_value = critic(last_global.unsqueeze(0)).squeeze(0)

        advantages, returns = compute_gae(buf.rewards, buf.values, buf.dones, last_value, args.gamma, args.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_per_agent = advantages.unsqueeze(1).expand(-1, N)
        ret_per_agent = returns.unsqueeze(1).expand(-1, N)

        flat_obs = buf.obs.reshape(-1, obs_dim)
        flat_global = buf.global_obs
        flat_actions = buf.actions.reshape(-1)
        flat_old_logprobs = buf.logprobs.reshape(-1)
        flat_adv = adv_per_agent.reshape(-1)
        flat_ret = ret_per_agent.reshape(-1)

        B_size = flat_obs.shape[0]
        idx = torch.arange(B_size, device=device)
        for _ in range(args.ppo_epochs):
            perm = idx[torch.randperm(B_size, device=device)]
            for start in range(0, B_size, args.minibatch_size):
                mb = perm[start:start + args.minibatch_size]
                t_idx = mb // N
                a_idx = mb % N

                unique_t, inverse_t = torch.unique(t_idx, return_inverse=True)
                obs_at_t = buf.obs[unique_t]
                logits_full, _, _, _ = pipeline(obs_at_t) # 4-TUPLE GRAPH RECOMPUTE
                logits_mb = logits_full[inverse_t, a_idx]

                dist = Categorical(logits=logits_mb)
                new_logp = dist.log_prob(flat_actions[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - flat_old_logprobs[mb]).exp()
                surr1 = ratio * flat_adv[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * flat_adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = critic(flat_global[t_idx])
                value_loss = F.mse_loss(value_pred, flat_ret[mb])

                loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy

                optim_a.zero_grad(set_to_none=True)
                optim_c.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optim_a.step()
                optim_c.step()

        mean_ret = float(np.mean(episode_returns[-20:])) if episode_returns else float("nan")
        fps = int(total_env_steps / (time.time() - t0 + 1e-9))
        with torch.no_grad():
            msg_norm = buf.msgs.norm(dim=-1).mean().item()
            msg_std = buf.msgs.std().item()
            attn_ent = attention_entropy(buf.attn_weights).item()
            uniform_ref = float(np.log(max(N - 1, 1)))

        print(f"steps={total_env_steps:>8d}  mean_ret={mean_ret:7.2f}  fps={fps}  |msg|={msg_norm:.3f}  attn_ent={attn_ent:.3f}/{uniform_ref:.3f}")
        if wandb_run is not None:
            wandb_run.log({
                "step": total_env_steps, "mean_return_last20": mean_ret, "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(), "entropy": entropy.item(), "fps": fps,
                "msg_norm_mean": msg_norm, "msg_std": msg_std, "attn_entropy": attn_ent
            })

    save_dir = REPO_ROOT / "experiments" / "attncomm_mappo" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pipeline": pipeline.state_dict(), "critic": critic.state_dict(), "args": vars(args)}, save_dir / "final.pt")
    print(f"[attn-comm/train] saved checkpoint to {save_dir}/final.pt")
    if wandb_run is not None:
        wandb_run.finish()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--n-agents", type=int, default=6, help="3=easy, 5+=harder")
    p.add_argument("--agent-neighbors", type=int, default=2, help="-1 = see all")
    p.add_argument("--landmark-neighbors", type=int, default=2, help="-1 = see all")
    p.add_argument("--num-landmarks", type=int, default=4)
    p.add_argument("--num-obstacles", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=50)
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
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.msg_dim % args.num_heads != 0:
        raise SystemExit(f"--msg-dim ({args.msg_dim}) must be divisible by --num-heads ({args.num_heads}).")
    train(args)