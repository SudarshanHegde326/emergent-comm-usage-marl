"""
MAPPO with No Communication on PlanC Simple Spread (fixed landmark order).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.seeding import set_seed


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, obs):
        return self.net(obs)


class CentralisedCritic(nn.Module):
    def __init__(self, global_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RolloutBuffer:
    def __init__(self, T, N, obs_dim, device):
        self.T, self.N, self.device = T, N, device
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.global_obs = torch.zeros(T, N * obs_dim, device=device)
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


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[no-comm/train] Device={device}, Seed={args.seed}")

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="msc-marl-comm",
            name=f"nocomm_simple-spread_seed{args.seed}",
            config=vars(args),
            tags=["no-comm", "simple-spread"],
        )

    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if args.agent_neighbors < 0 else args.agent_neighbors
    _ln = None if args.landmark_neighbors < 0 else args.landmark_neighbors
    env = planC_parallel_env(
        N=args.n_agents,
        num_landmarks=args.num_landmarks,
        num_obstacles=args.num_obstacles,
        local_ratio=0.5,
        num_agent_neighbors=_an,
        num_landmark_neighbors=_ln,
        max_cycles=args.max_cycles,
        render_mode=None,
    )
    obs_dict, _ = env.reset(seed=args.seed)
    agents = list(env.agents)
    N = len(agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    actor = Actor(obs_dim, action_dim, hidden=args.hidden).to(device)
    critic = CentralisedCritic(N * obs_dim, hidden=args.hidden).to(device)
    optim_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    buf = RolloutBuffer(args.rollout_len, N, obs_dim, device)
    total_env_steps = 0
    episode_returns = []
    t0 = time.time()
    ep_ret = 0.0

    while total_env_steps < args.total_steps:
        buf.reset()
        while not buf.is_full():
            arr = np.stack([obs_dict[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(arr, dtype=torch.float32, device=device)
            global_obs = obs_t.flatten()

            with torch.no_grad():
                logits = actor(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                value = critic(global_obs.unsqueeze(0)).squeeze(0)

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            next_obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            team_reward = float(np.mean(list(reward_dict.values())))
            done = float(any(term_dict.values()) or any(trunc_dict.values()))

            buf.add(obs_t, global_obs, actions, logprobs, value, team_reward, done)
            total_env_steps += N
            obs_dict = next_obs_dict

            # ---- correct episode return accumulation ----
            if "ep_ret" not in locals():
                ep_ret = 0.0
            ep_ret += team_reward

            if done:
                episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs_dict, _ = env.reset()

        # PPO update
        with torch.no_grad():
            last_obs = torch.as_tensor(
                np.stack([obs_dict[a] for a in agents], axis=0),
                dtype=torch.float32, device=device
            )
            last_value = critic(last_obs.flatten().unsqueeze(0)).squeeze(0)

        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, last_value, args.gamma, args.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            idx = torch.randperm(buf.T)
            for start in range(0, buf.T, args.minibatch_size):
                mb = idx[start:start + args.minibatch_size]
                logits = actor(buf.obs[mb])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(buf.actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - buf.logprobs[mb])
                surr1 = ratio * advantages[mb].unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantages[mb].unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy

                value_pred = critic(buf.global_obs[mb])
                value_loss = ((value_pred - returns[mb]) ** 2).mean()

                optim_a.zero_grad(set_to_none=True)
                policy_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optim_a.step()

                optim_c.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optim_c.step()

        if episode_returns:
            mean_ret = float(np.mean(episode_returns[-20:]))
            fps = total_env_steps / (time.time() - t0 + 1e-8)
            print(f"Steps={total_env_steps:6d} MeanReturn={mean_ret:7.2f} fps={fps:.0f}")
            if not args.no_wandb:
                import wandb
                wandb.log({"mean_return": mean_ret, "steps": total_env_steps})

    out_dir = Path(f"artifacts/checkpoints/nocomm_mappo/seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, out_dir / "final.pt")
    print(f"[no-comm/train] saved -> {out_dir / 'final.pt'}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--n-agents", type=int, default=5)
    p.add_argument("--num-landmarks", type=int, default=4)
    p.add_argument("--agent-neighbors", type=int, default=1)
    p.add_argument("--landmark-neighbors", type=int, default=1)
    p.add_argument("--num-obstacles", type=int, default=0)
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
    p.add_argument("--no-wandb", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)