"""MAPPO training with attention-comm routed through an AWGN noise channel."""

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
from src.comms.attention_comm import MessageHead, MultiHeadAttentionAggregator, CommActor
from src.comms.noise_channel import NoiseChannel


class NoiseAttentionPipeline(nn.Module):
    """Attention pipeline injecting Additive White Gaussian Noise (AWGN)."""

    def __init__(self, obs_dim: int, msg_dim: int, action_dim: int,
                 num_heads: int = 2, hidden: int = 128,
                 sigma: float = 0.0, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.msg_head = MessageHead(obs_dim, msg_dim, hidden=hidden // 2)
        self.channel = NoiseChannel(msg_dim=msg_dim, sigma=sigma)
        self.agg = MultiHeadAttentionAggregator(msg_dim, num_heads=num_heads, dropout=attn_dropout)
        self.actor = CommActor(obs_dim, msg_dim, action_dim, hidden=hidden)

        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.sigma = sigma

    def forward(self, obs_NB: torch.Tensor):
        msgs_pre = self.msg_head(obs_NB)
        z = self.channel(msgs_pre)               # Inject AWGN channel noise
        agg, weights = self.agg(z)
        logits = self.actor(obs_NB, agg)
        return logits, msgs_pre, z, agg, weights


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


class RolloutBuffer:
    def __init__(self, T, N, obs_dim, msg_dim, num_heads, device):
        self.T, self.N = T, N
        self.device = device
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.global_obs = torch.zeros(T, N * obs_dim, device=device)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, device=device)
        self.rewards = torch.zeros(T, device=device)
        self.dones = torch.zeros(T, device=device)
        self.msgs_pre = torch.zeros(T, N, msg_dim, device=device)
        self.zs = torch.zeros(T, N, msg_dim, device=device)
        self.attn_weights = torch.zeros(T, num_heads, N, N, device=device)
        self.ptr = 0

    def add(self, obs, global_obs, actions, logprobs, value, reward, done, msgs_pre, zs, attn_weights):
        i = self.ptr
        self.obs[i] = obs
        self.global_obs[i] = global_obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.msgs_pre[i] = msgs_pre
        self.zs[i] = zs
        self.attn_weights[i] = attn_weights
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


def make_env(seed, args):
    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if args.agent_neighbors < 0 else args.agent_neighbors
    _ln = None if args.landmark_neighbors < 0 else args.landmark_neighbors
    env = planC_parallel_env(
        N=args.n_agents, num_landmarks=args.num_landmarks, num_obstacles=args.num_obstacles, local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=args.max_cycles, continuous_actions=False,
    )
    env.reset(seed=seed)
    return env


def obs_dict_to_tensor(obs_dict, agents, device):
    arr = np.stack([obs_dict[a] for a in agents], axis=0)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def attention_entropy(weights, eps=1e-12):
    w = weights.clamp(min=eps)
    return -(w * w.log()).sum(dim=-1).mean()


def sigma_tag(sigma: float) -> str:
    s = round(sigma * 100)
    return f"s{s:03d}"


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[noise-attn/train] device={device} seed={args.seed} sigma={args.sigma}")

    env = make_env(args.seed, args)
    obs_dict, _ = env.reset(seed=args.seed)
    agents = list(env.agents)
    N = len(agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    pipeline = NoiseAttentionPipeline(
        obs_dim=obs_dim, msg_dim=args.msg_dim, action_dim=action_dim,
        num_heads=args.num_heads, hidden=args.hidden, sigma=args.sigma,
        attn_dropout=args.attn_dropout
    ).to(device)
    critic = CentralisedCritic(N * obs_dim, hidden=args.hidden).to(device)
    optim_a = torch.optim.Adam(pipeline.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    buf = RolloutBuffer(args.rollout_len, N, obs_dim, args.msg_dim, args.num_heads, device)
    total_env_steps = 0
    ep_ret = 0.0
    ep_rets: list[float] = []
    t0 = time.time()

    while total_env_steps < args.total_steps:
        buf.reset()
        while not buf.is_full():
            obs_t = obs_dict_to_tensor(obs_dict, agents, device)
            global_obs = obs_t.flatten()
            with torch.no_grad():
                logits, msgs_pre, z, _, weights = pipeline(obs_t.unsqueeze(0))
                logits = logits.squeeze(0)
                msgs_pre_s = msgs_pre.squeeze(0)
                z_s = z.squeeze(0)
                weights_s = weights.squeeze(0)

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

            buf.add(obs_t, global_obs, actions, logprobs, value, team_reward, terminated, msgs_pre_s, z_s, weights_s)
            ep_ret += team_reward
            obs_dict = next_obs_dict
            total_env_steps += 1  # real env transitions

            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs_dict, _ = env.reset()

        with torch.no_grad():
            last_obs_t = obs_dict_to_tensor(obs_dict, agents, device)
            last_value = critic(last_obs_t.flatten().unsqueeze(0)).squeeze(0)
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last_value, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_per = adv.unsqueeze(1).expand(-1, N)
        ret_per = ret.unsqueeze(1).expand(-1, N)

        flat_obs = buf.obs.reshape(-1, obs_dim)
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
                
                # Recompute pass draws fresh noise samples to properly mirror AWGN gradients
                logits_full, _, _, _, _ = pipeline(buf.obs[unique_t])
                logits_mb = logits_full[inverse_t, a_idx]
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

        mean_ret = float(np.mean(ep_rets[-20:])) if ep_rets else float("nan")
        fps = int(total_env_steps / (time.time() - t0 + 1e-9))
        with torch.no_grad():
            msg_pre_l2 = buf.msgs_pre.norm(dim=-1).mean().item()
            z_l2 = buf.zs.norm(dim=-1).mean().item()
            noise_l2 = (buf.zs - buf.msgs_pre).norm(dim=-1).mean().item()
            expected_noise_l2 = args.sigma * (args.msg_dim ** 0.5)

        print(f"steps={total_env_steps:>8d}  mean_ret={mean_ret:7.2f}  fps={fps}  |msg|={msg_pre_l2:.3f}  |z|={z_l2:.3f}  |z-msg|_l2={noise_l2:.3f}/{expected_noise_l2:.3f}")

    save_dir = REPO_ROOT / "experiments" / f"attncomm_noise_{sigma_tag(args.sigma)}" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pipeline": pipeline.state_dict(), "critic": critic.state_dict(), "args": vars(args)}, save_dir / "final.pt")
    print(f"[noise-attn/train] saved -> {save_dir}/final.pt")


def build_argparser():
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
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--no-wandb", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)