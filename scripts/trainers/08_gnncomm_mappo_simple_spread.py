"""MAPPO with **vanilla GNN-comm** (multi-hop + uniform mean) on PettingZoo Simple Spread.

Near-twin of `03_attentioncomm_mappo_simple_spread.py` (Day 10). The
three changes vs that trainer are tagged `# DIFF FROM attn-comm` and
documented in `docs/gnn_integration.md`:

  1. Import GNNCommPipeline (not AttentionCommPipeline).
  2. 5-tuple unpack on every pipeline forward (not 4-tuple).
  3. Per-layer L2 logging (replaces attention-entropy).

The pipeline is byte-equivalent to spec-v2 (Day 22): vanilla GNN with
fixed FC graph, mean aggregator, residual + LayerNorm, IdentityChannel
default for the D8 slot.

USAGE
-----
    # Smoke
    python 08_gnncomm_mappo_simple_spread.py \
        --seed 999 --total-steps 2000 --no-wandb

    # Full first run
    python 08_gnncomm_mappo_simple_spread.py \
        --seed 0 --total-steps 200000

OUTPUTS
-------
- wandb run (unless --no-wandb)
- artifacts/checkpoints/gnncomm_mappo/seed{N}/final.pt
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

# --- path setup -------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
candidate_roots = [THIS_FILE.parent.parent.parent, THIS_FILE.parent.parent]
REPO_ROOT = None
for root in candidate_roots:
    if (root / "src" / "comms" / "gnn_comm.py").exists():
        REPO_ROOT = root
        break
    if (root / "repo" / "src" / "comms" / "gnn_comm.py").exists():
        REPO_ROOT = root / "repo"
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate the repo root.")
sys.path.insert(0, str(REPO_ROOT))

from src.utils.seeding import set_seed                              # noqa: E402
from src.comms.gnn_comm import GNNCommPipeline
from src.comms.noise_channel import NoiseChannel# noqa: E402  # DIFF FROM attn-comm
from src.comms.ib_channel import StochasticBottleneckChannel
from src.comms.channels import LowRankChannel, IdentityChannel


# ----------------------------- Critic (unchanged) ---------------------------


class CentralisedCritic(nn.Module):
    """Identical to the no-comm / fc-comm / attn-comm critic so cross-
    condition differences are only the aggregator's doing."""

    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# ----------------------------- Rollout buffer -------------------------------


class RolloutBuffer:
    """Stores one on-policy batch for PPO.

    DIFF FROM attn-comm: stores per-layer embeddings instead of attention
    weights. Stored as (T, L+1, N, msg_dim) — for L=2, that's three
    layer-states per step.
    """

    def __init__(self, T, N, obs_dim, msg_dim, num_layers, device):
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
        # DIFF FROM attn-comm: per-layer hidden state stack
        # Shape: (T, num_layers+1, N, msg_dim)
        self.per_layer = torch.zeros(T, num_layers + 1, N, msg_dim, device=device)
        self.ptr = 0

    def add(self, obs, global_obs, actions, logprobs, value, reward, done,
            msgs, per_layer):
        i = self.ptr
        self.obs[i] = obs
        self.global_obs[i] = global_obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.msgs[i] = msgs
        self.per_layer[i] = per_layer
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


# ----------------------------- Env helpers ----------------------------------


def make_env(seed, args):
    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if args.agent_neighbors < 0 else args.agent_neighbors
    _ln = None if args.landmark_neighbors < 0 else args.landmark_neighbors
    env = planC_parallel_env(
        N=args.n_agents, num_landmarks=args.num_landmarks, num_obstacles=args.num_obstacles, local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=args.max_cycles, continuous_actions=False
    )
    env.reset(seed=seed)
    return env


def obs_dict_to_tensor(obs_dict, agents, device):
    arr = np.stack([obs_dict[a] for a in agents], axis=0)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


# ----------------------------- Per-layer diagnostic ------------------------


def per_layer_l2(per_layer_buffer: torch.Tensor) -> list[float]:
    """Mean L2 norm per layer, averaged across (T, N).

    Args:
        per_layer_buffer: tensor (T, L+1, N, D) from RolloutBuffer.

    Returns:
        list of L+1 floats — the per-layer mean L2.
    """
    # (T, L+1, N, D) → norm over D → (T, L+1, N) → mean over (T, N) → (L+1,)
    return per_layer_buffer.norm(dim=-1).mean(dim=(0, 2)).tolist()


# ----------------------------- Training loop -------------------------------


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[gnn-comm/train] device={device}, seed={args.seed}, "
        f"msg_dim={args.msg_dim}, num_layers={args.num_layers}"
    )

    if args.no_wandb:
        wandb_run = None
    else:
        import wandb
        wandb_run = wandb.init(
            project="msc-marl-comm",
            name=f"gnn_bw{args.bandwidth}_seed{args.seed}_msg{args.msg_dim}",
            config=vars(args),
            tags=["w4", "gnn-comm", "ib"],
        )

    env = make_env(args.seed, args)
    obs_dict, _ = env.reset(seed=args.seed)
    agents = list(env.agents)
    N = len(agents)
    obs_dim = obs_dict[agents[0]].shape[0]
    action_dim = env.action_space(agents[0]).n

    if getattr(args, "sigma", 0.0) > 0:
        from src.comms.noise_channel import NoiseChannel
        channel = NoiseChannel(msg_dim=args.msg_dim, sigma=args.sigma)
    elif getattr(args, "beta", 0.0) > 0:
        from src.comms.ib_channel import StochasticBottleneckChannel
        channel = StochasticBottleneckChannel(msg_dim=args.msg_dim)
    elif getattr(args, "bandwidth", args.msg_dim) < args.msg_dim:
        channel = LowRankChannel(msg_dim=args.msg_dim, bandwidth=args.bandwidth)
    else:
        channel = None   # full bandwidth / identity

    pipeline = GNNCommPipeline(
        obs_dim=obs_dim,
        msg_dim=args.msg_dim,
        action_dim=action_dim,
        num_layers=args.num_layers,
        hidden=args.hidden,
        channel=channel,
    ).to(device)
    critic = CentralisedCritic(N * obs_dim, hidden=args.hidden).to(device)
    optim_a = torch.optim.Adam(pipeline.parameters(), lr=args.lr)
    optim_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    print(f"[gnn-comm/train] channel={type(pipeline.channel).__name__}")

    buf = RolloutBuffer(args.rollout_len, N, obs_dim, args.msg_dim,
                        args.num_layers, device)

    total_env_steps = 0
    ep_ret = 0.0
    ep_rets: list[float] = []
    t0 = time.time()

    while total_env_steps < args.total_steps:
        buf.reset()
        while not buf.is_full():
            obs_t = obs_dict_to_tensor(obs_dict, agents, device)         # (N, obs_dim)
            global_obs = obs_t.flatten()

            with torch.no_grad():
                # DIFF FROM attn-comm: 5-tuple unpack
                logits, msgs, msgs_post, agg, per_layer_list = \
                    pipeline(obs_t.unsqueeze(0))
                logits = logits.squeeze(0)
                msgs_s = msgs.squeeze(0)
                # per_layer_list is a list[L+1] of (1, N, D); stack -> (L+1, N, D)
                per_layer_step = torch.stack(
                    [h.squeeze(0) for h in per_layer_list], dim=0
                )

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

            buf.add(obs_t, global_obs, actions, logprobs, value, team_reward, terminated,
                    msgs_s, per_layer_step)

            ep_ret += team_reward
            obs_dict = next_obs_dict
            total_env_steps += 1  # real env transitions

            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs_dict, _ = env.reset()

        # GAE + returns
        with torch.no_grad():
            last_obs_t = obs_dict_to_tensor(obs_dict, agents, device)
            last_value = critic(last_obs_t.flatten().unsqueeze(0)).squeeze(0)
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last_value,
                               args.gamma, args.gae_lambda)
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
                obs_at_t = buf.obs[unique_t]
                # DIFF FROM attn-comm: 5-tuple unpack again
                logits_full, _, _, _, _ = pipeline(obs_at_t)
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
                kl_term = 0.0
                if args.beta > 0 and hasattr(pipeline, "channel") and hasattr(pipeline.channel, "kl_loss"):
                    kl_term = pipeline.channel.kl_loss()

                loss = policy_loss + 0.5 * value_loss - args.ent_coef * ent + args.beta * kl_term

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
        with torch.no_grad():
            msg_l2 = buf.msgs.norm(dim=-1).mean().item()
            agg_l2 = buf.per_layer[:, -1].norm(dim=-1).mean().item()
            # DIFF FROM attn-comm: per-layer L2 trajectory
            per_layer_trajectory = per_layer_l2(buf.per_layer)

        layer_fmt = "[" + ", ".join(f"{x:.3f}" for x in per_layer_trajectory) + "]"
        print(
            f"steps={total_env_steps:>8d}  mean_ret={mean_ret:7.2f}  "
            f"fps={fps}  |msg|={msg_l2:.3f}  |agg|={agg_l2:.3f}  "
            f"per_layer_L2={layer_fmt}"
        )
        if wandb_run is not None:
            log_dict = {
                "step": total_env_steps,
                "mean_return_last20": mean_ret,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": ent.item(),
                "fps": fps,
                "msg_norm_mean": msg_l2,
                "agg_norm_mean": agg_l2,
            }
            for i, v in enumerate(per_layer_trajectory):
                log_dict[f"per_layer_L2_l{i}"] = v
            wandb_run.log(log_dict)

    # save final checkpoint
    save_dir = REPO_ROOT / "experiments" / "gnncomm_mappo" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pipeline": pipeline.state_dict(),
            "critic": critic.state_dict(),
            "args": vars(args),
        },
        save_dir / "final.pt",
    )
    print(f"[gnn-comm/train] saved checkpoint to {save_dir}/final.pt")
    if wandb_run is not None:
        wandb_run.finish()


# ----------------------------- CLI -----------------------------------------


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
    p.add_argument("--msg-dim", type=int, default=8,
                   help="Per-agent message dim (D).")
    p.add_argument("--num-layers", type=int, default=2,
                   help="GNN depth L (spec v2 default = 2).")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise std on messages")
    p.add_argument("--beta", type=float, default=0.0, help="IB KL weight (0 = off)")
    p.add_argument("--bandwidth", type=int, default=8, help="Low-rank bandwidth K (<= msg_dim). K=msg_dim means full")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
    
