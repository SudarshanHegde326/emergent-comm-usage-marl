"""SL-specific seed aggregator.

The Day-14 `aggregate_seeds.py` is hardcoded to `simple_spread_v3`. SL
checkpoints have different obs/action dimensions and a different env, so
they need their own aggregator. The interface mirrors aggregate_seeds.py
so the downstream plotting code (polish_3way_plot.py) can consume the
JSONs without changes.

USAGE
-----
    python aggregate_sl_seeds.py \
        --condition sl-attn \
        --checkpoints experiments/sl_attn/seed0/final.pt \
                      experiments/sl_attn/seed1/final.pt \
                      experiments/sl_attn/seed2/final.pt \
        --eval-episodes 20 \
        --out experiments/sl_attn/aggregate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

# --- path setup -------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
candidate_roots = [THIS_FILE.parent.parent.parent, THIS_FILE.parent.parent]
REPO_ROOT = None
for root in candidate_roots:
    if (root / "src" / "envs" / "sl_env.py").exists():
        REPO_ROOT = root
        break
    if (root / "repo" / "src" / "envs" / "sl_env.py").exists():
        REPO_ROOT = root / "repo"
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root (expected sl_env.py present)")
sys.path.insert(0, str(REPO_ROOT))

from src.envs.sl_env import (  # noqa: E402
    SL_AGENT_ORDER,
    SL_ACTION_DIMS,
    SL_MAX_ACTION_DIM,
    SL_N_AGENTS,
    SL_PADDED_OBS_DIM,
    action_mask,
    make_sl_env,
    pad_obs_to_max,
)
from src.comms.attention_comm import (  # noqa: E402
    MessageHead,
    MultiHeadAttentionAggregator,
    CommActor,
)
from src.comms.fc_comm import FCAggregator  # noqa: E402


# --------------------------------------------------------------------------
# Pipeline reconstruction — must match scripts/07_sl_mappo.py
# --------------------------------------------------------------------------

import torch.nn as nn
class SLPipelineNone(nn.Module):
    def __init__(self, padded_obs_dim, action_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(padded_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs_NB):
        return self.actor(obs_NB), None, None


class SLPipelineFC(nn.Module):
    def __init__(self, padded_obs_dim, msg_dim, action_dim, hidden=128):
        super().__init__()
        self.msg_head = MessageHead(padded_obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = FCAggregator(mode="mean")
        self.actor = CommActor(padded_obs_dim, msg_dim, action_dim, hidden=hidden)

    def forward(self, obs_NB):
        msgs = self.msg_head(obs_NB)
        agg = self.agg(msgs)
        return self.actor(obs_NB, agg), msgs, agg


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
        agg, _ = self.agg(msgs)
        return self.actor(obs_NB, agg), msgs, agg


def build_pipeline_from_ckpt(ckpt_args, device):
    ct = ckpt_args["comm_type"]
    if ct == "none":
        pipe = SLPipelineNone(SL_PADDED_OBS_DIM, SL_MAX_ACTION_DIM,
                              hidden=ckpt_args["hidden"])
    elif ct == "fc":
        pipe = SLPipelineFC(SL_PADDED_OBS_DIM, ckpt_args["msg_dim"],
                            SL_MAX_ACTION_DIM, hidden=ckpt_args["hidden"])
    elif ct == "attn":
        pipe = SLPipelineAttn(SL_PADDED_OBS_DIM, ckpt_args["msg_dim"],
                              SL_MAX_ACTION_DIM,
                              num_heads=ckpt_args["num_heads"],
                              hidden=ckpt_args["hidden"],
                              attn_dropout=ckpt_args.get("attn_dropout", 0.0))
    else:
        raise ValueError(f"unknown comm_type {ct!r}")
    return pipe.to(device)


# --------------------------------------------------------------------------
# Rollout — greedy eval with action masking
# --------------------------------------------------------------------------


def eval_one_seed(checkpoint_path: Path, n_episodes: int, seed_offset: int,
                  device: torch.device) -> List[float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    pipe = build_pipeline_from_ckpt(ckpt_args, device)
    pipe.load_state_dict(ckpt["pipeline"])
    pipe.eval()

    mask = action_mask(device=device)   # (N, A)
    returns = []
    env = make_sl_env(seed=seed_offset)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        done = False
        while not done:
            obs_t = pad_obs_to_max(obs, device=device)        # (N, padded_obs_dim)
            with torch.no_grad():
                logits, _, _ = pipe(obs_t.unsqueeze(0))        # (1, N, A)
                logits = logits.squeeze(0)
                logits_masked = logits + mask                  # invalid -> -inf
                actions = logits_masked.argmax(dim=-1)         # greedy
            action_dict = {agent: int(actions[i].item())
                           for i, agent in enumerate(SL_AGENT_ORDER)}
            obs, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            ep_ret += float(np.mean(list(reward_dict.values())))
            done = any(term_dict.values()) or any(trunc_dict.values())
        returns.append(ep_ret)

    return returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, type=str)
    p.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=12345)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sl-aggregate] device={device} condition={args.condition} "
          f"n_checkpoints={len(args.checkpoints)}")

    per_seed_returns: List[List[float]] = []
    per_seed_train_seed: List[int] = []

    for ck in args.checkpoints:
        if not ck.exists():
            print(f"[sl-aggregate] SKIP missing checkpoint: {ck}")
            continue
        ckpt_meta = torch.load(ck, map_location="cpu", weights_only=False)
        per_seed_train_seed.append(int(ckpt_meta["args"].get("seed", -1)))
        returns = eval_one_seed(ck, args.eval_episodes, args.seed_offset, device)
        per_seed_returns.append(returns)
        print(f"[sl-aggregate]   {ck.name}: seed={ckpt_meta['args'].get('seed')} "
              f"mean={np.mean(returns):+.2f} std={np.std(returns):.2f}")

    if not per_seed_returns:
        raise SystemExit("No checkpoints loaded.")

    per_seed_mean = [float(np.mean(r)) for r in per_seed_returns]
    per_seed_std = [float(np.std(r)) for r in per_seed_returns]
    all_flat = [float(x) for sub in per_seed_returns for x in sub]

    if len(per_seed_mean) >= 2:
        mean_across = float(np.mean(per_seed_mean))
        std_across = float(np.std(per_seed_mean, ddof=1))
    else:
        mean_across = float(per_seed_mean[0])
        std_across = 0.0

    payload: dict[str, Any] = {
        "condition": args.condition,
        "env": "simple_speaker_listener_v4",
        "num_seeds": len(per_seed_returns),
        "seeds": per_seed_train_seed,
        "per_seed_mean_return": per_seed_mean,
        "per_seed_std_return": per_seed_std,
        "all_returns_flat": all_flat,
        "mean_return_across_seeds": mean_across,
        "std_return_across_seeds": std_across,
        "eval_episodes_per_seed": int(args.eval_episodes),
        "checkpoints": [str(p) for p in args.checkpoints if p.exists()],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[sl-aggregate] saved -> {args.out}")
    print(f"[sl-aggregate] {args.condition}: {mean_across:+.2f} ± {std_across:.2f} "
          f"(n_seeds={len(per_seed_returns)})")


if __name__ == "__main__":
    main()