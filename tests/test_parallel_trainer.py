"""Tests for the parallel MAPPO trainer. No StarCraft II needed.

The dangerous part of parallelism is that a batch-axis or per-env boundary bug
corrupts results SILENTLY. These tests target exactly those failure modes:

  * batched GAE must equal the proven single-env GAE, computed per-env;
  * one env's episode boundary must not affect another env's advantages;
  * the whole training loop must run end-to-end across parallel fake envs and
    produce a valid checkpoint.

Run:  pytest tests/test_parallel_trainer.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((REPO / "scripts" / "trainers").resolve()))
sys.path.insert(0, str((REPO / "tests").resolve()))

import vec_smacv2 as V            # noqa: E402
import _fake_env_helper as H      # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "par", REPO / "scripts" / "trainers" / "15_mappo_smacv2_parallel.py")
P = importlib.util.module_from_spec(_spec)
sys.modules["par"] = P
_spec.loader.exec_module(P)

# the proven single-env module, for cross-checking GAE
S = P.S


# =============================================================================
# 1. Batched GAE must equal the single-env GAE run independently per env.
#    This is the highest-risk math in the whole parallel rewrite.
# =============================================================================
def test_batched_gae_matches_single_env_per_env():
    torch.manual_seed(0)
    T, B = 12, 5
    rewards = torch.randn(T, B)
    values = torch.randn(T, B)
    dones = (torch.rand(T, B) < 0.2).float()
    last_values = torch.randn(B)
    gamma, lam = 0.99, 0.95

    adv_b, ret_b = P.compute_gae_batched(rewards, values, dones, last_values,
                                         gamma, lam)

    # Reference: run the SINGLE-env GAE separately on each column.
    for b in range(B):
        adv_s, ret_s = S.compute_gae(rewards[:, b], values[:, b], dones[:, b],
                                     last_values[b], gamma, lam)
        assert torch.allclose(adv_b[:, b], adv_s, atol=1e-5), \
            f"batched GAE differs from single-env GAE in column {b}"
        assert torch.allclose(ret_b[:, b], ret_s, atol=1e-5)


def test_gae_env_isolation():
    """Changing env b's rewards must NOT change env b''s advantages. Proves no
    cross-env leakage in the batched recursion."""
    torch.manual_seed(1)
    T, B = 10, 4
    rewards = torch.randn(T, B)
    values = torch.randn(T, B)
    dones = torch.zeros(T, B)
    last_values = torch.zeros(B)

    adv1, _ = P.compute_gae_batched(rewards, values, dones, last_values, 0.99, 0.95)
    r2 = rewards.clone()
    r2[:, 2] += 100.0                       # perturb ONLY env 2
    adv2, _ = P.compute_gae_batched(r2, values, dones, last_values, 0.99, 0.95)

    # env 2 changes; all others identical.
    for b in range(B):
        if b == 2:
            assert not torch.allclose(adv1[:, b], adv2[:, b])
        else:
            assert torch.allclose(adv1[:, b], adv2[:, b], atol=1e-6), \
                f"perturbing env 2 leaked into env {b}"


def test_gae_done_stops_bootstrap_per_env():
    """A done in env b at time t must stop the return bootstrap for env b there,
    independently of the other envs' done flags."""
    T, B = 3, 2
    rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    values = torch.tensor([[0.0, 0.0], [100.0, 100.0], [0.0, 0.0]])
    dones = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])   # env0 done at t0
    last = torch.zeros(2)
    adv, _ = P.compute_gae_batched(rewards, values, dones, last, 0.99, 0.95)
    # env0: done at t0 -> adv[0,0] = reward - value = 1 - 0 = 1 (no bootstrap)
    assert abs(adv[0, 0].item() - 1.0) < 1e-6
    # env1: not done -> adv[0,1] bootstraps through the big value at t1
    assert adv[0, 1].item() > 5.0


# =============================================================================
# 2. Argument handling.
# =============================================================================
def test_num_envs_flag_parses():
    a = P.build_args(["--comm", "attn", "--num-envs", "8", "--no-wandb"])
    assert a.num_envs == 8
    a2 = P.build_args(["--comm", "none", "--no-wandb"])
    assert a2.num_envs == 8      # default


def test_rollout_must_divide_chunk():
    with pytest.raises(ValueError, match="multiple"):
        P.build_args(["--comm", "none", "--rollout-len", "30",
                      "--chunk-len", "40", "--no-wandb"])


# =============================================================================
# 3. End-to-end training across parallel fake envs, all comm conditions.
# =============================================================================
def _factory(rank):
    # 4-agent fake env with the SMACv2-style obs layout for the aux path.
    return lambda: H.FakeEnv(seed=200 + rank, n_agents=4, n_actions=6,
                             obs_dim=30, state_dim=40)


@pytest.mark.parametrize("comm", ["none", "fc", "gnn", "attn"])
def test_parallel_training_runs_end_to_end(comm, tmp_path):
    a = P.build_args(["--comm", comm, "--num-envs", "3", "--no-wandb"])
    a.total_steps = 16          # env-steps-per-env; tiny
    a.rollout_len = 8
    a.chunk_len = 4
    a.ppo_epochs = 2
    a.hidden = 16
    a.msg_dim = 8
    a.num_heads = 2
    a.num_layers = 1
    a.eval_episodes = 0
    a.final_eval_episodes = 0
    a.ckpt_interval = 8
    a.aux_coef = 0.1 if comm != "none" else 0.0
    a.out_dir = str(tmp_path / f"par_{comm}")
    res = P.train_parallel(a, env_factory_override=_factory)
    assert "train_win_rate" in res
    saved = tmp_path / f"par_{comm}" / "seed0" / "final.pt"
    assert saved.exists()
    ck = torch.load(saved, map_location="cpu", weights_only=False)
    assert "final_test_win_rate" in ck and "steps" in ck


def test_parallel_checkpoint_and_resume(tmp_path):
    # total_steps / ckpt_interval are TOTAL TRANSITIONS (= steps * num_envs).
    # One rollout gathers rollout_len * num_envs = 8 * 3 = 24 transitions, so the
    # budgets must be comfortably above that for resume to have work to do.
    B = 3
    def mk(total):
        a = P.build_args(["--comm", "attn", "--num-envs", str(B), "--no-wandb"])
        a.total_steps = total
        a.rollout_len = 8; a.chunk_len = 4; a.ppo_epochs = 1
        a.hidden = 16; a.msg_dim = 8; a.num_heads = 2; a.num_layers = 1
        a.eval_episodes = 0; a.final_eval_episodes = 0; a.ckpt_interval = 24
        a.aux_coef = 0.0
        a.out_dir = str(tmp_path)
        return a

    P.train_parallel(mk(48), env_factory_override=_factory)
    ck1 = torch.load(tmp_path / "seed0" / "latest.pt",
                     map_location="cpu", weights_only=False)
    assert ck1["steps"] * B >= 48          # reached the 48-transition budget
    # Full resumable checkpoint must carry optimiser + normaliser state.
    for k in ("opt_a", "opt_c", "obs_norm", "state_norm", "torch_rng"):
        assert k in ck1

    a2 = mk(96); a2.resume = True
    P.train_parallel(a2, env_factory_override=_factory)
    ck2 = torch.load(tmp_path / "seed0" / "latest.pt",
                     map_location="cpu", weights_only=False)
    assert ck2["steps"] > ck1["steps"]     # resume actually made progress
    assert ck2["steps"] * B >= 96          # reached the 96-transition budget


def test_parallel_greedy_eval_runs(tmp_path):
    a = P.build_args(["--comm", "attn", "--num-envs", "3", "--no-wandb"])
    a.total_steps = 8
    a.rollout_len = 8; a.chunk_len = 4; a.ppo_epochs = 1
    a.hidden = 16; a.msg_dim = 8; a.num_heads = 2; a.num_layers = 1
    a.eval_episodes = 4; a.final_eval_episodes = 4; a.eval_interval = 8
    a.aux_coef = 0.0
    a.out_dir = str(tmp_path)
    res = P.train_parallel(a, env_factory_override=_factory)
    assert 0.0 <= res["test_win_rate"] <= 1.0
