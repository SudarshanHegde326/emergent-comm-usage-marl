"""Tests for the subprocess vectorized SMACv2 wrapper. No StarCraft II needed.

These prove the parallelism machinery is correct in isolation: batched shapes,
auto-reset on episode end, distinct per-worker seeding, and clean shutdown.
The dangerous batching logic in the trainer is tested separately.

Run:  pytest tests/test_vec_smacv2.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# Import by REAL module name (not importlib under a fake name), because 'spawn'
# workers re-import the module to unpickle the env factory; a synthetic name
# would not exist in the child process. The real trainer imports it the same way.
sys.path.insert(0, str((REPO / "scripts" / "trainers").resolve()))
sys.path.insert(0, str((REPO / "tests").resolve()))
import vec_smacv2 as V  # noqa: E402
from _fake_env_helper import FakeEnv, factory as _factory  # noqa: E402




def test_vec_builds_and_reports_env_info():
    vec = V.make_vec(_factory, num_envs=4)
    try:
        assert vec.num_envs == 4
        info = vec.get_env_info()
        assert info["n_agents"] == 3
        assert info["n_actions"] == 5
    finally:
        vec.close()


def test_reset_returns_batched_shapes():
    vec = V.make_vec(_factory, num_envs=4)
    try:
        obs, state, avail = vec.reset()
        assert obs.shape == (4, 3, 8)
        assert state.shape == (4, 10)
        assert avail.shape == (4, 3, 5)
    finally:
        vec.close()


def test_workers_have_distinct_seeds():
    """Each worker's obs encodes its seed; they must differ, or all workers are
    playing identical procedurally-generated maps (the bug we must avoid)."""
    vec = V.make_vec(_factory, num_envs=4)
    try:
        obs, _, _ = vec.reset()
        per_env_seed = obs[:, 0, 0]           # we encoded seed into every obs cell
        assert len(set(per_env_seed.tolist())) == 4, \
            f"workers share seeds: {per_env_seed}"
        assert sorted(per_env_seed.tolist()) == [100.0, 101.0, 102.0, 103.0]
    finally:
        vec.close()


def test_step_shapes_and_auto_reset():
    vec = V.make_vec(_factory, num_envs=4)
    try:
        vec.reset()
        acts = np.ones((4, 3), dtype=np.int64)
        seen_done = False
        for _ in range(20):
            obs, state, avail, reward, done, infos = vec.step(acts)
            assert obs.shape == (4, 3, 8)
            assert state.shape == (4, 10)
            assert avail.shape == (4, 3, 5)
            assert reward.shape == (4,)
            assert done.shape == (4,)
            assert len(infos) == 4
            if done.any():
                seen_done = True
                # On a done step, info must carry battle_won, and the returned
                # obs must already be the NEXT episode's first obs (t reset).
                for i, d in enumerate(done):
                    if d:
                        assert "battle_won" in infos[i]
        assert seen_done, "no episode ended in 20 steps despite ep_len<=7"
    finally:
        vec.close()


def test_envs_end_at_different_times():
    """With ep_len depending on seed, the four envs must NOT all finish on the
    same step -- proving they run independently, not in lockstep."""
    vec = V.make_vec(_factory, num_envs=4)
    try:
        vec.reset()
        acts = np.ones((4, 3), dtype=np.int64)
        first_done_step = [None] * 4
        for step in range(10):
            _, _, _, _, done, _ = vec.step(acts)
            for i, d in enumerate(done):
                if d and first_done_step[i] is None:
                    first_done_step[i] = step
            if all(x is not None for x in first_done_step):
                break
        assert len(set(first_done_step)) > 1, \
            f"all envs ended on the same step: {first_done_step}"
    finally:
        vec.close()


def test_obs_layout_query():
    vec = V.make_vec(_factory, num_envs=2)
    try:
        layout = vec.get_obs_layout()
        assert layout == (2, 4)     # move=2, enemy=2*2=4
    finally:
        vec.close()


def test_close_is_idempotent():
    vec = V.make_vec(_factory, num_envs=2)
    vec.close()
    vec.close()   # must not raise
