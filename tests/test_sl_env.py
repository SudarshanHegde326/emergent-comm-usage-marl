"""Unit tests for src.envs.sl_env."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.sl_env import (
    SL_AGENT_ORDER,
    SL_ACTION_DIMS,
    SL_MAX_ACTION_DIM,
    SL_N_AGENTS,
    SL_OBS_DIMS,
    SL_PADDED_OBS_DIM,
    action_mask,
    make_sl_env,
    pad_obs_to_max,
)


def test_make_sl_env():
    env = make_sl_env(seed=0)
    assert list(env.agents) == SL_AGENT_ORDER, (
        f"agent order mismatch — got {list(env.agents)}"
    )


def test_pad_obs_shapes():
    env = make_sl_env(seed=0)
    obs, _ = env.reset(seed=0)
    padded = pad_obs_to_max(obs, device="cpu")
    assert padded.shape == (SL_N_AGENTS, SL_PADDED_OBS_DIM), (
        f"padded obs shape {padded.shape} != expected "
        f"({SL_N_AGENTS}, {SL_PADDED_OBS_DIM})"
    )
    assert padded.dtype == torch.float32


def test_padding_one_hot():
    env = make_sl_env(seed=0)
    obs, _ = env.reset(seed=0)
    padded = pad_obs_to_max(obs, device="cpu")

    one_hot = padded[:, -SL_N_AGENTS:].numpy()
    expected = np.eye(SL_N_AGENTS, dtype=np.float32)
    np.testing.assert_array_equal(one_hot, expected)

    speaker_raw = padded[0, : SL_OBS_DIMS["speaker_0"]].numpy()
    speaker_pad_zeros = padded[0, SL_OBS_DIMS["speaker_0"] : -SL_N_AGENTS].numpy()
    assert np.all(speaker_pad_zeros == 0.0), (
        f"speaker right-pad must be zeros; got {speaker_pad_zeros}"
    )
    assert np.all(np.isfinite(speaker_raw))


def test_action_mask_shapes():
    mask = action_mask(device="cpu")
    assert mask.shape == (SL_N_AGENTS, SL_MAX_ACTION_DIM)

    speaker_invalid = mask[0, SL_ACTION_DIMS["speaker_0"] :]
    assert torch.all(torch.isinf(speaker_invalid) & (speaker_invalid < 0)), (
        f"speaker invalid action indices should be -inf; got {speaker_invalid}"
    )
    speaker_valid = mask[0, : SL_ACTION_DIMS["speaker_0"]]
    assert torch.all(speaker_valid == 0.0), (
        f"speaker valid action indices should be 0.0; got {speaker_valid}"
    )

    listener_row = mask[1, :]
    assert torch.all(listener_row == 0.0), (
        f"listener should have all valid actions; got {listener_row}"
    )


def test_full_step_roundtrip():
    env = make_sl_env(seed=42)
    obs, _ = env.reset(seed=42)
    padded = pad_obs_to_max(obs, device="cpu")
    assert padded.shape == (SL_N_AGENTS, SL_PADDED_OBS_DIM)

    action_dict = {
        "speaker_0":  0,
        "listener_0": 1,
    }
    next_obs, rewards, terms, truncs, _ = env.step(action_dict)

    for agent in SL_AGENT_ORDER:
        assert agent in next_obs
        assert agent in rewards

    padded2 = pad_obs_to_max(next_obs, device="cpu")
    assert padded2.shape == (SL_N_AGENTS, SL_PADDED_OBS_DIM)

    action_dict = {"speaker_0": 1, "listener_0": 4}
    next_obs2, _, _, _, _ = env.step(action_dict)
    assert all(agent in next_obs2 for agent in SL_AGENT_ORDER)
