"""Simple Speaker-Listener env wrapper — heterogeneous-agent adapter.

Centralises every Speaker-Listener-specific detail in one file so the
unified MAPPO trainer (``07_sl_mappo.py``) can treat the env as if it
were homogeneous.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch

# Canonical, fixed agent order.
SL_AGENT_ORDER: List[str] = ["speaker_0", "listener_0"]

SL_OBS_DIMS: Dict[str, int] = {
    "speaker_0": 3,
    "listener_0": 11,
}
SL_ACTION_DIMS: Dict[str, int] = {
    "speaker_0": 3,
    "listener_0": 5,
}

SL_MAX_OBS_DIM: int = max(SL_OBS_DIMS.values())              # 11
SL_MAX_ACTION_DIM: int = max(SL_ACTION_DIMS.values())        # 5
SL_N_AGENTS: int = len(SL_AGENT_ORDER)                       # 2
SL_PADDED_OBS_DIM: int = SL_MAX_OBS_DIM + SL_N_AGENTS        # 11 + 2 = 13


def make_sl_env(seed: int = 0, max_cycles: int = 25):
    """Build a Simple Speaker-Listener parallel env and reset it."""
    from mpe2 import simple_speaker_listener_v4

    env = simple_speaker_listener_v4.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=False,
    )
    env.reset(seed=seed)
    actual = list(env.agents)
    if actual != SL_AGENT_ORDER:
        raise RuntimeError(
            f"PettingZoo simple_speaker_listener_v4 agent order has changed: "
            f"expected {SL_AGENT_ORDER}, got {actual}. "
            f"Update SL_AGENT_ORDER in sl_env.py."
        )
    return env


def pad_obs_to_max(obs_dict: Dict[str, np.ndarray], device: torch.device | str = "cpu") -> torch.Tensor:
    """Stack and right-pad obs from a parallel env step into a fixed tensor."""
    out = np.zeros((SL_N_AGENTS, SL_PADDED_OBS_DIM), dtype=np.float32)
    for i, agent in enumerate(SL_AGENT_ORDER):
        raw = np.asarray(obs_dict[agent], dtype=np.float32)
        if raw.shape != (SL_OBS_DIMS[agent],):
            raise ValueError(
                f"agent {agent} obs shape {raw.shape} != expected "
                f"({SL_OBS_DIMS[agent]},)"
            )
        # Place raw obs in the first slots
        out[i, : SL_OBS_DIMS[agent]] = raw
        # One-hot agent indicator in slots SL_MAX_OBS_DIM .. SL_PADDED_OBS_DIM
        out[i, SL_MAX_OBS_DIM + i] = 1.0
    return torch.as_tensor(out, device=device)


_ACTION_MASK_CPU: torch.Tensor | None = None


def action_mask(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return (N=2, max_action_dim=5) tensor: 0.0 for valid, -inf for invalid."""
    global _ACTION_MASK_CPU
    if _ACTION_MASK_CPU is None:
        mask = torch.zeros(SL_N_AGENTS, SL_MAX_ACTION_DIM, dtype=torch.float32)
        for i, agent in enumerate(SL_AGENT_ORDER):
            valid = SL_ACTION_DIMS[agent]
            if valid < SL_MAX_ACTION_DIM:
                mask[i, valid:] = float("-inf")
        _ACTION_MASK_CPU = mask
    return _ACTION_MASK_CPU.to(device)

