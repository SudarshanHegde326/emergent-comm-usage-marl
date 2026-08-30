"""Communication channel modules — apply a bandwidth-like constraint to
per-agent messages before any receiver consumes them.

DESIGN CONTRACT
---------------
All channel classes satisfy: input shape == output shape == (B, N, D).
They are pure ``nn.Module`` and slot directly between ``MessageHead``
and any ``Aggregator`` in the pipeline.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel(nn.Module):
    """Common interface for all channels.

    Subclasses must respect:
      forward(msgs: (B, N, D)) -> msgs_out: (B, N, D)
    """

    def __init__(self, msg_dim: int, bandwidth: int) -> None:
        super().__init__()
        self.msg_dim = int(msg_dim)
        self.bandwidth = int(bandwidth)

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"msg_dim={self.msg_dim}, bandwidth={self.bandwidth}"


class IdentityChannel(Channel):
    """No-op channel baseline."""

    def __init__(self, msg_dim: int, bandwidth: int = 0) -> None:
        super().__init__(msg_dim=msg_dim, bandwidth=msg_dim)

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        return msgs


class LowRankChannel(Channel):
    """Low-rank bottleneck: D -> K -> D."""

    def __init__(self, msg_dim: int, bandwidth: int, bias: bool = True) -> None:
        if not (1 <= bandwidth <= msg_dim):
            raise ValueError(
                f"bandwidth ({bandwidth}) must satisfy 1 <= bandwidth <= "
                f"msg_dim ({msg_dim})"
            )
        super().__init__(msg_dim=msg_dim, bandwidth=bandwidth)
        self.down = nn.Linear(msg_dim, bandwidth, bias=bias)
        self.up = nn.Linear(bandwidth, msg_dim, bias=bias)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        if bias:
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(msgs))


class QuantizedChannel(Channel):
    """Per-dim scalar quantisation with K levels in [-1, 1]."""

    def __init__(self, msg_dim: int, bandwidth: int) -> None:
        if bandwidth < 2:
            raise ValueError(
                f"bandwidth ({bandwidth}) must be >= 2 for quantisation to "
                f"have at least 2 levels."
            )
        super().__init__(msg_dim=msg_dim, bandwidth=bandwidth)

    def _quantise(self, x: torch.Tensor) -> torch.Tensor:
        K = self.bandwidth
        scale = (K - 1) / 2.0
        x_scaled = x * scale + scale
        x_rounded_int = torch.round(x_scaled)
        x_rounded = (x_rounded_int - scale) / scale
        return x_rounded

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        x_tanh = torch.tanh(msgs)
        x_q = self._quantise(x_tanh)
        # Straight-Through Estimator (STE) tracking logic
        return x_q + (x_tanh - x_tanh.detach())


def build_channel(channel_type: str, msg_dim: int, bandwidth: int) -> Channel:
    """Create a channel by string name."""
    ct = channel_type.lower()
    if ct in ("none", "identity"):
        return IdentityChannel(msg_dim=msg_dim)
    if ct in ("low-rank", "lowrank", "projection"):
        return LowRankChannel(msg_dim=msg_dim, bandwidth=bandwidth)
    if ct in ("quantized", "quantised", "quant"):
        return QuantizedChannel(msg_dim=msg_dim, bandwidth=bandwidth)
    raise ValueError(
        f"unknown channel_type {channel_type!r}. "
        f"choose from 'none', 'low-rank', 'quantized'."
    )
