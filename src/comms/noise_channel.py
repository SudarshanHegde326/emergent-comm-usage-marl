"""Additive Gaussian noise channel module — models physical link noise.

Perturbs continuous message tensors by adding zero-mean independent and
identically distributed (iid) Gaussian random samples in both train and eval modes.
"""

from __future__ import annotations
import torch
from src.comms.channels import Channel


class NoiseChannel(Channel):
    """Add iid Gaussian noise N(0, sigma^2) per dimension to inter-agent messages."""

    def __init__(self, msg_dim: int, sigma: float) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative; got {sigma}")

        # Use msg_dim as a uniform baseline parameter for the base class API contract
        super().__init__(msg_dim=msg_dim, bandwidth=msg_dim)
        self.sigma = float(sigma)

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0.0:
            return msgs  # Bit-exact identity optimization bypass

        # Additive white Gaussian random perturbation matrix
        noise = torch.randn_like(msgs) * self.sigma
        return msgs + noise

    def extra_repr(self) -> str:
        return f"msg_dim={self.msg_dim}, sigma={self.sigma:g}"
