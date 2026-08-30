"""Information Bottleneck (IB) communication channel module.

Encodes outgoing message representations stochastically into diagonal Gaussian
distributions, sampling latent representations using the reparameterisation trick.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.comms.channels import Channel

# Numerical stability bounds for log variance to prevent division instability
_LOGVAR_MIN = -6.0     # sigma >= exp(-3.0) ≈ 0.05
_LOGVAR_MAX = +2.0     # sigma <= exp(+1.0) ≈ 2.72


class StochasticBottleneckChannel(Channel):
    """Diagonal-Gaussian encoder mapping inputs to a bounded statistical space."""

    def __init__(self, msg_dim: int) -> None:
        super().__init__(msg_dim=msg_dim, bandwidth=msg_dim)

        self.mu_head = nn.Linear(msg_dim, msg_dim)
        self.logvar_head = nn.Linear(msg_dim, msg_dim)

        # Initialize near identity mappings so initialization starts close to baselines
        nn.init.eye_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

        # Cache buffer for loss aggregation across target graphics devices
        self.register_buffer("_last_kl", torch.zeros(()))

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        # msgs input contract shape: (B, N, D)
        mu = self.mu_head(msgs)
        logvar = self.logvar_head(msgs).clamp(min=_LOGVAR_MIN, max=_LOGVAR_MAX)
        sigma = torch.exp(0.5 * logvar)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + sigma * eps             # Reparameterization trick step
        else:
            z = mu                           # Greedy evaluation fallback

        # Closed-form analytical KL divergence evaluation: q(z||N(0, I))
        kl_per = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - logvar).sum(dim=-1)
        self._last_kl = kl_per.mean()

        return z

    def kl_loss(self) -> torch.Tensor:
        """Fetch the scalar mean KL divergence from the most recent pass."""
        return self._last_kl

    def extra_repr(self) -> str:
        return f"msg_dim={self.msg_dim} (KL prior=N(0,I), eval=mean)"


def build_ib_channel(channel_type: str, msg_dim: int, bandwidth: int):
    """Factory router extension mapping string configurations to channels."""
    from src.comms.channels import build_channel
    if channel_type.lower() in ("ib", "vib", "info-bottleneck", "info_bottleneck"):
        return StochasticBottleneckChannel(msg_dim=msg_dim)
    return build_channel(channel_type, msg_dim=msg_dim, bandwidth=bandwidth)
