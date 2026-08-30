"""Multi-hop GNN communication module — vanilla, uniform-weight mean.

WHAT THIS IS
------------
Implements the dissertation's fourth communication condition. The
design is fully specified in:

    docs/gnn_module_v2.md  (Day 22 spec, version 2)

MATH (spec v2 §4)
-----------------
    h^{(0)}      = msgs                                              # (B, N, D)
    agg_i^{(l)}  = (1 / (N-1)) * sum_{j != i} h_j^{(l)}              # eq.(1)
    h_i^{(l+1)}  = LayerNorm( h_i^{(l)} + f_l( agg_i^{(l)} ) )       # eq.(2)
    agg_msg_i    = h_i^{(L)}                                          # eq.(4)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.comms.attention_comm import MessageHead, CommActor  # noqa: F401


class GNNAggregator(nn.Module):
    """Multi-layer mean-aggregation GNN on a fully-connected (no self-loop)
    agent graph.
    """

    def __init__(self, msg_dim: int, num_layers: int = 2,
                 hidden: Optional[int] = None) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0; got {num_layers}")
        if hidden is None:
            hidden = msg_dim
        self.msg_dim = int(msg_dim)
        self.num_layers = int(num_layers)
        self.hidden = int(hidden)

        # L independent per-node transforms f_l = Linear(D, H) -> Tanh -> Linear(H, D)
        self.f_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, msg_dim),
            )
            for _ in range(num_layers)
        ])

        # L LayerNorms over the last dim (D)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(msg_dim)
            for _ in range(num_layers)
        ])

        # Xavier init on every Linear; LayerNorm gets default init.
        for f in self.f_layers:
            for m in f.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _mean_neighbour_aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """Mean of OTHER agents only."""
        B, N, D = h.shape
        if N == 1:
            return torch.zeros_like(h)
        total = h.sum(dim=1, keepdim=True)        # (B, 1, D)
        others = total - h                         # (B, N, D)
        return others / float(N - 1)

    def forward(self, msgs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = msgs
        per_layer: List[torch.Tensor] = [h]
        for l in range(self.num_layers):
            agg = self._mean_neighbour_aggregate(h)        # eq. (1) — (B, N, D)
            f_out = self.f_layers[l](agg)                  # per-node transform
            h = self.layer_norms[l](h + f_out)              # eq. (2) — residual + LayerNorm
            per_layer.append(h)
        return h, per_layer


class GNNCommPipeline(nn.Module):
    """End-to-end GNN-comm pipeline with D8 channel slot."""

    def __init__(
        self,
        obs_dim: int,
        msg_dim: int,
        action_dim: int,
        num_layers: int = 2,
        hidden: int = 128,
        channel: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.msg_head = MessageHead(obs_dim, msg_dim, hidden=hidden // 2)

        # D8 — channel slot. Default = IdentityChannel for byte-equal v1 behaviour.
        if channel is None:
            from src.comms.channels import IdentityChannel
            channel = IdentityChannel(msg_dim=msg_dim)
        self.channel = channel

        self.agg = GNNAggregator(msg_dim, num_layers=num_layers, hidden=msg_dim)
        self.actor = CommActor(obs_dim, msg_dim, action_dim, hidden=hidden)

        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.action_dim = action_dim
        self.num_layers = num_layers

    def forward(
        self, obs_NB: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, List[torch.Tensor]]:
        msgs = self.msg_head(obs_NB)                        # (B, N, D)
        msgs_post = self.channel(msgs)                      # (B, N, D)  — D8 channel
        agg, per_layer = self.agg(msgs_post)                # (B, N, D)
        logits = self.actor(obs_NB, agg)                    # (B, N, action_dim)
        return logits, msgs, msgs_post, agg, per_layer
