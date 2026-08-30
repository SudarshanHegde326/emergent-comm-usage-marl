"""Multi-head scaled-dot-product attention communication module.
File: src/comms/attention_comm.py
Author: Sudarshan Hegde (ID: 25866326)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MessageHead(nn.Module):
    """Encodes local observations into raw message vectors: (B, N, obs_dim) -> (B, N, msg_dim)."""
    def __init__(self, obs_dim: int, msg_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, msg_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class MultiHeadAttentionAggregator(nn.Module):
    """Computes a selective, parameterized attention matrix over distinct communication lines."""
    def __init__(self, msg_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if msg_dim % num_heads != 0:
            raise ValueError(
                f"msg_dim ({msg_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.msg_dim = msg_dim
        self.num_heads = num_heads
        self.d_head = msg_dim // num_heads

        # Linear projection matrices for key, query, and value calculations
        self.W_q = nn.Linear(msg_dim, msg_dim, bias=False)
        self.W_k = nn.Linear(msg_dim, msg_dim, bias=False)
        self.W_v = nn.Linear(msg_dim, msg_dim, bias=False)
        self.W_o = nn.Linear(msg_dim, msg_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Initialize linear layer weights using the Xavier uniform distribution scheme
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, msgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = msgs.shape
        H = self.num_heads
        d_h = self.d_head

        # 1. Generate Query, Key, and Value projections
        Q = self.W_q(msgs)
        K = self.W_k(msgs)
        V = self.W_v(msgs)

        # 2. Reshape and transpose shapes for Multi-Head calculations: (B, N, D) -> (B, H, N, d_h)
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, N, H, d_h).transpose(1, 2).contiguous()

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # 3. Calculate scaled dot-product similarity scores: (B, H, N, N)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (d_h ** 0.5)

        # 4. Self-Masking: Force an agent's self-attention score to -inf so it resets to 0 weight during softmax
        eye = torch.eye(N, device=msgs.device, dtype=torch.bool)
        scores = scores.masked_fill(eye.view(1, 1, N, N), float("-inf"))

        # 5. Normalize weights across senders using Softmax over the final dimension
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 6. Contract weights against Value sub-spaces: (B, H, N, N) @ (B, H, N, d_h) -> (B, H, N, d_h)
        agg_heads = torch.matmul(weights, Vh)

        # 7. Concatenate parallel head layouts back together: (B, H, N, d_h) -> (B, N, D)
        agg = agg_heads.transpose(1, 2).contiguous().view(B, N, D)

        # 8. Run final mixing projection
        agg = self.W_o(agg)

        return agg, weights

class CommActor(nn.Module):
    """Processes local observations combined with aggregated incoming signals to output action logits."""
    def __init__(self, obs_dim: int, msg_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = obs_dim + msg_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor, agg_msg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, agg_msg], dim=-1))

class AttentionCommPipeline(nn.Module):
    """Ties together observation mapping, multi-head attention aggregation, and policy actor modules."""
    def __init__(self, obs_dim: int, msg_dim: int, action_dim: int, num_heads: int = 4, hidden: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.msg_head = MessageHead(obs_dim, msg_dim, hidden=hidden // 2)
        self.agg = MultiHeadAttentionAggregator(msg_dim, num_heads=num_heads, dropout=dropout)
        self.actor = CommActor(obs_dim, msg_dim, action_dim, hidden=hidden)

        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.action_dim = action_dim
        self.num_heads = num_heads

    def forward(self, obs_NB: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        msgs = self.msg_head(obs_NB)
        agg, weights = self.agg(msgs)
        logits = self.actor(obs_NB, agg)
        return logits, msgs, agg, weights
