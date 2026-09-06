"""
Fixed Fully-Connected (FC) Communication Module for Simple Spread.
Implements:
1. Gated MLP Encoder (so blind agents output ~0 vectors).
2. Mean-Excluding-Self (so agents don't average their own messages).
3. Residual LayerNorm Integration.
"""
import torch
import torch.nn as nn

def _mean_excluding_self(m: torch.Tensor) -> torch.Tensor:
    """Mean of other agents' messages, excluding self."""
    B, N, D = m.shape
    if N <= 1:
        return torch.zeros_like(m)
    total = m.sum(dim=1, keepdim=True)
    return (total - m) / (N - 1)

def _zero_init(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class MessageEncoder(nn.Module):
    """Upgraded MessageHead with MLP and Learned Sigmoid Gate."""
    def __init__(self, obs_dim: int, msg_dim: int, hidden: int = 64):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, msg_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, msg_dim),
            nn.Sigmoid()
        )
        # Initialize gate to start mostly open
        nn.init.zeros_(self.gate[0].weight)
        nn.init.ones_(self.gate[0].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        m_raw = self.encode(obs)
        g_val = self.gate(obs)
        return m_raw * g_val

class FCCommPipeline(nn.Module):
    """Pipeline for FC Communication."""
    def __init__(self, obs_dim: int, msg_dim: int, action_dim: int, agg_mode: str = "mean", hidden: int = 128):
        super().__init__()
        self.msg_head = MessageEncoder(obs_dim, msg_dim, hidden=hidden // 2)
        self.transform = _zero_init(nn.Linear(msg_dim, hidden))
        self.norm = nn.LayerNorm(hidden)
        
        # Feature extractor
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU()
        )
        
        # Action head
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, obs_NB: torch.Tensor):
        # 1. Feature Encoding
        h = self.feat(obs_NB)
        
        # 2. Communication
        msgs = self.msg_head(obs_NB)
        agg = _mean_excluding_self(msgs)
        
        # 3. Residual Integration
        z = self.norm(h + torch.tanh(self.transform(agg)))
        
        # 4. Action Logits
        logits = self.actor(z)
        return logits, msgs, agg
    
class FCAggregator(nn.Module):
    """Simple mean aggregator excluding self (for Speaker-Listener trainer)."""
    def __init__(self, mode: str = "mean"):
        super().__init__()
        self.mode = mode

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        # msgs shape: (B, N, D)
        return _mean_excluding_self(msgs)