"""
Paper-matched MAPPO for SMACv2 - all communication conditions in ONE file.

WHAT CHANGED IN THIS REVISION (read this before running anything)
----------------------------------------------------------------
1. GNN condition is now a REAL graph attention network (Velickovic et al.,
   2018): additive attention with a learned attention vector and LeakyReLU,
   multi-head, multi-hop. In the previous revision GNNComm and FCComm were
   byte-identical mean aggregation, so two of the four experimental
   conditions were the same model. They are now genuinely different.

2. Communication can now enter the recurrent state (--comm-to-memory on,
   the default). Previously the communication output fed only the action
   head and the pre-communication hidden state was carried forward, so a
   receiving agent could not remember what it had been told. Under EPO
   (prob_obs_enemy=0.0) an ally that spots an enemy at t=5 is the ONLY
   agent that will ever see it, and the team needs that fact to survive
   for the rest of a 200-step episode. Without this, communication is
   structurally unable to help and a null result says nothing.
   Use --comm-to-memory off to reproduce the old behaviour as an ablation.

3. Greedy evaluation added (--eval-episodes / --eval-interval). Ellis et al.
   report TEST win rate over 32 greedy episodes. The previous code logged
   the training win rate of a sampled policy, which is not comparable.

4. --num-heads and --num-layers are now accepted (the launcher scripts were
   passing them; argparse rejected them and every attention/GNN run died
   in under a second while the launcher still printed "Run complete").

5. AttnComm no longer leaks dead agents' values when an agent is the sole
   survivor (a fully masked attention row used to soften to uniform
   weights over dead senders).

WHAT WAS ALREADY CORRECT AND IS PRESERVED
-----------------------------------------
  * EPO is prob_obs_enemy=0.0 AND action_mask=False, with conic_fov=False
    in BOTH modes. Verified line-by-line against the benchmark's own
    smacv2/examples/configs/sc2_gen_protoss_epo.yaml.
  * Liveness test av[:, 0] < 0.5 (SMACv2 returns [1, 0, 0, ...] for a dead
    unit and forbids the no-op for a living one).
  * Recurrent state reset during replay mirrors the collector.
  * Observation normaliser frozen during the update, refreshed after.
  * Truncation is treated as terminal, matching the paper's
    "Bootstrap Timeouts: False".
  * Zero-initialised output projections make every condition numerically
    identical at initialisation.

USAGE
-----
  python 14_mappo_smacv2_matched.py --comm none --seed 0
  python 14_mappo_smacv2_matched.py --comm fc   --seed 0
  python 14_mappo_smacv2_matched.py --comm gnn  --seed 0     # = GAT
  python 14_mappo_smacv2_matched.py --comm attn --seed 0

  # EPO - the setting the benchmark designed to require communication:
  python 14_mappo_smacv2_matched.py --comm attn --mode epo --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from src.channel_constraints import ChannelConstraint


# =============================================================================
# Observation normalisation  (paper: "Observation Normalized: True")
# Welford running statistics; updated only while collecting, frozen in updates.
# =============================================================================
class RunningNorm:
    """Welford running normaliser.

    Only the first `active_dim` feature columns are standardised; any trailing
    columns are passed through untouched. This is used so that the one-hot
    agent-ID columns appended to each observation stay clean 0/1 flags rather
    than being rescaled into arbitrary floats. Standard cooperative-MARL
    implementations (PyMARL2, the official MAPPO) normalise only the
    environment features and leave the agent identity as-is. Pass
    active_dim=None (the default) to normalise every column, as before.
    """

    def __init__(self, dim: int, device, eps: float = 1e-5, active_dim: int = None):
        self.active_dim = dim if active_dim is None else active_dim
        self.mean = torch.zeros(self.active_dim, device=device)
        self.var = torch.ones(self.active_dim, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])[:, :self.active_dim]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count,
                "active_dim": self.active_dim}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"]
        self.var = sd["var"]
        self.count = sd["count"]
        self.active_dim = sd.get("active_dim", self.mean.shape[0])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        head = torch.clamp((x[..., :self.active_dim] - self.mean)
                           / torch.sqrt(self.var + 1e-8), -10.0, 10.0)
        if self.active_dim >= x.shape[-1]:
            return head
        # Leave trailing columns (e.g. one-hot agent IDs) exactly as they are.
        return torch.cat([head, x[..., self.active_dim:]], dim=-1)


# =============================================================================
# Communication modules
#
# Every module maps (B, N, hidden) -> (B, N, hidden) and receives an
# `alive` mask of shape (B, N) so that dead agents neither send nor are
# attended to. Only the aggregation rule differs between conditions.
#
# Every module ends with  LayerNorm(h + tanh(zero_init_projection(...))),
# so at initialisation ALL conditions compute exactly LayerNorm(h) and are
# numerically identical. The message pathway is then the only thing that
# can make them diverge.
# =============================================================================


def _zero_init(layer: nn.Linear) -> nn.Linear:
    """Zero the weights and bias so the module starts as the identity."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class MessageEncoder(nn.Module):
    """Shared message encoder with an MLP body and an optional learned gate.

    UPGRADE 2 (MLP encoder): a single linear layer cannot separate different
    situations into distinct directions before they are averaged. A
    Linear->ReLU->Linear body can, so that mean-pooling downstream preserves
    more decodable structure. Set mlp=False to recover the old single-linear
    encoder (used to prove, in tests, that the MLP actually adds capacity).

    UPGRADE 1 (learned gating, "when to talk"): a parallel head emits a scalar
    gate in [0,1] per agent; the message is multiplied by it. Under EPO a blind
    agent can drive its gate toward 0 and send (approximately) a zero vector, so
    the mean of {one real message, four zeros} preserves the spotter's signal
    instead of diluting it 1-in-5 with noise. Set gate=False to disable.

    Exposes, after each forward:
      last_gate     (B,N,1)      the gate values actually applied (1.0 if off)
      last_message  (B,N,msg)    the gated message each agent emitted
    so the auxiliary sighting loss (Upgrade 3) can supervise the messages.
    """

    def __init__(self, hidden: int, msg_dim: int, mlp: bool = True,
                 gate: bool = True, channel_cfg: dict = None):
        super().__init__()
        if mlp:
            self.body = nn.Sequential(
                nn.Linear(hidden, msg_dim), nn.ReLU(),
                nn.Linear(msg_dim, msg_dim),
            )
        else:
            self.body = nn.Linear(hidden, msg_dim)
        self.use_gate = gate
        if gate:
            # Bias +1 so the gate starts near sigmoid(1) ~ 0.73 (mostly open):
            # agents begin by communicating and learn when to go quiet, rather
            # than starting silent and having to discover the channel.
            self.gate = nn.Linear(hidden, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)
        cc = channel_cfg or {}
        self.channel = ChannelConstraint(
            msg_dim,
            quant_bits=cc.get('quant_bits', 0),
            noise_std=cc.get('noise_std', 0.0),
            dropout_p=cc.get('dropout_p', 0.0),
            kl_bottleneck=cc.get('kl_bottleneck', False),
        )
        self.last_kl = None
        self.last_gate = None
        self.last_message = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        msg = self.body(h)
        if self.use_gate:
            g = torch.sigmoid(self.gate(h))          # (B,N,1)
            msg = msg * g
        else:
            g = torch.ones(*h.shape[:-1], 1, device=h.device)
        msg = self.channel(msg)                      # apply channel constraints
        self.last_kl = self.channel.last_kl
        self.last_gate = g
        self.last_message = msg
        return msg


def _mean_excluding_self(m: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Mean of other agents' messages, ignoring dead senders and self."""
    w = alive.unsqueeze(-1)                       # (B,N,1)
    total = (m * w).sum(dim=1, keepdim=True)      # sum over all agents
    denom = (w.sum(dim=1, keepdim=True) - w).clamp(min=1.0)
    return (total - m * w) / denom                # subtract own message

def _max_excluding_self(m: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Max pooling of other agents' messages, ignoring dead senders and self."""
    B, N, msg_dim = m.shape
    
    # 1. Create a mask to block self and dead agents (True means blocked)
    blocked = _sender_mask(alive, N)
    
    # 2. Expand messages so each agent i has a view of all senders j
    m_expanded = m.unsqueeze(1).expand(B, N, N, msg_dim)
    
    # 3. Mask out blocked messages by setting them to a very large negative number
    # so they are completely ignored by the max() function
    m_masked = m_expanded.masked_fill(blocked.unsqueeze(-1), -1e9)
    
    # 4. Take the maximum value across the sender dimension
    agg_max, _ = m_masked.max(dim=2)
    
    # 5. If an agent is the sole survivor, all senders are blocked (-1e9). 
    # We must catch this and zero it out to prevent network corruption.
    all_blocked = blocked.all(dim=-1).unsqueeze(-1)
    agg_max = agg_max.masked_fill(all_blocked, 0.0)
    
    return agg_max


def _sender_mask(alive: torch.Tensor, n: int) -> torch.Tensor:
    """(B,N,N) boolean: True where sender j is NOT a valid source for i.

    Blocks self-messages and messages from dead agents.
    """
    dead = (alive < 0.5).unsqueeze(1).expand(-1, n, -1)          # (B,N,N) over j
    eye = torch.eye(n, device=alive.device, dtype=torch.bool)     # (N,N)
    return dead | eye.unsqueeze(0)


def _masked_softmax(scores: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis with `blocked` entries forced to zero weight.

    Rows in which EVERY entry is blocked (an agent that is the sole survivor)
    return all-zero weights rather than a uniform distribution. Without this,
    softmax over a row of equal -inf values softens to uniform and the lone
    survivor silently reads the hidden states of its dead team-mates.
    """
    scores = scores.masked_fill(blocked, -1e9)
    w = torch.softmax(scores, dim=-1)
    w = w.masked_fill(blocked, 0.0)
    all_blocked = blocked.all(dim=-1, keepdim=True)
    return w.masked_fill(all_blocked, 0.0)


class NoComm(nn.Module):
    """Control condition: no message is exchanged."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.last_message = None      # NoComm emits nothing
        self.last_gate = None
        self.last_kl = None

    def forward(self, h: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        return self.norm(h)


class FCComm(nn.Module):
    """Fully-connected broadcast of other living agents' messages.

    The aggregation math is selectable via pool_type:
      * "mean" -- average of all other living agents' messages. With gating on,
        blind agents send ~0 so the mean is dominated by the few that see
        something. This is the classic mean-pooling baseline.
      * "max"  -- element-wise maximum across other living agents. Under EPO
        this cannot be diluted by adding more blind agents: one strong signal
        survives regardless of how many near-zero messages surround it. Note
        max is sign-asymmetric -- it preserves the most POSITIVE activation, so
        the encoder learns to encode salient information as large positive
        values. Dead/self senders are masked to -1e9 (never 0), and a lone
        survivor with no valid sender returns 0.
    """

    def __init__(self, hidden: int, msg_dim: int, mlp: bool = True,
                 gate: bool = True, pool_type: str = "mean", channel_cfg: dict = None):
        super().__init__()
        if pool_type not in ("mean", "max"):
            raise ValueError(f"pool_type must be 'mean' or 'max', got {pool_type!r}")
        self.encoder = MessageEncoder(hidden, msg_dim, mlp=mlp, gate=gate, channel_cfg=channel_cfg)
        self.transform = _zero_init(nn.Linear(msg_dim, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.pool_type = pool_type

    @property
    def last_message(self):
        return self.encoder.last_message

    @property
    def last_gate(self):
        return self.encoder.last_gate

    @property
    def last_kl(self):
        return self.encoder.last_kl

    def forward(self, h, alive):
        msg = self.encoder(h)
        if self.pool_type == "max":
            agg = _max_excluding_self(msg, alive)
        else:
            agg = _mean_excluding_self(msg, alive)
        return self.norm(h + torch.tanh(self.transform(agg)))


class GATLayer(nn.Module):
    """One multi-head graph attention layer (Velickovic et al., ICLR 2018).

    This is ADDITIVE attention with a learned attention vector `a` and a
    LeakyReLU nonlinearity:

        e_ij = LeakyReLU( a_src . W h_i  +  a_dst . W h_j )
        alpha_ij = softmax_j( e_ij )   over living j != i
        h'_i = concat_heads( sum_j alpha_ij  W h_j )

    It is deliberately a different mechanism from the scaled dot-product
    attention in AttnComm, so the two experimental conditions test two
    genuinely different architectures rather than the same one twice.
    """

    def __init__(self, dim: int, heads: int, negative_slope: float = 0.2):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must divide evenly by heads ({heads})")
        self.heads = heads
        self.dk = dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dk))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, alive):
        B, N, _ = x.shape
        wh = self.W(x).view(B, N, self.heads, self.dk)               # (B,N,H,dk)
        # Per-head projections onto the learned attention vector.
        f_src = (wh * self.a_src).sum(-1)                            # (B,N,H)
        f_dst = (wh * self.a_dst).sum(-1)                            # (B,N,H)
        # e[b,h,i,j] = leaky( f_src[b,i,h] + f_dst[b,j,h] )
        e = self.leaky(f_src.permute(0, 2, 1).unsqueeze(-1)
                       + f_dst.permute(0, 2, 1).unsqueeze(-2))       # (B,H,N,N)

        blocked = _sender_mask(alive, N).unsqueeze(1).expand_as(e)
        alpha = _masked_softmax(e, blocked)                          # (B,H,N,N)

        v = wh.permute(0, 2, 1, 3)                                   # (B,H,N,dk)
        ctx = (alpha @ v).permute(0, 2, 1, 3).reshape(B, N, -1)      # (B,N,dim)
        return self.norm(x + torch.tanh(ctx))


class GATComm(nn.Module):
    """Multi-hop graph attention communication.

    Exposed on the command line as --comm gnn (and --comm gat). This is the
    condition the Terms of Reference calls "graph attention networks". It uses
    the same gated MLP encoder as FC to turn hidden states into messages, then
    passes them through GAT layers instead of a plain mean.
    """

    def __init__(self, hidden: int, msg_dim: int, heads: int, layers: int = 2,
                 mlp: bool = True, gate: bool = True, channel_cfg: dict = None):
        super().__init__()
        if layers < 1:
            raise ValueError(f"--num-layers must be >= 1 for the GAT condition; got {layers}")
        self.encoder = MessageEncoder(hidden, msg_dim, mlp=mlp, gate=gate, channel_cfg=channel_cfg)
        self.layers = nn.ModuleList([GATLayer(msg_dim, heads) for _ in range(layers)])
        self.out = _zero_init(nn.Linear(msg_dim, hidden))
        self.norm = nn.LayerNorm(hidden)

    @property
    def last_message(self):
        return self.encoder.last_message

    @property
    def last_gate(self):
        return self.encoder.last_gate

    @property
    def last_kl(self):
        return self.encoder.last_kl

    def forward(self, h, alive):
        m = self.encoder(h)
        for layer in self.layers:
            m = layer(m, alive)
        return self.norm(h + torch.tanh(self.out(m)))


class AttnComm(nn.Module):
    """Multi-head scaled dot-product attention across agents, self-masked.

    The query/key/value projections operate on a gated MLP-encoded message, so
    a blind agent's gate can suppress the value it broadcasts.
    """

    def __init__(self, hidden: int, msg_dim: int, heads: int = 4,
                 mlp: bool = True, gate: bool = True, channel_cfg: dict = None):
        super().__init__()
        if msg_dim % heads != 0:
            raise ValueError(f"--msg-dim ({msg_dim}) must divide evenly by "
                             f"--num-heads ({heads})")
        self.h = heads
        self.dk = msg_dim // heads
        self.encoder = MessageEncoder(hidden, msg_dim, mlp=mlp, gate=gate, channel_cfg=channel_cfg)
        self.q = nn.Linear(msg_dim, msg_dim)
        self.k = nn.Linear(msg_dim, msg_dim)
        self.v = nn.Linear(msg_dim, msg_dim)
        self.out = _zero_init(nn.Linear(msg_dim, hidden))
        self.norm = nn.LayerNorm(hidden)

    @property
    def last_message(self):
        return self.encoder.last_message

    @property
    def last_gate(self):
        return self.encoder.last_gate

    @property
    def last_kl(self):
        return self.encoder.last_kl

    def forward(self, h, alive):
        B, N, _ = h.shape
        m = self.encoder(h)
        q = self.q(m).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(m).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(m).view(B, N, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)        # (B,H,N,N)
        blocked = _sender_mask(alive, N).unsqueeze(1).expand_as(scores)
        w = _masked_softmax(scores, blocked)

        ctx = (w @ v).transpose(1, 2).reshape(B, N, -1)
        return self.norm(h + torch.tanh(self.out(ctx)))


def build_comm(name: str, hidden: int, msg_dim: int, heads: int, layers: int,
               mlp: bool = True, gate: bool = True,
               pool_type: str = "mean", channel_cfg: dict = None) -> nn.Module:
    # pool_type ("mean" | "max") selects the FC aggregation math. GAT and Attn
    # use attention-weighted aggregation, so pool_type does not apply to them;
    # it is accepted and ignored for those, which keeps one clean call site.
    if name == "none":
        return NoComm(hidden)
    if name == "fc":
        return FCComm(hidden, msg_dim, mlp=mlp, gate=gate, pool_type=pool_type, channel_cfg=channel_cfg)
    if name in ("gnn", "gat"):
        return GATComm(hidden, msg_dim, heads, layers, mlp=mlp, gate=gate, channel_cfg=channel_cfg)
    if name == "attn":
        return AttnComm(hidden, msg_dim, heads, mlp=mlp, gate=gate, channel_cfg=channel_cfg)
    raise ValueError(f"unknown communication mode: {name}")


# =============================================================================
# Recurrent policy  (paper ablation: recurrent clearly beats feed-forward)
# obs -> encoder -> GRU -> communication -> action logits
# One shared network for all agents; agent identity supplied via one-hot.
# =============================================================================
class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, comm, msg_dim, heads,
                 layers=2, comm_to_memory=True, mlp_encoder=True, gate=True,
                 aux_sighting=False, pool_type="mean", channel_cfg=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        # pool_type is an EXPLICIT parameter (not read from a global `args`),
        # so RecurrentPolicy stays self-contained and testable.
        self.comm = build_comm(comm, hidden, msg_dim, heads, layers,
                               mlp=mlp_encoder, gate=gate, pool_type=pool_type,
                               channel_cfg=channel_cfg)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # UPGRADE 3 (auxiliary sighting loss): a tiny head that reads an
        # agent's emitted message and predicts whether that agent currently
        # sees an enemy. Training it forces the message to CARRY that fact,
        # grounding the channel in something physical from step 1 instead of
        # leaving it to emerge from reward alone. Only built for communicating
        # conditions (NoComm emits no message).
        self.aux_sighting = aux_sighting and not isinstance(self.comm, NoComm)
        if self.aux_sighting:
            self.aux_head = nn.Linear(msg_dim, 1)
        self.hidden = hidden
        self.comm_to_memory = comm_to_memory

    def forward(self, obs, h_in, alive):
        """obs (B,N,obs_dim) - h_in (B,N,hidden) -> logits (B,N,A), h_out.

        With comm_to_memory=True the post-communication representation IS the
        recurrent state, so a message received at time t is still available at
        time t+1. With gating and MLP encoding OFF, every communication module
        (including NoComm) still ends in the same LayerNorm, so all conditions
        remain numerically identical at initialisation. (Gating starts mostly
        open and the message projections are zero-init, so the identity-at-init
        property is preserved with gating on as well.)
        """
        B, N, _ = obs.shape
        x = self.encoder(obs).reshape(B * N, -1)
        h_gru = self.gru(x, h_in.reshape(B * N, -1)).reshape(B, N, -1)
        z = self.comm(h_gru, alive)
        h_out = z if self.comm_to_memory else h_gru
        return self.head(z), h_out

    def aux_logits(self):
        """Per-agent sighting logit (B,N) from the last emitted message."""
        if not self.aux_sighting or self.comm.last_message is None:
            return None
        # BUG 1 FIX: Explicit dim=-1 guarantees shape (B, N) even when B=1
        return self.aux_head(self.comm.last_message).squeeze(dim=-1)

    def init_hidden(self, B, N, device):
        return torch.zeros(B, N, self.hidden, device=device)


class RecurrentCritic(nn.Module):
    """Centralised value function over the global state (training only)."""

    def __init__(self, state_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, state, h_in):
        x = self.encoder(state)
        h_out = self.gru(x, h_in)
        return self.head(h_out).squeeze(-1), h_out

    def init_hidden(self, B, device):
        return torch.zeros(B, self.hidden, device=device)


# =============================================================================
# Generalised advantage estimation
# =============================================================================
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device)
    running = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        running = delta + gamma * lam * nonterm * running
        adv[t] = running
    return adv, adv + values


# =============================================================================
# Environment construction
# =============================================================================
def make_env(mode: str, seed: int, n_units: int = None, n_enemies: int = 5,
             map_name: str = "10gen_protoss"):
    """Build SMACv2 to match the official configs shipped with the benchmark.

    Mirrors smacv2/examples/configs/sc2_gen_protoss.yaml (standard) and
    sc2_gen_protoss_epo.yaml (EPO). The ONLY differences between the two
    official files are:

        standard :  prob_obs_enemy = 1.0   action_mask = True
        epo      :  prob_obs_enemy = 0.0   action_mask = False

    conic_fov is False in BOTH official configs. EPO is produced by the
    stochastic enemy-observation probability together with the removal of the
    available-action mask, NOT by the conic field of view. With
    action_mask=False every living enemy appears as a valid attack target
    regardless of range, so agents must infer or communicate which enemies can
    actually be hit.
    """
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    epo = (mode == "epo")
    # The paper's headline EPO challenge uses 6 allies against 5 enemies, which
    # widened the gap between the p=1 and p=0 settings. The shipped YAML uses
    # 5v5; both are reported in the paper.
    if n_units is None:
        n_units = 6 if epo else 5

    unit_types = {
        "10gen_protoss": ["stalker", "zealot", "colossus"],
        "10gen_terran": ["marine", "marauder", "medivac"],
        "10gen_zerg": ["zergling", "baneling", "hydralisk"],
    }
    weights = {
        "10gen_protoss": [0.45, 0.45, 0.1],
        "10gen_terran": [0.45, 0.45, 0.1],
        "10gen_zerg": [0.45, 0.1, 0.45],   # baneling is the rare unit for Zerg
    }
    if map_name not in unit_types:
        raise ValueError(f"map_name must be one of {list(unit_types)}, got {map_name!r}")

    cfg = dict(
        map_name=map_name,
        difficulty="7",
        move_amount=2,
        step_mul=8,
        # observation / state, matching the official YAML
        obs_all_health=True,
        obs_own_health=True,
        obs_own_pos=True,
        obs_last_action=False,
        obs_instead_of_state=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_timestep_number=False,
        state_last_action=True,
        state_timestep_number=False,
        # ranges
        conic_fov=False,
        num_fov_actions=12,
        use_unit_ranges=True,
        min_attack_range=2,
        # reward shaping, matching the official YAML
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        # THE two settings that define EPO
        prob_obs_enemy=0.0 if epo else 1.0,
        action_mask=False if epo else True,
        capability_config={
            "n_units": n_units,
            "n_enemies": n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": unit_types[map_name],
                "weights": weights[map_name],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
            },
        },
        seed=seed,
    )
    return StarCraftCapabilityEnvWrapper(**cfg)


# =============================================================================
# Greedy evaluation  (paper: test win rate over 32 greedy episodes)
# =============================================================================
@torch.no_grad()
def evaluate(env, policy, obs_norm, agent_ids, n_episodes, ep_limit, device):
    """Run `n_episodes` argmax episodes and return (win_rate, mean_return).

    Runs on the training environment at a rollout boundary. The in-progress
    training episode is discarded and the caller must reset its own hidden
    states afterwards, which train() does.
    """
    policy.eval()
    N = agent_ids.shape[0]
    wins, rets = [], []
    for _ in range(n_episodes):
        env.reset()
        h = policy.init_hidden(1, N, device).squeeze(0)
        done, ep_ret, steps = False, 0.0, 0
        while not done:
            o = torch.as_tensor(np.asarray(env.get_obs(), dtype=np.float32),
                                device=device)
            o = torch.cat([o, agent_ids], dim=-1)
            av = torch.as_tensor(np.asarray(env.get_avail_actions(), dtype=np.float32),
                                 device=device)
            alive = (av[:, 0] < 0.5).float()
            logits, h_next = policy(obs_norm(o).unsqueeze(0), h.unsqueeze(0),
                                    alive.unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(av < 0.5, -1e9)
            act = logits.argmax(dim=-1)
            reward, terminated, info = env.step(act.tolist())
            ep_ret += float(reward)
            steps += 1
            h = h_next.squeeze(0)
            done = bool(terminated) or steps >= ep_limit
        wins.append(int(info.get("battle_won", False)))
        rets.append(ep_ret)
    policy.train()
    return float(np.mean(wins)), float(np.mean(rets))


# =============================================================================
# Training
# =============================================================================
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Seed the CUDA generators too. Without this, exact reproducibility can
    # break on GPU (and across multi-GPU nodes) even though the CPU and numpy
    # generators are seeded, because some kernels draw from the unseeded CUDA
    # RNG. No-op on CPU-only machines.
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = args._env_factory(args) if getattr(args, "_env_factory", None) else \
        make_env(args.mode, args.seed, args.n_units, args.n_enemies, args.map_name)

    info = env.get_env_info()
    N = info["n_agents"]
    A = info["n_actions"]
    raw_obs_dim = info["obs_shape"]
    state_dim = info["state_shape"]
    ep_limit = info["episode_limit"]

    # Agent identity for a shared policy (paper: "Obs Agent Id: True")
    obs_dim = raw_obs_dim + N
    agent_ids = torch.eye(N, device=device)

    print(f"[smacv2/{args.comm}] mode={args.mode} map={args.map_name} N={N} "
          f"actions={A} raw_obs={raw_obs_dim} obs+id={obs_dim} state={state_dim} "
          f"ep_limit={ep_limit} comm_to_memory={args.comm_to_memory} "
          f"device={device}", flush=True)
    if args.expect_obs_dim > 0 and raw_obs_dim != args.expect_obs_dim:
        raise ValueError(f"obs mismatch: env gave {raw_obs_dim}, "
                         f"--expect-obs-dim was {args.expect_obs_dim}")

    # --- enemy-feature slice within the raw observation (for aux sighting) ---
    # SMACv2 lays out each agent's obs as
    #   [ move_feats | enemy_feats | ally_feats | own_feats ]
    # and enemy_feats is all-zero unless an enemy is visible AND alive. So
    # "this agent sees an enemy" == "its enemy-feature block has any nonzero
    # entry". We locate that block using the env's own size helpers rather than
    # hardcoding offsets, with a graceful disable if the helpers are absent.
    enemy_slice = None
    if args.aux_coef > 0.0 and args.comm != "none":
        raw_env = getattr(env, "env", env)   # unwrap the capability wrapper
        try:
            move_dim = raw_env.get_obs_move_feats_size()
            en = raw_env.get_obs_enemy_feats_size()
            enemy_dim = en if isinstance(en, int) else int(np.prod(en))
            enemy_slice = (int(move_dim), int(move_dim) + int(enemy_dim))
            print(f"[smacv2/{args.comm}] aux sighting: enemy feats at raw obs "
                  f"cols [{enemy_slice[0]}:{enemy_slice[1]}]", flush=True)
        except Exception as e:
            print(f"[smacv2/{args.comm}] WARNING: could not locate enemy feats "
                  f"({e}); auxiliary sighting loss DISABLED.", flush=True)
            enemy_slice = None

    aux_on = enemy_slice is not None
    channel_cfg = {
        'quant_bits': args.quant_bits,
        'noise_std': args.noise_std,
        'dropout_p': args.agent_dropout,
        'kl_bottleneck': args.kl_coef > 0.0,
    }
    policy = RecurrentPolicy(obs_dim, A, args.hidden, args.comm, args.msg_dim,
                             args.num_heads, args.num_layers,
                             comm_to_memory=args.comm_to_memory,
                             mlp_encoder=args.mlp_encoder, gate=args.gate,
                             aux_sighting=aux_on,
                             pool_type=args.pool_type,
                             channel_cfg=channel_cfg).to(device)
    critic = RecurrentCritic(state_dim, args.hidden).to(device)
    opt_a = torch.optim.Adam(policy.parameters(), lr=args.lr_actor, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)

    # Normalise only the raw environment features; the trailing N one-hot
    # agent-ID columns are left as clean 0/1 flags (see RunningNorm docstring).
    obs_norm = RunningNorm(obs_dim, device, active_dim=raw_obs_dim)
    state_norm = RunningNorm(state_dim, device)

    # Plain, picklable config embedded in every checkpoint / final file. A
    # non-picklable attribute on `args` (e.g. an injected _env_factory) would
    # otherwise crash torch.save at the end of a multi-hour run.
    safe_config = {k: v for k, v in vars(args).items()
                   if not k.startswith("_")
                   and isinstance(v, (int, float, str, bool, type(None), list, tuple))}

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="msc-marl-comm",
                         name=f"smacv2_{args.mode}_{args.comm}_seed{args.seed}",
                         config=vars(args),
                         tags=["paper-matched", args.mode, args.comm])

    T = args.rollout_len
    L = args.chunk_len
    if T % L != 0:
        raise ValueError(
            f"--rollout-len ({T}) must be an exact multiple of --chunk-len ({L}); "
            f"otherwise the trailing {T % L} timesteps are silently discarded "
            f"from every update.")

    buf = {
        "obs": torch.zeros(T, N, obs_dim, device=device),
        "state": torch.zeros(T, state_dim, device=device),
        "avail": torch.zeros(T, N, A, device=device),
        "alive": torch.zeros(T, N, device=device),
        "act": torch.zeros(T, N, dtype=torch.long, device=device),
        "logp": torch.zeros(T, N, device=device),
        "val": torch.zeros(T, device=device),
        "rew": torch.zeros(T, device=device),
        "done": torch.zeros(T, device=device),
        "h_pol": torch.zeros(T, N, args.hidden, device=device),
        "h_cri": torch.zeros(T, args.hidden, device=device),
        "sight": torch.zeros(T, N, device=device),   # aux label: sees enemy?
    }

    def read_env():
        raw = np.asarray(env.get_obs(), dtype=np.float32)   # (N, raw_obs_dim)
        o = torch.as_tensor(raw, device=device)
        # Aux sighting label: does agent i currently see an enemy? True iff its
        # enemy-feature block is not all zeros (see enemy_slice comment above).
        if enemy_slice is not None:
            lo, hi = enemy_slice
            sight = torch.as_tensor(
                (np.abs(raw[:, lo:hi]).sum(axis=1) > 1e-6).astype(np.float32),
                device=device)
        else:
            sight = torch.zeros(o.shape[0], device=device)
        o = torch.cat([o, agent_ids], dim=-1)
        s = torch.as_tensor(np.asarray(env.get_state(), dtype=np.float32), device=device)
        av = torch.as_tensor(np.asarray(env.get_avail_actions(), dtype=np.float32),
                             device=device)
        # Exact liveness test. SMACv2 gives a dead unit [1, 0, 0, ...] and
        # explicitly forbids the no-op for a living unit ("cannot choose no-op
        # when alive"), so index 0 is a precise death flag. Summing the vector
        # instead would misclassify a living but fully blocked agent -- one
        # whose only legal action is "stop" -- as dead.
        alive = (av[:, 0] < 0.5).float()
        return o, s, av, alive, sight

    # ------------------------- checkpointing --------------------------
    # A checkpoint captures EVERYTHING needed to continue training exactly:
    # network weights, BOTH optimiser states (so Adam's momentum is not reset),
    # the running-normaliser statistics, the environment-step counter and the
    # RNG states. Saving only the weights would silently reset the optimiser
    # and normaliser on every resume and corrupt a chained multi-job run.
    out = Path(args.out_dir or
               f"artifacts/checkpoints/smacv2_{args.mode}_{args.comm}") / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "latest.pt"

    def save_checkpoint(step_count: int, final: bool = False) -> None:
        # Write to a temporary file and atomically rename, so a job killed
        # mid-write (e.g. at the wall-clock limit) can never leave a corrupt
        # checkpoint that breaks the next resume.
        payload = {
            "steps": step_count,
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "opt_a": opt_a.state_dict(),
            "opt_c": opt_c.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "state_norm": state_norm.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "wins_log": wins_log[-200:],
            "returns_log": returns_log[-200:],
            "lens_log": lens_log[-200:],
            "next_eval": next_eval,
            "config": safe_config,
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(ckpt_path)
        tag = "FINAL" if final else "checkpoint"
        print(f"[smacv2/{args.comm}] {tag} saved at step {step_count:,} "
              f"-> {ckpt_path}", flush=True)

    steps = 0
    ep_ret, ep_len = 0.0, 0
    returns_log, wins_log, lens_log = [], [], []
    next_eval = args.eval_interval if args.eval_episodes > 0 else float("inf")
    test_wr, test_ret = float("nan"), float("nan")
    wr = rr = ll = float("nan")   # defined up-front so an already-complete
                                  # resume (loop body never runs) can still save
    policy_loss = value_loss = ent_term = torch.zeros((), device=device)
    grad_updates = [0]   # single-element list so the update loop can mutate it

    # ------------------------- resume, if asked -----------------------
    # --resume with an existing latest.pt continues the run; --resume with no
    # checkpoint yet (the first job in a chain) simply starts fresh. This lets
    # the SAME command be resubmitted after every wall-clock timeout.
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"])
        critic.load_state_dict(ck["critic"])
        opt_a.load_state_dict(ck["opt_a"])
        opt_c.load_state_dict(ck["opt_c"])
        obs_norm.load_state_dict(ck["obs_norm"])
        state_norm.load_state_dict(ck["state_norm"])
        torch.set_rng_state(ck["torch_rng"].cpu()
                            if hasattr(ck["torch_rng"], "cpu") else ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        if torch.cuda.is_available() and "cuda_rng" in ck:
            try:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            except Exception:
                pass  # e.g. resuming on a different GPU count; not fatal
        steps = int(ck["steps"])
        wins_log = list(ck.get("wins_log", []))
        returns_log = list(ck.get("returns_log", []))
        lens_log = list(ck.get("lens_log", []))
        next_eval = ck.get("next_eval", next_eval)
        print(f"[smacv2/{args.comm}] RESUMED from {ckpt_path} at step {steps:,} "
              f"(target {args.total_steps:,})", flush=True)
        if steps >= args.total_steps:
            print(f"[smacv2/{args.comm}] already at target; nothing to do.",
                  flush=True)
    elif args.resume:
        print(f"[smacv2/{args.comm}] --resume set but no checkpoint at "
              f"{ckpt_path}; starting fresh.", flush=True)

    next_ckpt = ((steps // args.ckpt_interval) + 1) * args.ckpt_interval

    env.reset()
    obs_t, state_t, avail_t, alive_t, sight_t = read_env()
    h_pol = policy.init_hidden(1, N, device).squeeze(0)
    h_cri = critic.init_hidden(1, device).squeeze(0)
    t0 = time.time()
    steps_at_start = steps

    while steps < args.total_steps:
        # ---- linear learning-rate decay (paper applies decay to MAPPO) ----
        frac = max(0.0, 1.0 - steps / args.total_steps)
        for g in opt_a.param_groups:
            g["lr"] = args.lr_actor * frac
        for g in opt_c.param_groups:
            g["lr"] = args.lr_critic * frac

        # ---------------------------- collect ----------------------------
        for i in range(T):
            buf["obs"][i] = obs_t
            buf["state"][i] = state_t
            buf["avail"][i] = avail_t
            buf["alive"][i] = alive_t
            buf["sight"][i] = sight_t
            buf["h_pol"][i] = h_pol
            buf["h_cri"][i] = h_cri

            with torch.no_grad():
                logits, h_pol_next = policy(obs_norm(obs_t).unsqueeze(0),
                                            h_pol.unsqueeze(0),
                                            alive_t.unsqueeze(0))
                logits = logits.squeeze(0)
                # Always respect the environment's available-action vector. The
                # env itself decides how restrictive that vector is: with
                # action_mask=False (EPO) every living enemy is listed as a
                # valid target regardless of range, so the mask no longer tells
                # the agent what it can actually hit.
                logits = logits.masked_fill(avail_t < 0.5, -1e9)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                val, h_cri_next = critic(state_norm(state_t).unsqueeze(0),
                                         h_cri.unsqueeze(0))
                val = val.squeeze(0)

            reward, terminated, env_info = env.step(act.tolist())
            truncated = (ep_len + 1) >= ep_limit
            # Episode boundary of EITHER kind ends the return bootstrap. The
            # environment resets below, so allowing GAE to bootstrap across the
            # boundary would mix values from two different episodes.
            episode_over = bool(terminated) or bool(truncated)

            buf["act"][i] = act
            buf["logp"][i] = logp
            buf["val"][i] = val
            buf["rew"][i] = float(reward)
            buf["done"][i] = float(episode_over)

            ep_ret += float(reward)
            ep_len += 1
            steps += 1
            h_pol, h_cri = h_pol_next.squeeze(0), h_cri_next.squeeze(0)

            if episode_over:
                returns_log.append(ep_ret)
                wins_log.append(int(env_info.get("battle_won", False)))
                lens_log.append(ep_len)
                ep_ret, ep_len = 0.0, 0
                env.reset()
                h_pol = policy.init_hidden(1, N, device).squeeze(0)
                h_cri = critic.init_hidden(1, device).squeeze(0)

            obs_t, state_t, avail_t, alive_t, sight_t = read_env()

        # NOTE: the normaliser statistics must NOT be refreshed yet. The stored
        # log-probabilities were produced under the current statistics; if the
        # statistics moved before the update, the recomputed log-probabilities
        # would see differently scaled inputs and the PPO ratio would reflect
        # the normalisation shift rather than the policy change. The statistics
        # are refreshed after the update instead.
        with torch.no_grad():
            last_val, _ = critic(state_norm(state_t).unsqueeze(0), h_cri.unsqueeze(0))
            last_val = last_val.squeeze(0)

        adv, ret = compute_gae(buf["rew"], buf["val"], buf["done"],
                               last_val, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------------------- update (chunked BPTT) ----------------------
        n_chunks = T // L
        chunk_ids = np.arange(n_chunks)
        obs_n = obs_norm(buf["obs"])
        state_n = state_norm(buf["state"])

        for _ in range(args.ppo_epochs):
            np.random.shuffle(chunk_ids)
            for c in chunk_ids:
                s0, s1 = int(c) * L, (int(c) + 1) * L
                hp = buf["h_pol"][s0].unsqueeze(0)
                hc = buf["h_cri"][s0].unsqueeze(0)

                logits_seq, vals_seq = [], []
                aux_seq = []
                kl_seq = []
                for t in range(s0, s1):
                    lg, hp = policy(obs_n[t].unsqueeze(0), hp,
                                    buf["alive"][t].unsqueeze(0))
                    logits_seq.append(lg.squeeze(0))
                    if policy.aux_sighting:
                        al = policy.aux_logits()          # (1,N) or None
                        aux_seq.append(None if al is None else al.squeeze(0))
                    lk = getattr(policy.comm, 'last_kl', None)
                    if lk is not None:
                        kl_seq.append(lk)
                    v, hc = critic(state_n[t].unsqueeze(0), hc)
                    vals_seq.append(v.reshape(()))
                    # The collector zeroed the recurrent state after an episode
                    # ended. The replay must do the same, or the recomputed
                    # log-probabilities drift from the ones that were stored and
                    # the PPO ratio becomes meaningless.
                    if buf["done"][t] > 0.5:
                        hp = torch.zeros_like(hp)
                        hc = torch.zeros_like(hc)

                logits = torch.stack(logits_seq)                 # (L,N,A)
                vals = torch.stack(vals_seq)                     # (L,)
                logits = logits.masked_fill(buf["avail"][s0:s1] < 0.5, -1e9)

                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(buf["act"][s0:s1])
                entropy = dist.entropy()

                alive = buf["alive"][s0:s1]                      # death masking
                denom = alive.sum().clamp(min=1.0)

                ratio = (new_logp - buf["logp"][s0:s1]).exp()
                a = adv[s0:s1].unsqueeze(1).expand(-1, N)
                s_1 = ratio * a
                s_2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                policy_loss = -(torch.min(s_1, s_2) * alive).sum() / denom
                ent_term = (entropy * alive).sum() / denom
                value_loss = F.mse_loss(vals, ret[s0:s1])

                # UPGRADE 3: auxiliary sighting loss. Predict, from each living
                # agent's emitted message, whether it currently sees an enemy.
                # Binary cross-entropy, masked to living agents. This grounds
                # the message in a real physical fact from the start of
                # training. Off when --aux-coef 0 or for the no-comm baseline.
                aux_loss = torch.zeros((), device=device)
                if policy.aux_sighting and aux_seq and all(a is not None for a in aux_seq):
                    aux_logits = torch.stack(aux_seq)            # (L,N)
                    aux_target = buf["sight"][s0:s1]             # (L,N)
                    per = F.binary_cross_entropy_with_logits(
                        aux_logits, aux_target, reduction="none")
                    aux_loss = (per * alive).sum() / denom

                kl_term = (torch.stack(kl_seq).mean()
                           if kl_seq else torch.zeros((), device=device))
                loss = (policy_loss
                        + args.vf_coef * value_loss
                        - args.ent_coef * ent_term
                        + args.aux_coef * aux_loss
                        + args.kl_coef * kl_term)

                opt_a.zero_grad(set_to_none=True)
                opt_c.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)

                # Optional diagnostic: confirm gradients are actually reaching
                # the communication module during real training. This LOGS and
                # continues -- it never calls sys.exit(), so it is safe to leave
                # on. (The test suite already proves non-zero comm gradients
                # before training; this is for spotting a mid-run collapse.)
                if args.grad_log_interval > 0 and not isinstance(policy.comm, NoComm):
                    grad_updates[0] += 1
                    if grad_updates[0] % args.grad_log_interval == 0:
                        gs = [p.grad.abs().mean().item()
                              for p in policy.comm.parameters()
                              if p.grad is not None]
                        gmean = float(np.mean(gs)) if gs else 0.0
                        flag = "OK" if gmean > 0 else "ZERO-GRAD!"
                        print(f"    [grad] comm mean|grad|={gmean:.3e} {flag}",
                              flush=True)
                        if run is not None:
                            run.log({"step": steps, "comm_grad_mean": gmean})

                opt_a.step()
                opt_c.step()

        # Statistics are refreshed only now, so the whole update used exactly
        # the same normalisation as data collection.
        obs_norm.update(buf["obs"])
        state_norm.update(buf["state"])

        # --------------------- greedy evaluation -------------------------
        if steps >= next_eval:
            test_wr, test_ret = evaluate(env, policy, obs_norm, agent_ids,
                                         args.eval_episodes, ep_limit, device)
            next_eval += args.eval_interval
            # evaluate() left the env at the start of a fresh episode; the
            # partially collected training episode is gone, so start clean.
            env.reset()
            h_pol = policy.init_hidden(1, N, device).squeeze(0)
            h_cri = critic.init_hidden(1, device).squeeze(0)
            ep_ret, ep_len = 0.0, 0
            obs_t, state_t, avail_t, alive_t, sight_t = read_env()

        # ----------------- periodic checkpoint (resume safety) -----------
        # Written every --ckpt-interval steps so a wall-clock timeout loses at
        # most that many steps, not the whole run. Set --ckpt-interval 0 to
        # disable (not recommended on a time-limited cluster).
        if args.ckpt_interval > 0 and steps >= next_ckpt:
            save_checkpoint(steps)
            next_ckpt += args.ckpt_interval

        # ------------------------------ log ------------------------------
        wr = float(np.mean(wins_log[-20:])) if wins_log else float("nan")
        rr = float(np.mean(returns_log[-20:])) if returns_log else float("nan")
        ll = float(np.mean(lens_log[-20:])) if lens_log else float("nan")
        # fps over THIS job's own steps, so a resumed job reports its true
        # throughput rather than being skewed by the pre-resume step count.
        fps = int((steps - steps_at_start) / (time.time() - t0 + 1e-9))
        print(f"steps={steps:>8d}  train_wr={wr:.3f}  test_wr={test_wr:.3f}  "
              f"ep_ret={rr:6.2f}  ep_len={ll:5.1f}  fps={fps}", flush=True)
        if run is not None:
            run.log({"step": steps,
                     "train_win_rate_last20": wr,
                     "test_win_rate": test_wr,
                     "test_return": test_ret,
                     "ep_return_last20": rr, "ep_length_last20": ll,
                     "policy_loss": policy_loss.item(),
                     "value_loss": value_loss.item(),
                     "entropy": ent_term.item(), "fps": fps})

    # -------------------- final evaluation + save ------------------------
    if args.eval_episodes > 0:
        test_wr, test_ret = evaluate(env, policy, obs_norm, agent_ids,
                                     args.final_eval_episodes, ep_limit, device)
        print(f"[smacv2/{args.comm}] FINAL greedy test_win_rate={test_wr:.4f} "
              f"over {args.final_eval_episodes} episodes", flush=True)
    env.close()

    # Persist the resumable checkpoint at the true final step, so that a
    # re-submitted --resume job sees the run is complete and exits immediately
    # instead of redoing work.
    save_checkpoint(steps, final=True)

    # And a compact, human-facing results file (weights + headline metrics).
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                "args": safe_config,
                "steps": steps,
                "final_test_win_rate": test_wr,
                "final_test_return": test_ret,
                "final_train_win_rate": wr}, out / "final.pt")
    print(f"[smacv2/{args.comm}] saved -> {out}/final.pt")
    if run is not None:
        run.finish()
    return {"test_win_rate": test_wr, "train_win_rate": wr}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--comm", choices=["none", "fc", "gnn", "gat", "attn"], required=True,
                   help="'gnn' and 'gat' are the same condition: a graph attention network")
    p.add_argument("--pool-type", dest="pool_type", choices=["mean", "max"],
                   default="max",
                                  help="FC aggregation math: 'mean' (average) or 'max' "
                    "(element-wise maximum). Applies to --comm fc only; "
                    "ignored by gnn/attn. Default 'max'.")
    p.add_argument("--mode", choices=["standard", "epo"], default="standard")
    p.add_argument("--map-name", default="10gen_protoss",
                   choices=["10gen_protoss", "10gen_terran", "10gen_zerg"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-units", type=int, default=None,
                   help="allies; default 5 standard / 6 epo (paper EPO is 6v5)")
    p.add_argument("--n-enemies", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--expect-obs-dim", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-wandb", action="store_true")
    # --- checkpointing / resume (for time-limited HPC jobs) ---
    p.add_argument("--resume", action="store_true",
                   help="continue from <out-dir>/seed<seed>/latest.pt if it "
                        "exists; the SAME command can be resubmitted after a "
                        "wall-clock timeout to keep training")
    p.add_argument("--ckpt-interval", type=int, default=200_000,
                   help="environment steps between checkpoint writes; 0 disables")
    p.add_argument("--grad-log-interval", type=int, default=0,
                   help="if >0, log mean |grad| of the comm module every N "
                        "updates (diagnostic; 0 = off)")
    # --- architecture ---
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--msg-dim", type=int, default=64)
    p.add_argument("--num-heads", "--heads", dest="num_heads", type=int, default=4)
    p.add_argument("--num-layers", dest="num_layers", type=int, default=2,
                   help="number of GAT hops; ignored by other conditions")
    p.add_argument("--comm-to-memory", dest="comm_to_memory",
                   action="store_true", default=True,
                   help="(default) messages enter the recurrent state")
    p.add_argument("--no-comm-to-memory", dest="comm_to_memory",
                   action="store_false",
                   help="ablation: messages affect only the action head")
    # --- message-pathway upgrades (all inside ToR: architecture, not channel
    #     constraints). Defaults ON for the communicating conditions. ---
    p.add_argument("--gate", dest="gate", action="store_true", default=True,
                   help="(default) learned sigmoid gate: agents learn when to "
                        "stay silent, so blind agents don't dilute the channel")
    p.add_argument("--no-gate", dest="gate", action="store_false",
                   help="ablation: disable message gating")
    p.add_argument("--mlp-encoder", dest="mlp_encoder", action="store_true",
                   default=True, help="(default) MLP message encoder")
    p.add_argument("--no-mlp-encoder", dest="mlp_encoder", action="store_false",
                   help="ablation: single-linear message encoder")
    p.add_argument("--aux-coef", type=float, default=0.2,
                   help="weight of the auxiliary 'do I see an enemy?' loss "
                        "(Upgrade 3); 0 disables it. Ignored for --comm none.")
    # --- evaluation (paper: 32 greedy test episodes) ---
    p.add_argument("--eval-episodes", type=int, default=32,
                   help="greedy episodes per evaluation; 0 disables evaluation")
    p.add_argument("--final-eval-episodes", type=int, default=32)
    p.add_argument("--eval-interval", type=int, default=10_000,
                   help="environment steps between greedy evaluations")
    # --- optimisation (Ellis et al. 2023, Table 5) ---
    p.add_argument("--rollout-len", type=int, default=400)
    p.add_argument("--chunk-len", type=int, default=40)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.99)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    # ----- channel constraints (ToR: bandwidth / KL bottleneck / noise / dropout) -----
    # All default OFF, so omitting them reproduces current behaviour exactly.
    p.add_argument("--quant-bits", type=int, default=0,
                   help="channel: bits per message dim (0=off; deterministic, PPO-safe)")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="channel: Gaussian message-noise std (0=off; test-time robustness)")
    p.add_argument("--agent-dropout", type=float, default=0.0,
                   help="channel: prob of muting a whole sender (0=off; test-time robustness)")
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="channel: KL info-bottleneck weight (0=off; samples in train, mean at eval)")
    return p


def build_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.comm == "gat":
        args.comm = "gnn"          # one canonical name in output paths
    if args.chunk_len < 2:
        raise ValueError("--chunk-len must be >= 2 for BPTT to mean anything")
    return args


if __name__ == "__main__":
    train(build_args())


help
