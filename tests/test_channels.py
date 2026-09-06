"""Unit tests for src.comms.channels.

Verifies identity passthrough, low-rank tensor preservation, straight-through
gradient properties, and factory creation methods.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.comms.channels import (
    IdentityChannel,
    LowRankChannel,
    QuantizedChannel,
    build_channel,
)

B, N, D = 4, 3, 8


def test_identity_passthrough():
    ch = IdentityChannel(msg_dim=D)
    msgs = torch.randn(B, N, D)
    out = ch(msgs)
    assert out.shape == msgs.shape
    assert torch.equal(out, msgs), "Identity channel must not modify values"


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_lowrank_shape_preserved(K):
    ch = LowRankChannel(msg_dim=D, bandwidth=K)
    msgs = torch.randn(B, N, D)
    out = ch(msgs)
    assert out.shape == (B, N, D)


def test_lowrank_gradient_flow():
    K = 4
    ch = LowRankChannel(msg_dim=D, bandwidth=K)
    msgs = torch.randn(B, N, D, requires_grad=True)
    out = ch(msgs)
    loss = out.sum()
    loss.backward()
    assert ch.down.weight.grad is not None
    assert ch.up.weight.grad is not None
    assert ch.down.weight.grad.abs().sum() > 0
    assert ch.up.weight.grad.abs().sum() > 0
    assert msgs.grad is not None
    assert msgs.grad.shape == msgs.shape


def test_quantized_ste():
    K_levels = 4
    ch = QuantizedChannel(msg_dim=D, bandwidth=K_levels)
    msgs = torch.randn(B, N, D, requires_grad=True)
    out = ch(msgs)

    assert out.shape == msgs.shape

    scale = (K_levels - 1) / 2.0
    allowed_levels = torch.linspace(-1.0, 1.0, K_levels)
    out_flat = out.detach().reshape(-1)
    dists = (out_flat.unsqueeze(1) - allowed_levels.unsqueeze(0)).abs()
    min_dists = dists.min(dim=1).values
    assert torch.all(min_dists < 1e-5)

    loss = out.sum()
    loss.backward()
    assert msgs.grad is not None
    assert msgs.grad.shape == msgs.shape
    assert msgs.grad.abs().sum() > 0


@pytest.mark.parametrize("name,cls", [
    ("none",     IdentityChannel),
    ("identity", IdentityChannel),
    ("low-rank", LowRankChannel),
    ("lowrank",  LowRankChannel),
    ("quantized", QuantizedChannel),
])
def test_factory(name, cls):
    bw = 4 if cls is not IdentityChannel else 0
    ch = build_channel(name, msg_dim=D, bandwidth=bw)
    assert isinstance(ch, cls)
