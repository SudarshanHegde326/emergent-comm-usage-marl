"""Unit tests for src.comms.ib_channel.StochasticBottleneckChannel.

Four tests covering shape preservation, non-negative KL tracking bounds,
reparameterization backpropagation gradients, and mode determinism.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.comms.ib_channel import StochasticBottleneckChannel

B, N, D = 4, 3, 8


def test_shape_preserved():
    ch = StochasticBottleneckChannel(msg_dim=D)
    msgs = torch.randn(B, N, D)
    out = ch(msgs)
    assert out.shape == (B, N, D), f"output shape {out.shape} != expected"


def test_kl_nonneg_and_finite():
    ch = StochasticBottleneckChannel(msg_dim=D)
    msgs = torch.randn(B, N, D)
    ch(msgs)
    kl = ch.kl_loss()
    assert kl.ndim == 0, f"kl_loss must be scalar; got shape {kl.shape}"
    assert torch.isfinite(kl), f"KL is not finite: {kl.item()}"
    assert kl.item() >= -1e-6, f"KL is negative: {kl.item()}"


def test_gradient_flow_through_reparam():
    ch = StochasticBottleneckChannel(msg_dim=D)
    msgs = torch.randn(B, N, D, requires_grad=True)
    ch.train()
    out = ch(msgs)
    loss = out.sum() + ch.kl_loss()
    loss.backward()
    assert ch.mu_head.weight.grad is not None, "mu_head has no grad"
    assert ch.logvar_head.weight.grad is not None, "logvar_head has no grad"
    assert ch.mu_head.weight.grad.abs().sum() > 0, "mu_head grad is zero"
    assert ch.logvar_head.weight.grad.abs().sum() > 0, "logvar_head grad is zero"
    assert msgs.grad is not None and msgs.grad.shape == msgs.shape


def test_eval_mode_deterministic():
    ch = StochasticBottleneckChannel(msg_dim=D)
    msgs = torch.randn(B, N, D)

    # 4a. eval mode -> two calls give identical outputs (no sampling)
    ch.eval()
    out1 = ch(msgs)
    out2 = ch(msgs)
    assert torch.allclose(out1, out2), "Eval mode should return the deterministic encoder mean"

    # 4b. train mode -> two calls give DIFFERENT outputs (sampling)
    ch.train()
    out3 = ch(msgs)
    out4 = ch(msgs)
    assert not torch.allclose(out3, out4), "Train mode must sample stochastically"
