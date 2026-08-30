"""Unit tests verifying the additive properties of NoiseChannel."""

from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.comms.noise_channel import NoiseChannel

B, N, D = 4, 3, 8


@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_shape_preserved(sigma):
    ch = NoiseChannel(msg_dim=D, sigma=sigma)
    msgs = torch.randn(B, N, D)
    out = ch(msgs)
    assert out.shape == msgs.shape


def test_sigma_zero_passthrough():
    ch = NoiseChannel(msg_dim=D, sigma=0.0)
    msgs = torch.randn(B, N, D)
    out = ch(msgs)
    assert torch.equal(out, msgs), "sigma=0 must return bit-exact identity data streams"


def test_gradient_flow():
    ch = NoiseChannel(msg_dim=D, sigma=0.25)
    msgs = torch.randn(B, N, D, requires_grad=True)
    out = ch(msgs)
    loss = out.sum()
    loss.backward()
    assert msgs.grad is not None
    assert torch.allclose(msgs.grad, torch.ones_like(msgs))


def test_eval_mode_still_noisy():
    """Verify that eval mode persistently applies random medium perturbations."""
    ch = NoiseChannel(msg_dim=D, sigma=0.25)
    ch.eval()
    msgs = torch.randn(B, N, D)
    out_a = ch(msgs)
    out_b = ch(msgs)
    assert not torch.allclose(out_a, out_b), "Eval cycles must retain active noise sampling paths"


def test_noise_statistics():
    """Verify empirical standard deviation distributions match sigma."""
    torch.manual_seed(0)
    sigma = 0.25
    ch = NoiseChannel(msg_dim=D, sigma=sigma)
    B_big = 256
    msgs = torch.zeros(B_big, N, D)
    out = ch(msgs)
    empirical_std = out.std().item()
    assert abs(empirical_std - sigma) / sigma < 0.10
