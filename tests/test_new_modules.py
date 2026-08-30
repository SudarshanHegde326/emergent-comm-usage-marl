"""Tests for the new channel-constraints + analysis modules.

Run: pytest tests/test_new_modules.py -v   (no StarCraft needed)
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.channel_constraints import ChannelConstraint, quantize_ste, set_channel_cfg
from src.analysis.mine import estimate_mine, estimate_infonce
from src.analysis.metrics import (hoyer_sparsity, activation_sparsity,
                                  gate_open_rate, hedges_g, bootstrap_ci)


# ------------------------- channel constraints ---------------------------

def test_identity_when_all_off():
    """Defaults must be a numerical no-op and expose zero KL."""
    cc = ChannelConstraint(msg_dim=8)
    assert cc.is_identity
    m = torch.randn(4, 5, 8)
    out = cc(m)
    assert torch.equal(out, m)
    assert float(cc.last_kl) == 0.0


def test_quantise_reduces_unique_values_but_keeps_shape():
    m = torch.randn(3, 5, 8)
    q = quantize_ste(m, bits=2, value_range=2.0)      # 4 levels
    assert q.shape == m.shape
    assert torch.unique(q).numel() <= 4 + 1           # <=4 levels (+possible clip)


def test_quantise_straight_through_gradient():
    m = torch.randn(2, 3, 4, requires_grad=True)
    q = quantize_ste(m, bits=3, value_range=2.0)
    q.sum().backward()
    # STE => gradient is all ones (identity backward).
    assert torch.allclose(m.grad, torch.ones_like(m))


def test_agent_dropout_mutes_whole_senders():
    torch.manual_seed(0)
    cc = ChannelConstraint(msg_dim=6, dropout_p=0.5)
    cc.train()
    m = torch.randn(1, 100, 6)
    out = cc(m)
    row_norms = out.abs().sum(-1)[0]                  # (100,)
    muted = (row_norms == 0).float().mean().item()
    assert 0.3 < muted < 0.7                          # ~50% senders muted


def test_kl_bottleneck_exposes_positive_kl_and_is_deterministic_at_eval():
    torch.manual_seed(0)
    cc = ChannelConstraint(msg_dim=8, kl_bottleneck=True)
    m = torch.randn(4, 5, 8)
    cc.train()
    _ = cc(m)
    assert float(cc.last_kl) >= 0.0
    # eval mode: uses the mean, so two forwards are identical (no sampling).
    cc.eval()
    a, b = cc(m), cc(m)
    assert torch.equal(a, b)


def test_set_channel_cfg_updates_in_place():
    torch.nn = __import__("torch").nn
    net = torch.nn.Sequential(ChannelConstraint(8), ChannelConstraint(8))
    n = set_channel_cfg(net, noise_std=0.7, dropout_p=0.2)
    assert n == 2
    for mod in net:
        assert mod.noise_std == 0.7 and mod.dropout_p == 0.2


# ------------------------------ MINE -------------------------------------

def test_mine_recovers_known_gaussian_mi():
    """For a bivariate Gaussian with correlation rho, I = -0.5*log(1-rho^2).
    MINE and InfoNCE should both land in the right neighbourhood."""
    torch.manual_seed(0)
    rho, S = 0.8, 4000
    true_mi = -0.5 * np.log(1 - rho ** 2)             # ~0.51 nats
    z = torch.randn(S, 1)
    x = z
    y = rho * z + np.sqrt(1 - rho ** 2) * torch.randn(S, 1)
    mine = estimate_mine(x, y, iters=1500, seed=0)
    nce = estimate_infonce(x, y, iters=1500, seed=0)
    assert abs(mine - true_mi) < 0.2, (mine, true_mi)
    assert abs(nce - true_mi) < 0.2, (nce, true_mi)


def test_mine_near_zero_for_independent():
    torch.manual_seed(0)
    x = torch.randn(4000, 1)
    y = torch.randn(4000, 1)                          # independent
    mi = estimate_mine(x, y, iters=1200, seed=0)
    assert abs(mi) < 0.1, mi


# ---------------------------- sparsity -----------------------------------

def test_hoyer_extremes():
    onehot = torch.zeros(10, 16); onehot[:, 0] = 5.0
    uniform = torch.ones(10, 16)
    assert hoyer_sparsity(onehot) > 0.95
    assert hoyer_sparsity(uniform) < 0.05


def test_activation_and_gate():
    m = torch.zeros(4, 8); m[:, 0] = 1.0
    assert abs(activation_sparsity(m) - 7 / 8) < 1e-6
    assert gate_open_rate(torch.tensor([0.9, 0.1, 0.8, 0.2])) == 0.5
    assert np.isnan(gate_open_rate(None))


# --------------------------- effect sizes --------------------------------

def test_hedges_g_sign_and_correction():
    a = np.array([1.0, 1.1, 0.9])          # higher group
    b = np.array([0.0, 0.1, -0.1])
    g = hedges_g(a, b)
    assert g > 0                            # a > b
    # correction shrinks |g| relative to raw Cohen's d
    na = nb = 3
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1))/(na+nb-2))
    d = (a.mean()-b.mean())/pooled
    assert abs(g) < abs(d)


def test_bootstrap_ci_orders():
    a = np.array([1.0, 1.2, 0.8, 1.1])
    b = np.array([0.0, 0.1, -0.1, 0.05])
    lo, hi = bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert lo <= hi


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
