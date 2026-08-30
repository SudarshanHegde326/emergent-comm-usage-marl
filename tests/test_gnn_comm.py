from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.comms.gnn_comm import GNNAggregator, GNNCommPipeline
from src.comms.channels import IdentityChannel

B, N, D, A = 2, 3, 8, 5


def test_shape_forward_pass():
    obs_dim = 18
    pipe = GNNCommPipeline(obs_dim=obs_dim, msg_dim=D, action_dim=A, num_layers=2, hidden=64)
    logits, msgs, msgs_post, agg, per_layer = pipe(torch.randn(B, N, obs_dim))
    assert logits.shape == (B, N, A)
    assert msgs.shape == (B, N, D)
    assert agg.shape == (B, N, D)
    assert len(per_layer) == 3


def test_no_self_loop():
    agg_mod = GNNAggregator(msg_dim=D, num_layers=1, hidden=D)
    h = torch.randn(B, N, D)
    agg = agg_mod._mean_neighbour_aggregate(h)
    for b in range(B):
        for i in range(N):
            others = torch.cat([h[b, :i, :], h[b, i+1:, :]], dim=0)
            expected = others.mean(dim=0)
            torch.testing.assert_close(agg[b, i], expected)


def test_gradient_flow():
    obs_dim = 18
    pipe = GNNCommPipeline(obs_dim=obs_dim, msg_dim=D, action_dim=A, num_layers=2, hidden=64)
    logits, _, _, _, _ = pipe(torch.randn(B, N, obs_dim))
    logits.pow(2).sum().backward()
    for name, p in pipe.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} missed gradient flow"


def test_n_equals_1_no_nan():
    agg_mod = GNNAggregator(msg_dim=D, num_layers=2, hidden=D)
    out, _ = agg_mod(torch.randn(B, 1, D))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("L", [0, 1, 2])
def test_residual_shape_safe(L):
    agg_mod = GNNAggregator(msg_dim=D, num_layers=L, hidden=D)
    out, per_layer = agg_mod(torch.randn(B, N, D))
    assert out.shape == (B, N, D)
    assert len(per_layer) == L + 1


def test_l_zero_passthrough():
    agg_mod = GNNAggregator(msg_dim=D, num_layers=0, hidden=D)
    h = torch.randn(B, N, D)
    out, _ = agg_mod(h)
    assert torch.equal(out, h)


def test_with_identity_channel_no_op():
    obs_dim = 18
    torch.manual_seed(0)
    pipe_default = GNNCommPipeline(obs_dim=obs_dim, msg_dim=D, action_dim=A, num_layers=2, hidden=64, channel=None)
    torch.manual_seed(0)
    pipe_explicit = GNNCommPipeline(obs_dim=obs_dim, msg_dim=D, action_dim=A, num_layers=2, hidden=64, channel=IdentityChannel(msg_dim=D))
    obs = torch.randn(B, N, obs_dim)
    assert torch.equal(pipe_default(obs)[0], pipe_explicit(obs)[0])
