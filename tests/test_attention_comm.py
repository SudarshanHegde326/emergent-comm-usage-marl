"""Automated verification suite checking shapes, self-masking, and gradients for attention modules.
File: tests/test_attention_comm.py
"""

from __future__ import annotations
import pytest
import torch
from src.comms.attention_comm import AttentionCommPipeline, MultiHeadAttentionAggregator

def test_aggregator_output_shape():
    """Confirms output tensor dimension tensors conform to structural constraints."""
    agg = MultiHeadAttentionAggregator(msg_dim=8, num_heads=2)
    msgs = torch.randn(4, 3, 8)  # Batch=4, Agents=3, Dimension=8
    out, w = agg(msgs)
    assert out.shape == (4, 3, 8)
    assert w.shape == (4, 2, 3, 3)  # Shape constraint matching: (B, H, N, N)

def test_pipeline_output_shapes():
    """Validates full end-to-end forward pass matching structural constraints."""
    pipe = AttentionCommPipeline(obs_dim=18, msg_dim=8, action_dim=5, num_heads=2)
    obs = torch.randn(2, 3, 18)
    logits, msgs, agg, w = pipe(obs)
    assert logits.shape == (2, 3, 5)
    assert msgs.shape == (2, 3, 8)
    assert agg.shape == (2, 3, 8)
    assert w.shape == (2, 2, 3, 3)

def test_msg_dim_must_be_divisible_by_heads():
    """Verifies assertion guards identify incompatible head configurations."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttentionAggregator(msg_dim=7, num_heads=2)

def test_self_mask_zero_weights_on_diagonal():
    """Verifies self-mask layer forces diagonal index elements strictly to zero weight."""
    agg = MultiHeadAttentionAggregator(msg_dim=8, num_heads=2)
    msgs = torch.randn(3, 4, 8)
    _, w = agg(msgs)
    N = msgs.shape[1]
    eye = torch.eye(N, dtype=torch.bool)
    diag_values = w[..., eye]  # Shape constraint matching: (B, H, N)
    assert torch.allclose(diag_values, torch.zeros_like(diag_values), atol=1e-7)

def test_softmax_weights_sum_to_one():
    """Confirms incoming weight matrices correctly normalize row matrices down to 1.0."""
    agg = MultiHeadAttentionAggregator(msg_dim=8, num_heads=2)
    msgs = torch.randn(3, 4, 8)
    _, w = agg(msgs)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_gradients_flow_to_message_head():
    """Validates backward propagation layers compute gradients cleanly back to the input structures."""
    pipe = AttentionCommPipeline(obs_dim=4, msg_dim=4, action_dim=2, num_heads=2)
    obs = torch.randn(2, 3, 4)
    logits, _, _, _ = pipe(obs)
    loss = logits.pow(2).mean()
    loss.backward()
    msg_grads = [p.grad for p in pipe.msg_head.parameters()]
    assert all(g is not None for g in msg_grads)
    assert any((g.abs().sum() > 0).item() for g in msg_grads)

def test_permutation_equivariance_of_aggregator():
    """Confirms row-index shuffling maps consistently without breaking tensor shapes."""
    agg = MultiHeadAttentionAggregator(msg_dim=8, num_heads=2)
    msgs = torch.randn(1, 4, 8)
    out_orig, _ = agg(msgs)
    perm = torch.tensor([2, 0, 3, 1])
    msgs_p = msgs[:, perm, :]
    out_perm, _ = agg(msgs_p)
    assert torch.allclose(out_perm, out_orig[:, perm, :], atol=1e-5)
