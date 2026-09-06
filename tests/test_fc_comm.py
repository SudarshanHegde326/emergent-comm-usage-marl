import sys
from pathlib import Path
import pytest
import torch

from src.comms.fc_comm import FCCommPipeline, FCAggregator

def test_pipeline_shapes():
    pipe = FCCommPipeline(obs_dim=18, msg_dim=8, action_dim=5)
    obs_NB = torch.randn(2, 3, 18) # Batch=2, Agents=3, Obs=18
    logits, msgs, agg = pipe(obs_NB)
    assert logits.shape == (2, 3, 5)
    assert msgs.shape == (2, 3, 8)
    assert agg.shape == (2, 3, 8)

def test_aggregator_excludes_self():
    """Confirms agent i receives information ONLY from peers j != i."""
    agg = FCAggregator("sum")
    msgs = torch.zeros(1, 3, 3)
    msgs[0, 0] = torch.tensor([1.0, 0.0, 0.0]) # Agent 0 signal
    msgs[0, 1] = torch.tensor([0.0, 1.0, 0.0]) # Agent 1 signal
    msgs[0, 2] = torch.tensor([0.0, 0.0, 1.0]) # Agent 2 signal

    out = agg(msgs)[0]

    # Agent 0 must only see Agent 1 + Agent 2 -> [0.0, 1.0, 1.0]
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 1.0]))
    # Agent 1 must only see Agent 0 + Agent 2 -> [1.0, 0.0, 1.0]
    assert torch.allclose(out[1], torch.tensor([1.0, 0.0, 1.0]))

def test_gradient_flows_to_message_head():
    """Ensures policy optimization gradients can flow back into message parameters."""
    pipe = FCCommPipeline(obs_dim=4, msg_dim=3, action_dim=2)
    obs = torch.randn(2, 3, 4, requires_grad=False)
    logits, _, _ = pipe(obs)
    loss = logits.pow(2).mean()
    loss.backward()

    msg_grads = [p.grad for p in pipe.msg_head.parameters()]
    assert all(g is not None for g in msg_grads)
    assert any((g.abs().sum() > 0).item() for g in msg_grads)
