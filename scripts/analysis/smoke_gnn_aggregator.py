from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.comms.gnn_comm import GNNAggregator, GNNCommPipeline
from src.comms.channels import IdentityChannel


def main() -> None:
    torch.manual_seed(0)
    B, N, obs_dim, msg_dim, action_dim, L = 2, 3, 18, 8, 5, 2

    print(f"[smoke] B={B} N={N} obs_dim={obs_dim} msg_dim={msg_dim} action_dim={action_dim} L={L}")

    pipe = GNNCommPipeline(
        obs_dim=obs_dim,
        msg_dim=msg_dim,
        action_dim=action_dim,
        num_layers=L,
        hidden=64,
    )

    obs = torch.randn(B, N, obs_dim)
    logits, msgs, msgs_post, agg, per_layer = pipe(obs)

    print(f"[smoke] shapes — logits {tuple(logits.shape)}  msgs {tuple(msgs.shape)}  agg {tuple(agg.shape)}")

    msg_l2 = msgs.norm(dim=-1).mean().item()
    agg_l2 = agg.norm(dim=-1).mean().item()
    print(f"[smoke] msgs |.|_l2 mean      ~ {msg_l2:.3f}")
    print(f"[smoke] agg  |.|_l2 mean      ~ {agg_l2:.3f}")

    traj = [h.norm(dim=-1).mean().item() for h in per_layer]
    traj_fmt = ", ".join(f"{x:.3f}" for x in traj)
    print(f"[smoke] per-layer L2 trajectory: [{traj_fmt}]")

    # D8 sanity check — feed the EXACT same obs tensor to verify behavior matching
    torch.manual_seed(123)
    pipe_default = GNNCommPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_layers=L, hidden=64, channel=None)
    torch.manual_seed(123)
    pipe_explicit = GNNCommPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_layers=L, hidden=64, channel=IdentityChannel(msg_dim=msg_dim))

    test_obs = torch.randn(B, N, obs_dim)
    out_d = pipe_default(test_obs)[0]
    out_e = pipe_explicit(test_obs)[0]
    match = bool(torch.equal(out_d, out_e))
    print(f"[smoke] D8 sanity (Identity == default) — match: {match}")

    if match:
        print("[smoke] OK")
    else:
        print("[smoke] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
