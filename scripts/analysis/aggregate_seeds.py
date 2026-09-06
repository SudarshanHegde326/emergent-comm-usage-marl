"""
MSc AI Dissertation — Master Data Aggregation Engine (REPAIRED & GNN-READY)
File: scripts/aggregate_seeds.py
Author: Sudarshan Hegde (ID 25866326)

Description:
    Aggregate evaluation metrics across N seed checkpoints into one JSON file.
    Supports dynamic fallback for missing arguments, cross-architecture loading, 
    and multi-agent rollout pipelines (No-Comm, FC, Attention, GNN).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, List
import numpy as np
import torch
import torch.nn as nn

# Path configuration to resolve the core source directory layers
THIS_FILE = Path(__file__).resolve()
candidate_roots = [THIS_FILE.parent.parent.parent, THIS_FILE.parent.parent]
REPO_ROOT = None
for root in candidate_roots:
    if (root / "src" / "comms" / "attention_comm.py").exists():
        REPO_ROOT = root
        break
    if (root / "repo" / "src" / "comms" / "attention_comm.py").exists():
        REPO_ROOT = root / "repo"
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root.")
sys.path.insert(0, str(REPO_ROOT))

# Structural Layer Neural Architecture Definition for Decentralized Baselines
class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# Safe dictionary-namespace structural wrapper object
class SafeArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

# default env config (overridden by CLI args in main); matches harder training env
ENV_CFG = {"n_agents": 6, "agent_neighbors": 1, "landmark_neighbors": 1, "max_cycles": 50, "num_landmarks": 4, "num_obstacles": 2}

def make_env(seed: int, n_agents: int = 6, agent_neighbors: int = 2,
             landmark_neighbors: int = 2, max_cycles: int = 50,
             num_landmarks: int = 4, num_obstacles: int = 0):
    """Instantiate the target parallel pettingzoo evaluation environment.

    Config comes from ENV_CFG, which main() fills from REQUIRED CLI flags.
    """
    from src.envs.simple_spread_planC import planC_parallel_env
    _an = None if agent_neighbors < 0 else agent_neighbors
    _ln = None if landmark_neighbors < 0 else landmark_neighbors
    env = planC_parallel_env(
        N=n_agents, num_landmarks=num_landmarks, num_obstacles=num_obstacles,
        local_ratio=0.5,
        num_agent_neighbors=_an, num_landmark_neighbors=_ln,
        max_cycles=max_cycles, continuous_actions=False,
    )
    env.reset(seed=seed)
    return env

def load_pipeline(checkpoint_path, device):
    """Loads a saved checkpoint and returns the instantiated model pipeline."""
    import importlib.util
    from pathlib import Path
    import argparse
    
    # Allowlist argparse Namespace for secure loading of local metadata objects
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Safe fallback initialization using our SafeArgs hybrid structure
    if "args" in ckpt:
        train_args = ckpt["args"]
        if not hasattr(train_args, "get"):
            train_args = SafeArgs(**vars(train_args))
    else:
        train_args = SafeArgs(hidden_dim=64, msg_dim=8, num_heads=2)

    # Read the REAL observation/action dimensions from a probe env built with
    # the same difficulty config as training (so harder-env checkpoints load
    # correctly). Falls back to 18/5 only if the probe fails.
    try:
        _probe = make_env(0, n_agents=ENV_CFG["n_agents"],
                          agent_neighbors=ENV_CFG["agent_neighbors"],
                          landmark_neighbors=ENV_CFG["landmark_neighbors"],
                          max_cycles=ENV_CFG["max_cycles"],
                          num_landmarks=ENV_CFG["num_landmarks"],
                          num_obstacles=ENV_CFG["num_obstacles"])
        _a0 = list(_probe.agents)[0]
        obs_dim = _probe.observation_space(_a0).shape[0]
        action_dim = _probe.action_space(_a0).n
        _probe.close()
    except Exception:
        obs_dim = 18      # Simple Spread easy-default fallback
        action_dim = 5
    
    msg_dim = train_args.get("msg_dim", 8)
    num_heads = train_args.get("num_heads", 2)
    hidden = train_args.get("hidden", 128)

    # Robustly isolate state dictionaries across historical and stable frameworks
    if "pipeline" in ckpt:
        state_dict = ckpt["pipeline"]
    elif "actor_state_dicts" in ckpt:
        state_dict = ckpt["actor_state_dicts"]
    elif "actor_state_dict" in ckpt:
        state_dict = ckpt["actor_state_dict"]
    else:
        state_dict = ckpt

    keys = state_dict.keys() if hasattr(state_dict, "keys") else []
    
    # Structural condition discovery
    is_ib = any("channel.mu_head" in str(k) for k in keys)
    is_bandwidth = any("channel.down" in str(k) or "channel.up" in str(k) for k in keys)
    is_noise = float(train_args.get("sigma", 0.0)) > 0.0
    
    # 1. Pure Decentralized No-Comm
    if "agent_0" in keys or "actor_state_dicts" in ckpt or "nocomm" in str(checkpoint_path):
        pipe = {f"agent_{i}": MLPNetwork(obs_dim, action_dim).to(device) for i in range(3)}
        if "agent_0" in keys:
            for k, sd in state_dict.items():
                pipe[k].load_state_dict(sd)
        elif isinstance(state_dict, dict) and any(str(k).startswith("network") for k in keys):
            for i in range(3):
                pipe[f"agent_{i}"].load_state_dict(state_dict)
        elif "actor_state_dicts" in ckpt:
            for i, agent_name in enumerate(["agent_0", "agent_1", "agent_2"]):
                pipe[agent_name].load_state_dict(ckpt["actor_state_dicts"][agent_name])
        condition = "no-comm"
        return pipe, train_args, condition, obs_dim

    # 2. Information Bottleneck Attention
    elif is_ib:
        script_path = Path("scripts/05_attncomm_ib_mappo.py").resolve()
        spec = importlib.util.spec_from_file_location("ib_trainer", script_path)
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        pipe = trainer_module.IBAttentionPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_heads=num_heads, hidden=hidden).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "attn-ib-b0p01"
        
    # 3. Bandwidth Attention
    elif is_bandwidth:
        script_path = Path("scripts/04_attncomm_bandwidth_mappo.py").resolve()
        spec = importlib.util.spec_from_file_location("bandwidth_trainer", script_path)
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        channel_type = train_args.get("channel_type", "low-rank")
        bandwidth = train_args.get("bandwidth", 8)
        pipe = trainer_module.BandwidthAttentionPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_heads=num_heads, hidden=hidden, channel_type=channel_type, bandwidth=bandwidth).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "attn-bw8"
        
    # 4. Noise Attention
    elif is_noise:
        script_path = Path("scripts/06_attncomm_noise_mappo.py").resolve()
        spec = importlib.util.spec_from_file_location("noise_trainer", script_path)
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        sigma = float(train_args.get("sigma", 0.0))
        pipe = trainer_module.NoiseAttentionPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_heads=num_heads, hidden=hidden, sigma=sigma).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = f"attn-noise-{trainer_module.sigma_tag(sigma)}"

    # 5. GNN Communication (Must precede FC check)
    elif "gnncomm" in str(checkpoint_path) or any("agg.layer_norms" in str(k) for k in keys):
        from src.comms.gnn_comm import GNNCommPipeline
        num_layers = int(train_args.get("num_layers", 2))
        pipe = GNNCommPipeline(
            obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, 
            num_layers=num_layers, hidden=hidden, channel=None
        ).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "gnn-comm"

    # 6. Attention Communication
    elif "attncomm" in str(checkpoint_path) or any("agg.W_q" in str(k) for k in keys):
        from src.comms.attention_comm import AttentionCommPipeline
        pipe = AttentionCommPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_heads=num_heads, hidden=hidden).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "attn-comm"

    # 7. Fully-Connected Communication Baseline
    elif "fccomm" in str(checkpoint_path) or any("msg_head" in str(k) for k in keys):
        from src.comms.fc_comm import FCCommPipeline
        pipe = FCCommPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, hidden=hidden).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "fc-comm"

    # Default fallback to Attention
    else:
        from src.comms.attention_comm import AttentionCommPipeline
        pipe = AttentionCommPipeline(obs_dim=obs_dim, msg_dim=msg_dim, action_dim=action_dim, num_heads=num_heads, hidden=hidden).to(device)
        pipe.load_state_dict(state_dict)
        pipe.eval()
        condition = "attn-comm"
        
    return pipe, train_args, condition, obs_dim

def rollout(pipe_or_actor, condition: str, n_episodes: int, seed_offset: int, device: torch.device) -> List[float]:
    """Execute purely greedy evaluation episodes and return their total rewards."""
    env = make_env(seed_offset, n_agents=ENV_CFG["n_agents"],
                   agent_neighbors=ENV_CFG["agent_neighbors"],
                   landmark_neighbors=ENV_CFG["landmark_neighbors"],
                   max_cycles=ENV_CFG["max_cycles"],
                   num_landmarks=ENV_CFG["num_landmarks"],
                   num_obstacles=ENV_CFG["num_obstacles"])
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        agents = list(env.agents)
        ep_ret = 0.0
        done = False
        while not done:
            obs_arr = np.stack([obs[a] for a in agents], axis=0)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                if condition == "no-comm" or isinstance(pipe_or_actor, dict):
                    if isinstance(pipe_or_actor, dict):
                        actions_list = []
                        for i, a in enumerate(agents):
                            logits = pipe_or_actor[f"agent_{i}"](obs_t[i].unsqueeze(0))
                            actions_list.append(logits.argmax(dim=-1).item())
                        action_dict = {a: actions_list[i] for i, a in enumerate(agents)}
                    else:
                        logits = pipe_or_actor(obs_t)
                        actions = logits.argmax(dim=-1)
                        action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
                else:
                    if condition == "fc-comm":
                        logits, _, _ = pipe_or_actor(obs_t.unsqueeze(0))
                        logits = logits.squeeze(0)
                    else:
                        # GNN, Attention, etc., return a tuple where idx 0 is logits
                        out = pipe_or_actor(obs_t.unsqueeze(0))
                        logits = out[0].squeeze(0)
                    actions = logits.argmax(dim=-1)
                    action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}

            obs, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            ep_ret += float(np.mean(list(reward_dict.values())))
            done = any(term_dict.values()) or any(trunc_dict.values())
        returns.append(ep_ret)
    env.close()
    return returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, type=str, help="Condition label.")
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path, help="Paths to checkpoints.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=12345)
    parser.add_argument("--out", required=True, type=Path)
    # Env difficulty — MUST match the env the checkpoints were trained on
    # [FIX] NO DEFAULTS. State the world you are evaluating in, out loud.
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--agent-neighbors", type=int, required=True, help="-1 = see all")
    parser.add_argument("--landmark-neighbors", type=int, required=True, help="-1 = see all")
    parser.add_argument("--num-landmarks", type=int, required=True)
    parser.add_argument("--num-obstacles", type=int, required=True)
    parser.add_argument("--max-cycles", type=int, default=50)   # same in every plan
    args = parser.parse_args()

    # publish env config so load_pipeline / rollout build the matching env
    global ENV_CFG
    ENV_CFG = {
        "n_agents": args.n_agents,
        "agent_neighbors": args.agent_neighbors,
        "landmark_neighbors": args.landmark_neighbors,
        "max_cycles": args.max_cycles,
        "num_landmarks": args.num_landmarks,
        "num_obstacles": args.num_obstacles,
    }

    # [FIX] shout the config + verify obs_dim before trusting a single number
    _probe_env = make_env(0, **ENV_CFG)
    _obs_dim = _probe_env.observation_space(_probe_env.agents[0]).shape[0]
    _probe_env.close()

    _EXPECTED = {16: "Plan A", 10: "Plan B", 20: "Plan C", 26: "Plan D"}
    _which = _EXPECTED.get(_obs_dim, "*** MATCHES NO PLAN - CHECK YOUR FLAGS ***")
    print(f"[aggregate/env] {ENV_CFG}", flush=True)
    print(f"[aggregate/env] obs_dim={_obs_dim} -> {_which}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aggregate] device={device} condition={args.condition} n_checkpoints={len(args.checkpoints)}")

    per_seed_returns: List[List[float]] = []
    per_seed_train_seed: List[int] = []
    detected_conditions: List[str] = []

    for ck in args.checkpoints:
        if not ck.exists():
            print(f"[aggregate] SKIP missing checkpoint: {ck}")
            continue
        try:
            pipe_or_actor, train_args, condition, obs_dim = load_pipeline(ck, device)
            detected_conditions.append(condition)
            per_seed_train_seed.append(int(train_args.get("seed", -1)))
            returns = rollout(pipe_or_actor, condition, args.eval_episodes, args.seed_offset, device)
            per_seed_returns.append(returns)
            print(f"[aggregate]   {ck.name}: mean={np.mean(returns):+.2f} std={np.std(returns):.2f}")
        except Exception as e:
            print(f"[aggregate] ERROR processing {ck.name}: {str(e)}")
            continue

    if not per_seed_returns:
        raise SystemExit("No checkpoints successfully processed.")

    per_seed_mean = [float(np.mean(r)) for r in per_seed_returns]
    per_seed_std = [float(np.std(r)) for r in per_seed_returns]
    all_flat = [float(x) for sub in per_seed_returns for x in sub]

    if len(per_seed_mean) >= 2:
        mean_across = float(np.mean(per_seed_mean))
        std_across = float(np.std(per_seed_mean, ddof=1))
    else:
        mean_across = float(per_seed_mean[0])
        std_across = 0.0

    payload: dict[str, Any] = {
        "condition": args.condition,
        "detected_conditions": detected_conditions,
        "num_seeds": len(per_seed_returns),
        "seeds": per_seed_train_seed,
        "per_seed_mean_return": per_seed_mean,
        "per_seed_std_return": per_seed_std,
        "all_returns_flat": all_flat,
        "mean_return_across_seeds": mean_across,
        "std_return_across_seeds": std_across,
        "eval_episodes_per_seed": int(args.eval_episodes),
        "checkpoints": [str(p) for p in args.checkpoints if p.exists()],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[aggregate] saved successfully -> {args.out}")

if __name__ == "__main__":
    main()