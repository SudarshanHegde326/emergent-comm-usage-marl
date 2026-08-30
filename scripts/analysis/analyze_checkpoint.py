"""Offline analysis of a trained SMACv2 checkpoint.

Loads a checkpoint saved by 14_mappo_smacv2_matched.py, reconstructs the policy
from the config stored inside it, then computes the four ToR evaluations and
writes ONE tidy CSV row:

    test_win_rate,
    mine_I_M_obs, mine_I_M_enemy, infonce_I_M_enemy,     (mutual information, nats)
    hoyer_sparsity, activation_sparsity, gate_open_rate,  (sparsity)
    abl_zero_winrate_drop, abl_shuffle_winrate_drop,      (message ablation)
    robust_noise_<s>_retained, robust_drop_<p>_retained   (robustness sweep, % kept)

It needs the real StarCraft II env (it runs greedy rollouts), so run it on the
HPC AFTER training. Everything is computed on ONE set of rollouts (collect once,
analyse many).

Usage
-----
    python analyze_checkpoint.py \
        --trainer scripts/trainers/14_mappo_smacv2_matched.py \
        --checkpoint experiments/smacv2_epo_attn/seed0/final.pt \
        --episodes 64 --out results/epo_attn_seed0.csv

MINE/sparsity require messages, so the no-comm baseline reports NaN for those
(correctly — it emits nothing) but still reports win rate and ablation (a no-op),
so the row schema stays uniform across conditions.
"""

from __future__ import annotations
import argparse, csv, importlib.util, os, sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.analysis.mine import estimate_mine, estimate_infonce
from src.analysis.metrics import hoyer_sparsity, activation_sparsity, gate_open_rate
from src.channel_constraints import quantize_ste


# --------------------------------------------------------------------------
# Load the trainer module by file path (its name starts with a digit and lives
# in a directory with spaces, so a normal import will not work).
# --------------------------------------------------------------------------
def load_trainer(path: str):
    spec = importlib.util.spec_from_file_location("smacv2_trainer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_policy(trainer, ckpt, env, device):
    """Reconstruct the RecurrentPolicy from the config saved in the checkpoint."""
    cfg = ckpt.get("config") or ckpt.get("args")
    info = env.get_env_info()
    N, A = info["n_agents"], info["n_actions"]
    obs_dim = info["obs_shape"] + N
    policy = trainer.RecurrentPolicy(
        obs_dim, A, cfg["hidden"], cfg["comm"], cfg["msg_dim"],
        cfg["num_heads"], cfg["num_layers"],
        comm_to_memory=cfg["comm_to_memory"],
        mlp_encoder=cfg["mlp_encoder"], gate=cfg["gate"],
        aux_sighting=False,                 # aux head not needed for analysis
        pool_type=cfg.get("pool_type", "mean"),
    ).to(device)
    # Non-strict: the trained model may carry an aux_head this analysis policy lacks.
    policy.load_state_dict(ckpt["policy"], strict=False)
    obs_norm = trainer.RunningNorm(obs_dim, device, active_dim=info["obs_shape"])
    obs_norm.load_state_dict(ckpt["obs_norm"])
    policy.eval()
    return policy, obs_norm, N, info["episode_limit"]


def enemy_slice(env):
    """Same (lo, hi) enemy-feature columns the trainer's aux label uses."""
    raw = getattr(env, "env", env)
    try:
        lo = int(raw.get_obs_move_feats_size())
        en = raw.get_obs_enemy_feats_size()
        hi = lo + (en if isinstance(en, int) else int(np.prod(en)))
        return lo, hi
    except Exception:
        return None


# --------------------------------------------------------------------------
# Collect ONE set of greedy rollouts, recording per living-agent
# (obs, message, gate, enemy_visible). Mirrors evaluate()'s rollout exactly.
# --------------------------------------------------------------------------
@torch.no_grad()
def collect(policy, obs_norm, env, N, ep_limit, agent_ids, eslice, n_episodes, device):
    obs_rows, msg_rows, gate_rows, enemy_rows = [], [], [], []
    for _ in range(n_episodes):
        env.reset()
        h = policy.init_hidden(1, N, device).squeeze(0)
        done, steps = False, 0
        while not done:
            raw = np.asarray(env.get_obs(), dtype=np.float32)          # (N, raw)
            o = torch.as_tensor(raw, device=device)
            o = torch.cat([o, agent_ids], dim=-1)
            av = torch.as_tensor(np.asarray(env.get_avail_actions(), dtype=np.float32),
                                 device=device)
            alive = (av[:, 0] < 0.5)
            logits, h = policy(obs_norm(o).unsqueeze(0), h.unsqueeze(0),
                               alive.float().unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(av < 0.5, -1e9)
            act = logits.argmax(dim=-1)
            h = h.squeeze(0)

            live = alive.cpu().numpy().astype(bool)
            if live.any():
                obs_rows.append(o[live].cpu().numpy())
                if eslice is not None:
                    lo, hi = eslice
                    ev = (np.abs(raw[:, lo:hi]).sum(1) > 1e-6).astype(np.float32)
                    enemy_rows.append(ev[live])
                msg = policy.comm.last_message
                if msg is not None:
                    msg_rows.append(msg.squeeze(0)[live].cpu().numpy())
                g = policy.comm.last_gate
                if g is not None:
                    gate_rows.append(g.squeeze(0)[live].cpu().numpy())

            reward, terminated, info = env.step(act.tolist())
            steps += 1
            done = bool(terminated) or steps >= ep_limit

    def stack(rows):
        return torch.as_tensor(np.concatenate(rows, 0)) if rows else None
    return {
        "obs": stack(obs_rows),
        "message": stack(msg_rows),
        "gate": stack(gate_rows),
        "enemy_visible": stack(enemy_rows),
    }


# --------------------------------------------------------------------------
# Message interventions for ablation + robustness, via a forward hook on the
# message encoder (works on any checkpoint, no trainer edits required).
# --------------------------------------------------------------------------
def _encoder(policy):
    return getattr(policy.comm, "encoder", None)   # None for NoComm


def _hook_zero(mod, inp, out):
    return torch.zeros_like(out)


def _hook_shuffle(mod, inp, out):
    # permute messages across the agent dimension (dim=1 of (B, N, msg))
    perm = torch.randperm(out.shape[1], device=out.device)
    return out[:, perm, :]


def _hook_noise(std):
    def h(mod, inp, out):
        return out + std * torch.randn_like(out)
    return h


def _hook_drop(p):
    def h(mod, inp, out):
        keep = (torch.rand(out.shape[:-1], device=out.device) >= p)
        return out * keep.unsqueeze(-1).to(out.dtype)
    return h


def _hook_quant(bits):
    def h(mod, inp, out):
        return quantize_ste(out, bits, value_range=2.0)
    return h


def eval_with_hook(trainer, policy, obs_norm, env, agent_ids, n_episodes,
                   ep_limit, device, hook=None):
    enc = _encoder(policy)
    handle = enc.register_forward_hook(hook) if (hook and enc is not None) else None
    try:
        wr, ret = trainer.evaluate(env, policy, obs_norm, agent_ids,
                                   n_episodes, ep_limit, device)
    finally:
        if handle is not None:
            handle.remove()
    return wr, ret


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True, help="path to 14_mappo_smacv2_matched.py")
    ap.add_argument("--checkpoint", required=True, help="final.pt / latest.pt")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--mine-iters", type=int, default=3000)
    ap.add_argument("--noise-sweep", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    ap.add_argument("--drop-sweep", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    ap.add_argument("--quant-sweep", type=int, nargs="*", default=[4, 2, 1],
                    help="bits per message dim to test as a bandwidth perturbation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    trainer = load_trainer(args.trainer)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or ckpt.get("args")
    if cfg is None:
        raise SystemExit("Checkpoint has neither 'config' nor 'args' -- is this a "
                         "14_mappo_smacv2_matched.py checkpoint?")
    if "obs_norm" not in ckpt:
        raise SystemExit(
            "This checkpoint has no 'obs_norm' (the observation normaliser).\n"
            "final.pt does NOT store it, so evaluation would use unnormalised\n"
            "inputs and report meaningless win rates. Point --checkpoint at\n"
            "latest.pt instead -- save_checkpoint() writes it with obs_norm,\n"
            "state_norm and config, and the final step is saved there too.")

    env = trainer.make_env(cfg["mode"], cfg["seed"], cfg.get("n_units"),
                           cfg.get("n_enemies", 5), cfg.get("map_name", "10gen_protoss"))
    policy, obs_norm, N, ep_limit = rebuild_policy(trainer, ckpt, env, device)
    agent_ids = torch.eye(N, device=device)
    eslice = enemy_slice(env)

    row = {
        "map": cfg.get("map_name"), "mode": cfg.get("mode"),
        "comm": cfg.get("comm"), "seed": cfg.get("seed"),
        "aux_on": float(cfg.get("aux_coef", 0.0) > 0.0),
        "steps": int(ckpt.get("steps", -1)),
    }

    # ---- intact win rate (baseline for ablation + robustness) ----
    wr0, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                            args.episodes, ep_limit, device, hook=None)
    row["test_win_rate"] = wr0

    # ---- collect once, then MINE + sparsity ----
    data = collect(policy, obs_norm, env, N, ep_limit, agent_ids, eslice,
                   args.episodes, device)
    msg = data["message"]
    if msg is not None and msg.shape[0] >= 64:
        row["mine_I_M_obs"] = estimate_mine(msg, data["obs"], iters=args.mine_iters, device=str(device))
        if data["enemy_visible"] is not None:
            row["mine_I_M_enemy"] = estimate_mine(msg, data["enemy_visible"], iters=args.mine_iters, device=str(device))
            row["infonce_I_M_enemy"] = estimate_infonce(msg, data["enemy_visible"], iters=args.mine_iters, device=str(device))
        row["hoyer_sparsity"] = hoyer_sparsity(msg)
        row["activation_sparsity"] = activation_sparsity(msg)
    else:
        for k in ("mine_I_M_obs", "mine_I_M_enemy", "infonce_I_M_enemy",
                  "hoyer_sparsity", "activation_sparsity"):
            row[k] = float("nan")
    row["gate_open_rate"] = gate_open_rate(data["gate"]) if data["gate"] is not None else float("nan")

    # ---- message ablation (zero + shuffle) ----
    if _encoder(policy) is not None:
        wr_zero, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                                    args.episodes, ep_limit, device, hook=_hook_zero)
        wr_shuf, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                                    args.episodes, ep_limit, device, hook=_hook_shuffle)
        row["abl_zero_winrate_drop"] = wr0 - wr_zero
        row["abl_shuffle_winrate_drop"] = wr0 - wr_shuf
    else:
        row["abl_zero_winrate_drop"] = float("nan")
        row["abl_shuffle_winrate_drop"] = float("nan")

    # ---- robustness sweeps (% of clean win rate retained) ----
    denom = wr0 if wr0 > 1e-6 else float("nan")
    for s in args.noise_sweep:
        wr, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                               args.episodes, ep_limit, device, hook=_hook_noise(s))
        row[f"robust_noise_{s}_retained"] = wr / denom
    for p in args.drop_sweep:
        wr, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                               args.episodes, ep_limit, device, hook=_hook_drop(p))
        row[f"robust_drop_{p}_retained"] = wr / denom
    for b in args.quant_sweep:
        wr, _ = eval_with_hook(trainer, policy, obs_norm, env, agent_ids,
                               args.episodes, ep_limit, device, hook=_hook_quant(b))
        row[f"robust_quant_{b}bit_retained"] = wr / denom

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("wrote", args.out)
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
