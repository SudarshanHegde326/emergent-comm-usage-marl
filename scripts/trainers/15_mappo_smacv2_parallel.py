
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# --- import the PROVEN single-env components by real module name -------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import vec_smacv2 as vec                              

_S = _HERE / "14_mappo_smacv2_matched.py"
_spec = importlib.util.spec_from_file_location("smacv2_single", _S)
S = importlib.util.module_from_spec(_spec)
sys.modules["smacv2_single"] = S
_spec.loader.exec_module(S)

# Reused verbatim from the proven trainer:
RecurrentPolicy = S.RecurrentPolicy
RecurrentCritic = S.RecurrentCritic
RunningNorm = S.RunningNorm
compute_gae = S.compute_gae
make_env = S.make_env
NoComm = S.NoComm


# =============================================================================
# Per-env GAE. The single-env compute_gae handles a 1-D (T,) sequence; here we
# have (T, B) and must run the recursion INDEPENDENTLY per env so that env b's
# bootstrap never crosses env b's episode boundary OR bleeds into env b'.
# =============================================================================
def compute_gae_batched(rewards, values, dones, truncs, last_values, gamma, lam):
    T, B = rewards.shape
    adv = torch.zeros(T, B, device=rewards.device)
    running = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        true_terminal = dones[t] * (1.0 - truncs[t])   # 1 only for real end
        nonterm = 1.0 - true_terminal
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        running = delta + gamma * lam * nonterm * running
        adv[t] = running
    return adv, adv + values


def build_args(argv=None):
    p = S.build_parser()
    p.add_argument("--num-envs", type=int, default=8,
                   help="number of parallel StarCraft II environments")
    p.add_argument("--amp", action="store_true",
                   help="use automatic mixed precision in the PPO update "
                        "(little benefit here: the bottleneck is the CPU/SC2, "
                        "not the GPU)")
    args = p.parse_args(argv)
    if args.comm == "gat":
        args.comm = "gnn"
    if args.chunk_len < 2:
        raise ValueError("--chunk-len must be >= 2 for BPTT to mean anything")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.rollout_len % args.chunk_len != 0:
        raise ValueError(
            f"--rollout-len ({args.rollout_len}) must be a multiple of "
            f"--chunk-len ({args.chunk_len}).")
    return args


def train_parallel(args, env_factory_override=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = args.num_envs

    # --- build the vectorized env (B workers, distinct seeds) ---------------
    if env_factory_override is not None:
        factory = env_factory_override
    else:
        mode, seed = args.mode, args.seed

        def factory(rank):
            return lambda: make_env(mode, seed * 1000 + rank,
                                    args.n_units, args.n_enemies, args.map_name)

    _probe = vec.make_vec(factory, 1)
    _info = _probe.get_env_info()
    _ep_limit = _info["episode_limit"]
    _probe.close()
    venv = vec.make_vec(factory, B, ep_limit=_ep_limit)

    info = venv.get_env_info()
    N = info["n_agents"]
    A = info["n_actions"]
    raw_obs_dim = info["obs_shape"]
    state_dim = info["state_shape"]
    ep_limit = info["episode_limit"]
    obs_dim = raw_obs_dim + N
    agent_ids = torch.eye(N, device=device)          # (N,N)

    enemy_slice = None
    if args.aux_coef > 0.0 and args.comm != "none":
        layout = venv.get_obs_layout()
        if layout is not None:
            move_dim, enemy_dim = layout
            enemy_slice = (int(move_dim), int(move_dim) + int(enemy_dim))
            print(f"[par/{args.comm}] aux sighting: enemy feats at raw cols "
                  f"[{enemy_slice[0]}:{enemy_slice[1]}]", flush=True)
        else:
            print(f"[par/{args.comm}] WARNING: enemy feats not locatable; "
                  f"aux sighting DISABLED.", flush=True)

    print(f"[par/{args.comm}] mode={args.mode} envs={B} N={N} actions={A} "
          f"raw_obs={raw_obs_dim} obs+id={obs_dim} state={state_dim} "
          f"ep_limit={ep_limit} device={device} amp={args.amp}", flush=True)

    aux_on = enemy_slice is not None
    channel_cfg = {
        "quant_bits": args.quant_bits,
        "noise_std": args.noise_std,
        "dropout_p": args.agent_dropout,
        "kl_bottleneck": args.kl_coef > 0.0,
    }
    policy = RecurrentPolicy(obs_dim, A, args.hidden, args.comm, args.msg_dim,
                             args.num_heads, args.num_layers,
                             comm_to_memory=args.comm_to_memory,
                             mlp_encoder=args.mlp_encoder, gate=args.gate,
                             aux_sighting=aux_on,
                             pool_type=args.pool_type,
                             channel_cfg=channel_cfg).to(device)
    critic = RecurrentCritic(state_dim, args.hidden).to(device)
    opt_a = torch.optim.Adam(policy.parameters(), lr=args.lr_actor, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)

    obs_norm = RunningNorm(obs_dim, device, active_dim=raw_obs_dim)
    state_norm = RunningNorm(state_dim, device)
    safe_config = {k: v for k, v in vars(args).items()
                   if not k.startswith("_")
                   and isinstance(v, (int, float, str, bool, type(None), list, tuple))}

    use_amp = args.amp and device.type == "cuda"
    class _NullScaler:
        def scale(self, loss): return loss
        def unscale_(self, opt): pass
        def step(self, opt): opt.step()
        def update(self): pass
        def is_enabled(self): return False
    if not use_amp:
        scaler = _NullScaler()
    else:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="msc-marl-comm",
                         name=f"par_{args.mode}_{args.comm}_e{B}_seed{args.seed}",
                         config=vars(args), tags=["parallel", args.mode, args.comm])

    T = args.rollout_len
    L = args.chunk_len
    n_chunks = T // L

    # --- rollout buffer, now with an env-batch dimension B ------------------
    def zeros(*shape):
        return torch.zeros(*shape, device=device)

    buf = {
        "obs": zeros(T, B, N, obs_dim),
        "state": zeros(T, B, state_dim),
        "avail": zeros(T, B, N, A),
        "alive": zeros(T, B, N),
        "act": torch.zeros(T, B, N, dtype=torch.long, device=device),
        "logp": zeros(T, B, N),
        "val": zeros(T, B),
        "rew": zeros(T, B),
        "done": zeros(T, B),
        "h_pol": zeros(T, B, N, args.hidden),
        "h_cri": zeros(T, B, args.hidden),
        "sight": zeros(T, B, N),
        "trunc": zeros(T, B),
    }

    # ---- checkpointing (same format/keys as the single-env trainer) --------
    out = Path(args.out_dir or
               f"experiments/par_smacv2_{args.mode}_{args.comm}") / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "latest.pt"

    def save_checkpoint(step_count, final=False):
        payload = {
            "steps": step_count,
            "policy": policy.state_dict(), "critic": critic.state_dict(),
            "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
            "obs_norm": obs_norm.state_dict(), "state_norm": state_norm.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "wins_log": wins_log[-400:], "returns_log": returns_log[-400:],
            "lens_log": lens_log[-400:], "next_eval": next_eval,
            "config": safe_config,
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(ckpt_path)
        print(f"[par/{args.comm}] {'FINAL' if final else 'checkpoint'} saved at "
              f"step {step_count:,} -> {ckpt_path}", flush=True)

    def read_batch(obs_np, state_np, avail_np):
        o = torch.as_tensor(obs_np, device=device)                    # (B,N,raw)
        if enemy_slice is not None:
            lo, hi = enemy_slice
            sight = torch.as_tensor(
                (np.abs(obs_np[:, :, lo:hi]).sum(axis=2) > 1e-6).astype(np.float32),
                device=device)                                        # (B,N)
        else:
            sight = torch.zeros(o.shape[0], o.shape[1], device=device)
        ids = agent_ids.unsqueeze(0).expand(o.shape[0], -1, -1)       # (B,N,N)
        o = torch.cat([o, ids], dim=-1)                               # (B,N,obs_dim)
        s = torch.as_tensor(state_np, device=device)
        av = torch.as_tensor(avail_np, device=device)
        alive = (av[:, :, 0] < 0.5).float()                           # (B,N)
        return o, s, av, alive, sight

    # ---- training bookkeeping ----
    steps = 0                       # counts ENV-STEPS-PER-ENV (so one rollout of length T advances `steps` by T, while gathering T*B transitions)
    total_env_steps = 0             # true total transitions, for logging/fps
    returns_log, wins_log, lens_log = [], [], []
    ep_ret = np.zeros(B, dtype=np.float64)
    ep_len = np.zeros(B, dtype=np.int64)
    next_eval = args.eval_interval if args.eval_episodes > 0 else float("inf")
    test_wr, test_ret = float("nan"), float("nan")
    wr = rr = ll = float("nan")
    policy_loss = value_loss = ent_term = aux_loss = torch.zeros((), device=device)
    grad_updates = [0]

    # ---- resume ----
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"]); critic.load_state_dict(ck["critic"])
        opt_a.load_state_dict(ck["opt_a"]); opt_c.load_state_dict(ck["opt_c"])
        obs_norm.load_state_dict(ck["obs_norm"]); state_norm.load_state_dict(ck["state_norm"])
        torch.set_rng_state(ck["torch_rng"].cpu() if hasattr(ck["torch_rng"], "cpu")
                            else ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        if torch.cuda.is_available() and "cuda_rng" in ck:
            try: torch.cuda.set_rng_state_all(ck["cuda_rng"])
            except Exception: pass
        steps = int(ck["steps"])
        wins_log = list(ck.get("wins_log", [])); returns_log = list(ck.get("returns_log", []))
        lens_log = list(ck.get("lens_log", [])); next_eval = ck.get("next_eval", next_eval)
        print(f"[par/{args.comm}] RESUMED at env-step {steps:,} "
              f"(target {args.total_steps:,})", flush=True)
    elif args.resume:
        print(f"[par/{args.comm}] --resume set but no checkpoint; starting fresh.",
              flush=True)
    next_ckpt = ((((steps * B) // args.ckpt_interval) + 1) * args.ckpt_interval
                 if args.ckpt_interval > 0 else float("inf"))

    obs_np, state_np, avail_np = venv.reset()
    obs_t, state_t, avail_t, alive_t, sight_t = read_batch(obs_np, state_np, avail_np)
    h_pol = policy.init_hidden(B, N, device)          # (B,N,hidden)
    h_cri = critic.init_hidden(B, device)             # (B,hidden)
    t0 = time.time()
    steps_at_start = steps

    while steps * B < args.total_steps:
        frac = max(0.0, 1.0 - (steps * B) / args.total_steps)
        for g in opt_a.param_groups: g["lr"] = args.lr_actor * frac
        for g in opt_c.param_groups: g["lr"] = args.lr_critic * frac

        # ---------------------------- collect ----------------------------
        for i in range(T):
            buf["obs"][i] = obs_t; buf["state"][i] = state_t
            buf["avail"][i] = avail_t; buf["alive"][i] = alive_t
            buf["h_pol"][i] = h_pol; buf["h_cri"][i] = h_cri
            buf["sight"][i] = sight_t

            with torch.no_grad():
                logits, h_pol_next = policy(obs_norm(obs_t), h_pol, alive_t)
                logits = logits.masked_fill(avail_t < 0.5, -1e9)
                dist = Categorical(logits=logits)
                act = dist.sample()                                   # (B,N)
                logp = dist.log_prob(act)                             # (B,N)
                val, h_cri_next = critic(state_norm(state_t), h_cri)  # (B,), (B,hidden)

            actions_np = act.cpu().numpy()
            n_obs, n_state, n_avail, reward, done_env, infos = venv.step(actions_np)

            buf["act"][i] = act; buf["logp"][i] = logp
            buf["val"][i] = val
            buf["rew"][i] = torch.as_tensor(reward, device=device)
            buf["done"][i] = torch.as_tensor(done_env.astype(np.float32),
                                            device=device)
            trunc_flags = np.array([float(bool(info.get("truncated", False))) for info in infos],
                                dtype=np.float32)
            buf["trunc"][i] = torch.as_tensor(trunc_flags, device=device)

            ep_ret += reward; ep_len += 1
            steps += 1; total_env_steps += B

            for b in range(B):
                if done_env[b]:
                    returns_log.append(float(ep_ret[b]))
                    wins_log.append(int(infos[b].get("battle_won", False)))
                    lens_log.append(int(ep_len[b]))
                    ep_ret[b] = 0.0; ep_len[b] = 0
                    # reset that env's recurrent state for the next episode
                    h_pol_next[b] = 0.0
                    h_cri_next[b] = 0.0

            obs_t, state_t, avail_t, alive_t, sight_t = read_batch(n_obs, n_state, n_avail)
            h_pol, h_cri = h_pol_next, h_cri_next

        # EPO diagnostic: fraction of (env, agent, step) where agent sees an enemy
        sight_frac = float(buf["sight"].mean().item())
        if run is not None:
            run.log({"sight_frac": sight_frac, "env_steps": steps * B})
            
        # freeze normaliser stats for the update; refresh after
        with torch.no_grad():
            last_val, _ = critic(state_norm(state_t), h_cri)          # (B,)

        adv, ret = compute_gae_batched(buf["rew"], buf["val"], buf["done"],
                               buf["trunc"], last_val,
                               args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_n = obs_norm(buf["obs"])
        state_n = state_norm(buf["state"])

        # -------------------- update (chunked BPTT over B envs) ----------
        chunk_ids = np.arange(n_chunks)
        for _ in range(args.ppo_epochs):
            np.random.shuffle(chunk_ids)
            for c in chunk_ids:
                s0, s1 = int(c) * L, (int(c) + 1) * L
                hp = buf["h_pol"][s0]                                 # (B,N,hidden)
                hc = buf["h_cri"][s0]                                 # (B,hidden)
                if use_amp:
                    try:
                        _amp_ctx = torch.amp.autocast("cuda")
                    except (AttributeError, TypeError):
                        _amp_ctx = torch.cuda.amp.autocast()
                else:
                    _amp_ctx = __import__("contextlib").nullcontext()
                with _amp_ctx:
                    logits_seq, vals_seq, aux_seq = [], [], []
                    kl_seq = []
                    for t in range(s0, s1):
                        lg, hp = policy(obs_n[t], hp, buf["alive"][t])   # (B,N,A)
                        logits_seq.append(lg)
                        if policy.aux_sighting:
                            al = policy.aux_logits()                    # (B,N) or None
                            aux_seq.append(al)
                        lk = getattr(policy.comm, "last_kl", None)
                        if lk is not None:
                            kl_seq.append(lk)
                        v, hc = critic(state_n[t], hc)                  # (B,)
                        vals_seq.append(v)
                        done_t = buf["done"][t]                        # (B,)
                        if done_t.any():
                            m = (done_t > 0.5)
                            hp = hp.clone(); hc = hc.clone()
                            hp[m] = 0.0; hc[m] = 0.0

                    logits = torch.stack(logits_seq)                   # (L,B,N,A)
                    vals = torch.stack(vals_seq)                       # (L,B)
                    logits = logits.masked_fill(buf["avail"][s0:s1] < 0.5, -1e9)

                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(buf["act"][s0:s1])        # (L,B,N)
                    entropy = dist.entropy()                           # (L,B,N)

                    alive = buf["alive"][s0:s1]                        # (L,B,N)
                    denom = alive.sum().clamp(min=1.0)

                    ratio = (new_logp - buf["logp"][s0:s1]).exp()      # (L,B,N)
                    a = adv[s0:s1].unsqueeze(-1).expand(-1, -1, N)     # (L,B,N)
                    s_1 = ratio * a
                    s_2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                    policy_loss = -(torch.min(s_1, s_2) * alive).sum() / denom
                    ent_term = (entropy * alive).sum() / denom
                    value_loss = F.mse_loss(vals, ret[s0:s1])

                    aux_loss = torch.zeros((), device=device)
                    if policy.aux_sighting and aux_seq and all(a is not None for a in aux_seq):
                        aux_logits = torch.stack(aux_seq)              # (L,B,N)
                        aux_target = buf["sight"][s0:s1]               # (L,B,N)
                        per = F.binary_cross_entropy_with_logits(
                            aux_logits, aux_target, reduction="none")
                        aux_loss = (per * alive).sum() / denom

                    kl_term = (torch.stack(kl_seq).mean()
                               if kl_seq else torch.zeros((), device=device))
                    loss = (policy_loss + args.vf_coef * value_loss
                            - args.ent_coef * ent_term + args.aux_coef * aux_loss
                            + args.kl_coef * kl_term)

                opt_a.zero_grad(set_to_none=True)
                opt_c.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt_a); scaler.unscale_(opt_c)
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)

                if args.grad_log_interval > 0 and not isinstance(policy.comm, NoComm):
                    grad_updates[0] += 1
                    if grad_updates[0] % args.grad_log_interval == 0:
                        gs = [p.grad.abs().mean().item()
                              for p in policy.comm.parameters() if p.grad is not None]
                        gmean = float(np.mean(gs)) if gs else 0.0
                        print(f"    [grad] comm mean|grad|={gmean:.3e} "
                              f"{'OK' if gmean>0 else 'ZERO-GRAD!'}", flush=True)

                scaler.step(opt_a); scaler.step(opt_c)
                scaler.update()

        obs_norm.update(buf["obs"]); state_norm.update(buf["state"])

        # ------------------------- checkpoint ----------------------------
        if args.ckpt_interval > 0 and steps * B >= next_ckpt:
            save_checkpoint(steps); next_ckpt += args.ckpt_interval

        # ------------------------ greedy eval ----------------------------
        if steps * B >= next_eval:
            test_wr, test_ret = evaluate_parallel(venv, policy, obs_norm,
                                                  agent_ids, args.eval_episodes,
                                                  ep_limit, device, N)
            next_eval += args.eval_interval
            obs_np, state_np, avail_np = venv.reset()
            obs_t, state_t, avail_t, alive_t, sight_t = read_batch(obs_np, state_np, avail_np)
            h_pol = policy.init_hidden(B, N, device)
            h_cri = critic.init_hidden(B, device)
            ep_ret[:] = 0.0; ep_len[:] = 0

        # ------------------------------ log ------------------------------
        wr = float(np.mean(wins_log[-40:])) if wins_log else float("nan")
        rr = float(np.mean(returns_log[-40:])) if returns_log else float("nan")
        ll = float(np.mean(lens_log[-40:])) if lens_log else float("nan")
        fps = int((total_env_steps) / (time.time() - t0 + 1e-9))
        print(f"env_steps={steps*B:>9,} (x{B})  train_wr={wr:.3f}  "
              f"test_wr={test_wr:.3f}  ep_ret={rr:6.2f}  ep_len={ll:5.1f}  "
              f"fps={fps}", flush=True)
        if run is not None:
            run.log({"env_steps": steps * B, "train_win_rate_last40": wr,
                     "test_win_rate": test_wr, "test_return": test_ret,
                     "ep_return_last40": rr, "ep_length_last40": ll,
                     "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                     "entropy": ent_term.item(), "aux_loss": float(aux_loss),
                     "fps": fps})

    if args.eval_episodes > 0:
        test_wr, test_ret = evaluate_parallel(venv, policy, obs_norm, agent_ids,
                                              args.final_eval_episodes, ep_limit,
                                              device, N)
        print(f"[par/{args.comm}] FINAL greedy test_win_rate={test_wr:.4f}", flush=True)

    save_checkpoint(steps, final=True)
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                "args": safe_config, "steps": steps,
                "final_test_win_rate": test_wr, "final_test_return": test_ret,
                "final_train_win_rate": wr}, out / "final.pt")
    print(f"[par/{args.comm}] saved -> {out}/final.pt", flush=True)
    venv.close()
    if run is not None:
        run.finish()
    return {"test_win_rate": test_wr, "train_win_rate": wr}


@torch.no_grad()
def evaluate_parallel(venv, policy, obs_norm, agent_ids, n_episodes, ep_limit,
                      device, N):
    """Greedy evaluation across the B parallel envs until n_episodes finish.

    Uses argmax actions. Counts each env's natural terminations. Because the vec
    wrapper auto-resets, we simply keep stepping and tally finished episodes.
    """
    policy.eval()
    B = venv.num_envs
    obs_np, state_np, avail_np = venv.reset()
    o = torch.as_tensor(obs_np, device=device)
    ids = agent_ids.unsqueeze(0).expand(B, -1, -1)
    o = torch.cat([o, ids], dim=-1)
    av = torch.as_tensor(avail_np, device=device)
    alive = (av[:, :, 0] < 0.5).float()
    h = policy.init_hidden(B, N, device)
    steps_in_ep = np.zeros(B, dtype=np.int64)

    wins, done_count = [], 0
    guard = 0
    while done_count < n_episodes and guard < n_episodes * ep_limit + ep_limit:
        guard += 1
        logits, h = policy(obs_norm(o), h, alive)
        logits = logits.masked_fill(av < 0.5, -1e9)
        act = logits.argmax(dim=-1)                     # (B,N)
        n_obs, n_state, n_avail, reward, term, infos = venv.step(act.cpu().numpy())
        steps_in_ep += 1
        trunc = steps_in_ep >= ep_limit
        for b in range(B):
            if term[b] or trunc[b]:
                if done_count < n_episodes:
                    wins.append(int(infos[b].get("battle_won", False)))
                    done_count += 1
                steps_in_ep[b] = 0
                h[b] = 0.0
        o = torch.as_tensor(n_obs, device=device)
        o = torch.cat([o, ids], dim=-1)
        av = torch.as_tensor(n_avail, device=device)
        alive = (av[:, :, 0] < 0.5).float()
    policy.train()
    wr = float(np.mean(wins)) if wins else 0.0
    return wr, float("nan")


if __name__ == "__main__":
    train_parallel(build_args())

