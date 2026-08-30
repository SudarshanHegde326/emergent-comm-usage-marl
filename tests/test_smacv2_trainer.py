"""Tests for the SMACv2 MAPPO trainer. NO StarCraft II installation required.

These tests exercise the code that actually runs on the HPC. The previous
test suite only covered src/comms/, which the SMACv2 trainer does not import,
so every SMACv2 bug was invisible to pytest.

Run:  pytest tests/test_smacv2_trainer.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "trainers" / "14_mappo_smacv2_matched.py"


def _load():
    spec = importlib.util.spec_from_file_location("_smacv2_trainer", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_smacv2_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


# =============================================================================
# A fake SMACv2 environment with the exact API surface the trainer uses.
# Agents "die" over time so the death-masking paths are genuinely exercised.
# =============================================================================
class FakeSMAC:
    def __init__(self, n_agents=5, n_actions=11, obs_dim=92, state_dim=130,
                 ep_limit=20, seed=0):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.ep_limit = ep_limit
        self.rng = np.random.RandomState(seed)
        self.closed = False
        self.reset()

    def get_env_info(self):
        return {"n_agents": self.n_agents, "n_actions": self.n_actions,
                "obs_shape": self.obs_dim, "state_shape": self.state_dim,
                "episode_limit": self.ep_limit}

    def reset(self):
        self.t = 0
        self.health = np.ones(self.n_agents)
        return self.get_obs(), self.get_state()

    # SMACv2-style obs-size helpers so the aux-sighting path can locate the
    # enemy-feature block (move=4, enemy=5*4=20, rest is ally/own).
    def get_obs_move_feats_size(self):
        return 4

    def get_obs_enemy_feats_size(self):
        return (5, 4)

    def get_obs(self):
        obs = self.rng.randn(self.n_agents, self.obs_dim).astype(np.float32)
        obs[:, 4:24] = 0.0                       # enemy block: zero == unseen
        seers = self.rng.rand(self.n_agents) < 0.4
        for i in range(self.n_agents):
            if seers[i] and self.health[i] > 0:
                obs[i, 4:8] = self.rng.rand(4) + 0.1   # this agent sees an enemy
        return obs

    def get_state(self):
        return self.rng.randn(self.state_dim).astype(np.float32)

    def get_avail_actions(self):
        av = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for i in range(self.n_agents):
            if self.health[i] > 0:
                av[i, 1:] = 1.0          # alive: no-op forbidden
            else:
                av[i, 0] = 1.0           # dead: only no-op
        return av

    def step(self, actions):
        assert len(actions) == self.n_agents
        self.t += 1
        # kill one agent every few steps so `alive` actually varies
        if self.t % 4 == 0:
            living = np.flatnonzero(self.health > 0)
            if len(living) > 1:
                self.health[self.rng.choice(living)] = 0.0
        terminated = (self.health.sum() <= 1) or (self.t >= self.ep_limit)
        return float(self.rng.rand()), bool(terminated), {"battle_won": False}

    def close(self):
        self.closed = True


def _args(**over):
    argv = ["--comm", over.pop("comm", "none"), "--no-wandb"]
    for k, v in over.pop("flags", {}).items():
        argv += [k, str(v)]
    a = M.build_args(argv)
    for k, v in over.items():
        setattr(a, k, v)
    return a


# =============================================================================
# 1. Argument parsing must accept every flag the launcher scripts pass.
#    This is the exact failure that killed the previous attention/GNN runs.
# =============================================================================
@pytest.mark.parametrize("flags", [
    ["--comm", "none", "--seed", "0", "--total-steps", "200000", "--hidden", "128"],
    ["--comm", "fc", "--seed", "0", "--msg-dim", "8", "--hidden", "128"],
    ["--comm", "attn", "--seed", "0", "--msg-dim", "8", "--num-heads", "2",
     "--hidden", "128"],
    ["--comm", "attn", "--seed", "0", "--msg-dim", "8", "--heads", "2"],
    ["--comm", "gnn", "--seed", "0", "--msg-dim", "8", "--num-layers", "2",
     "--num-heads", "2", "--hidden", "128"],
    ["--comm", "gat", "--mode", "epo", "--no-comm-to-memory"],
])
def test_launcher_flags_parse(flags):
    a = M.build_args(flags + ["--no-wandb"])
    assert a.comm in ("none", "fc", "gnn", "attn")


def test_gat_alias_normalised():
    assert M.build_args(["--comm", "gat", "--no-wandb"]).comm == "gnn"


# =============================================================================
# 2. The four communication conditions must be DIFFERENT models.
#    Previously FCComm and GNNComm were byte-identical.
# =============================================================================
def test_conditions_are_distinct_architectures():
    kinds = {}
    for name in ("none", "fc", "gnn", "attn"):
        m = M.build_comm(name, hidden=16, msg_dim=8, heads=2, layers=2)
        kinds[name] = type(m).__name__
    assert len(set(kinds.values())) == 4, f"conditions collapsed: {kinds}"
    assert kinds["gnn"] == "GATComm"


def test_gat_is_not_mean_aggregation():
    """The whole point of attention is that it weights neighbours unequally.

    Both inputs give agent 0 neighbours with the SAME mean but DIFFERENT
    content ({1,3} vs {2,2}). A *single-linear, ungated, unweighted mean* (FC
    with mlp=False, gate=False) cannot tell them apart; a graph attention
    network must. Note this is deliberately not a permutation of the
    neighbours -- attention is permutation-equivariant, so a permutation would
    prove nothing. FC's default MLP encoder is itself able to separate these
    (that is Upgrade 2's purpose), so the blind baseline here is FC with the
    MLP and gate switched off.
    """
    torch.manual_seed(0)
    gat = M.build_comm("gnn", hidden=8, msg_dim=8, heads=2, layers=1)
    fc_linear = M.build_comm("fc", hidden=8, msg_dim=8, heads=2, layers=1,
                             mlp=False, gate=False)
    for mod in (gat, fc_linear):
        for p in mod.parameters():
            if p.dim() == 2:
                torch.nn.init.xavier_uniform_(p)
    alive = torch.ones(1, 3)
    a = torch.tensor([[[0.5] * 8, [1.0] * 8, [3.0] * 8]])   # neighbour mean 2.0
    b = torch.tensor([[[0.5] * 8, [2.0] * 8, [2.0] * 8]])   # neighbour mean 2.0
    assert torch.allclose(fc_linear(a, alive)[0, 0], fc_linear(b, alive)[0, 0],
                          atol=1e-5), \
        "single-linear mean-pool FC should be blind to this difference"
    assert not torch.allclose(gat(a, alive)[0, 0], gat(b, alive)[0, 0], atol=1e-4), \
        "GAT collapsed to mean aggregation"


# =============================================================================
# 3. All conditions must be numerically identical at initialisation.
# =============================================================================
def test_all_conditions_identical_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 16)
    alive = torch.ones(2, 5)
    ref = None
    for name in ("none", "fc", "gnn", "attn"):
        torch.manual_seed(0)
        out = M.build_comm(name, 16, 8, 2, 2)(h, alive)
        if ref is None:
            ref = out
        else:
            assert torch.allclose(ref, out, atol=1e-6), f"{name} differs at init"


# =============================================================================
# 4. Dead agents must never contribute a message, in ANY condition.
# =============================================================================
@pytest.mark.parametrize("name", ["fc", "gnn", "attn"])
def test_dead_agents_do_not_influence_output(name):
    torch.manual_seed(0)
    mod = M.build_comm(name, 16, 8, 2, 2)
    for p in mod.parameters():
        if p.dim() == 2:
            torch.nn.init.xavier_uniform_(p)
    h = torch.randn(1, 4, 16)
    alive = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out_a = mod(h, alive)
    h2 = h.clone()
    h2[0, 2] = torch.randn(16) * 100     # scramble the dead agents
    h2[0, 3] = torch.randn(16) * 100
    out_b = mod(h2, alive)
    assert torch.allclose(out_a[0, :2], out_b[0, :2], atol=1e-5), \
        f"{name}: dead agents leaked into living agents' output"


@pytest.mark.parametrize("name", ["gnn", "attn"])
def test_sole_survivor_gets_zero_attention(name):
    """A lone survivor has no valid sender. The attention row is fully masked
    and must yield zero weight, not a uniform average over dead team-mates."""
    torch.manual_seed(0)
    mod = M.build_comm(name, 16, 8, 2, 1)
    for p in mod.parameters():
        if p.dim() == 2:
            torch.nn.init.xavier_uniform_(p)
    h = torch.randn(1, 4, 16)
    alive = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out_a = mod(h, alive)
    h2 = h.clone()
    h2[0, 1:] = torch.randn(3, 16) * 100
    out_b = mod(h2, alive)
    assert torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)


def test_masked_softmax_rows_sum_correctly():
    scores = torch.randn(2, 3, 4, 4)
    blocked = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
    blocked[..., 0, :] = True                      # fully blocked row
    w = M._masked_softmax(scores, blocked)
    assert torch.allclose(w[..., 0, :].sum(-1), torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(w[..., 1, :].sum(-1), torch.ones(2, 3), atol=1e-6)
    assert (w >= 0).all()


# =============================================================================
# 5. Communication must reach the recurrent state when enabled.
# =============================================================================
def test_comm_to_memory_changes_hidden_state():
    torch.manual_seed(0)
    obs = torch.randn(1, 4, 12)
    alive = torch.ones(1, 4)
    h0 = torch.zeros(1, 4, 16)

    on = M.RecurrentPolicy(12, 5, 16, "attn", 8, 2, 2, comm_to_memory=True)
    off = M.RecurrentPolicy(12, 5, 16, "attn", 8, 2, 2, comm_to_memory=False)
    off.load_state_dict(on.state_dict())

    _, h_on = on(obs, h0, alive)
    _, h_off = off(obs, h0, alive)
    # LayerNorm alone changes the representation, so the two must differ.
    assert not torch.allclose(h_on, h_off, atol=1e-5)


# =============================================================================
# 6. GAE: no bootstrapping across an episode boundary.
# =============================================================================
def test_gae_does_not_bootstrap_across_done():
    rew = torch.tensor([1.0, 1.0, 1.0])
    val = torch.tensor([0.0, 100.0, 0.0])
    done = torch.tensor([1.0, 0.0, 0.0])        # episode ends at t=0
    adv, ret = M.compute_gae(rew, val, done, torch.tensor(0.0), 0.99, 0.95)
    assert abs(adv[0].item() - 1.0) < 1e-6, "value from the next episode leaked in"
    assert ret.shape == rew.shape


# =============================================================================
# 7. Gradients must flow into the communication module in every condition.
# =============================================================================
@pytest.mark.parametrize("comm", ["fc", "gnn", "attn"])
def test_gradients_reach_comm_module(comm):
    torch.manual_seed(0)
    pol = M.RecurrentPolicy(12, 5, 16, comm, 8, 2, 2)
    obs = torch.randn(2, 4, 12)
    logits, _ = pol(obs, torch.zeros(2, 4, 16), torch.ones(2, 4))
    logits.sum().backward()
    grads = [p.grad for p in pol.comm.parameters() if p.grad is not None]
    assert grads, f"{comm}: no gradient reached the communication module"
    assert any(g.abs().sum() > 0 for g in grads), \
        f"{comm}: all communication gradients are exactly zero"


# =============================================================================
# 8. End-to-end training smoke test on the fake environment.
# =============================================================================
@pytest.mark.parametrize("comm", ["none", "fc", "gnn", "attn"])
def test_end_to_end_training_runs(comm, tmp_path):
    a = _args(comm=comm)
    a.total_steps = 80
    a.rollout_len = 40
    a.chunk_len = 20
    a.ppo_epochs = 2
    a.hidden = 16
    a.msg_dim = 8
    a.num_heads = 2
    a.num_layers = 2
    a.eval_episodes = 2
    a.final_eval_episodes = 2
    a.eval_interval = 40
    a.out_dir = str(tmp_path / f"exp_{comm}")
    a._env_factory = lambda _a: FakeSMAC(seed=0)
    res = M.train(a)
    assert 0.0 <= res["test_win_rate"] <= 1.0
    saved = tmp_path / f"exp_{comm}" / "seed0" / "final.pt"
    assert saved.exists()
    ck = torch.load(saved, map_location="cpu", weights_only=False)
    assert "final_test_win_rate" in ck


def test_rollout_len_must_divide_chunk_len():
    a = _args(comm="none")
    a.total_steps = 40
    a.rollout_len = 30
    a.chunk_len = 40
    a._env_factory = lambda _a: FakeSMAC(seed=0)
    with pytest.raises(ValueError, match="exact multiple"):
        M.train(a)


# =============================================================================
# 9. EPO configuration must match the benchmark's own config exactly.
#    (Checks the kwargs we would pass; does not launch StarCraft II.)
# =============================================================================
def test_epo_config_matches_official_yaml(monkeypatch):
    captured = {}

    class _Spy:
        def __init__(self, **kw):
            captured.update(kw)

    import types
    fake_wrapper = types.ModuleType("smacv2.env.starcraft2.wrapper")
    fake_wrapper.StarCraftCapabilityEnvWrapper = _Spy
    fake_sc2 = types.ModuleType("smacv2.env.starcraft2")
    fake_env = types.ModuleType("smacv2.env")
    fake_root = types.ModuleType("smacv2")
    monkeypatch.setitem(sys.modules, "smacv2", fake_root)
    monkeypatch.setitem(sys.modules, "smacv2.env", fake_env)
    monkeypatch.setitem(sys.modules, "smacv2.env.starcraft2", fake_sc2)
    monkeypatch.setitem(sys.modules, "smacv2.env.starcraft2.wrapper", fake_wrapper)

    M.make_env("standard", seed=0)
    assert captured["prob_obs_enemy"] == 1.0
    assert captured["action_mask"] is True
    assert captured["conic_fov"] is False
    assert captured["capability_config"]["n_units"] == 5

    captured.clear()
    M.make_env("epo", seed=0)
    # The ONLY two differences in the benchmark's own EPO config.
    assert captured["prob_obs_enemy"] == 0.0
    assert captured["action_mask"] is False
    # conic_fov is False in BOTH official configs; it is NOT what makes EPO.
    assert captured["conic_fov"] is False
    assert captured["capability_config"]["n_units"] == 6      # paper EPO is 6v5
    assert captured["capability_config"]["n_enemies"] == 5


# =============================================================================
# 10. RunningNorm must leave the one-hot agent-ID tail untouched (Fix #2).
# =============================================================================
def test_running_norm_preserves_agent_id_tail():
    device = torch.device("cpu")
    raw_dim, n_agents = 6, 3
    obs_dim = raw_dim + n_agents
    norm = M.RunningNorm(obs_dim, device, active_dim=raw_dim)
    # Feed batches with large-magnitude raw features and clean one-hot tails.
    for _ in range(50):
        raw = torch.randn(4, raw_dim) * 7.0 + 3.0
        ids = torch.eye(n_agents)[torch.randint(0, n_agents, (4,))]
        norm.update(torch.cat([raw, ids], dim=-1))
    probe_ids = torch.eye(n_agents)
    x = torch.cat([torch.randn(n_agents, raw_dim) * 7.0, probe_ids], dim=-1)
    out = norm(x)
    # The tail must be IDENTICAL to the input one-hot (still clean 0/1).
    assert torch.allclose(out[:, raw_dim:], probe_ids, atol=1e-6), \
        "agent-ID columns were altered by normalisation"
    # The raw head must actually be standardised (mean shrunk toward 0).
    assert out[:, :raw_dim].abs().mean() < x[:, :raw_dim].abs().mean()


def test_running_norm_full_width_still_works():
    """active_dim=None keeps the original behaviour (normalise everything)."""
    norm = M.RunningNorm(5, torch.device("cpu"))
    for _ in range(20):
        norm.update(torch.randn(8, 5) * 4 + 2)
    out = norm(torch.randn(3, 5) * 4 + 2)
    assert out.shape == (3, 5)
    assert out.abs().mean() < 2.0


def test_running_norm_state_roundtrip():
    a = M.RunningNorm(7, torch.device("cpu"), active_dim=4)
    for _ in range(10):
        a.update(torch.randn(6, 7))
    b = M.RunningNorm(7, torch.device("cpu"), active_dim=4)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 7)
    assert torch.allclose(a(x), b(x), atol=1e-7)


# =============================================================================
# 11. Checkpoint + resume must continue, not restart (Fix #1).
# =============================================================================
def _short_args(tmp_path, comm="attn", **over):
    a = _args(comm=comm)
    a.total_steps = over.pop("total_steps", 80)
    a.rollout_len = 40
    a.chunk_len = 20
    a.ppo_epochs = 2
    a.hidden = 16
    a.msg_dim = 8
    a.num_heads = 2
    a.num_layers = 2
    a.eval_episodes = 0            # keep the smoke test fast
    a.final_eval_episodes = 0
    a.eval_interval = 40
    a.ckpt_interval = over.pop("ckpt_interval", 40)
    a.out_dir = str(tmp_path)
    a._env_factory = lambda _a: FakeSMAC(seed=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_checkpoint_is_written(tmp_path):
    a = _short_args(tmp_path)
    M.train(a)
    assert (tmp_path / "seed0" / "latest.pt").exists()
    assert (tmp_path / "seed0" / "final.pt").exists()
    ck = torch.load(tmp_path / "seed0" / "latest.pt",
                    map_location="cpu", weights_only=False)
    for key in ("steps", "policy", "critic", "opt_a", "opt_c",
                "obs_norm", "state_norm", "torch_rng", "numpy_rng"):
        assert key in ck, f"checkpoint missing '{key}'"
    assert ck["steps"] >= a.total_steps


def test_resume_continues_from_checkpoint(tmp_path, capsys):
    # Phase 1: train to 40 steps and stop.
    a1 = _short_args(tmp_path, total_steps=40, ckpt_interval=40)
    M.train(a1)
    ck1 = torch.load(tmp_path / "seed0" / "latest.pt",
                     map_location="cpu", weights_only=False)
    steps_after_phase1 = ck1["steps"]
    assert steps_after_phase1 >= 40

    # Phase 2: resume with a higher target; must continue, not restart at 0.
    a2 = _short_args(tmp_path, total_steps=120, ckpt_interval=40)
    a2.resume = True
    M.train(a2)
    out = capsys.readouterr().out
    assert "RESUMED" in out, "resume did not report continuing from checkpoint"
    ck2 = torch.load(tmp_path / "seed0" / "latest.pt",
                     map_location="cpu", weights_only=False)
    assert ck2["steps"] >= 120


def test_resume_without_checkpoint_starts_fresh(tmp_path, capsys):
    a = _short_args(tmp_path, total_steps=40)
    a.resume = True                # asked to resume, but nothing saved yet
    M.train(a)
    out = capsys.readouterr().out
    assert "starting fresh" in out
    assert (tmp_path / "seed0" / "latest.pt").exists()


def test_completed_run_resume_is_noop(tmp_path, capsys):
    a1 = _short_args(tmp_path, total_steps=40)
    M.train(a1)
    a2 = _short_args(tmp_path, total_steps=40)
    a2.resume = True
    M.train(a2)
    out = capsys.readouterr().out
    assert "already at target" in out


# =============================================================================
# 12. Optimiser state must survive a resume (not silently reset).
# =============================================================================
def test_resume_restores_optimiser_state(tmp_path):
    a1 = _short_args(tmp_path, total_steps=40, ckpt_interval=40)
    M.train(a1)
    ck = torch.load(tmp_path / "seed0" / "latest.pt",
                    map_location="cpu", weights_only=False)
    # Adam should have accumulated per-parameter state by now.
    assert ck["opt_a"]["state"], "actor optimiser state was not checkpointed"
    assert ck["opt_c"]["state"], "critic optimiser state was not checkpointed"


# =============================================================================
# 13. Message-pathway upgrades: gating, MLP encoder, auxiliary sighting loss.
# =============================================================================
def test_gate_can_close_for_blind_agents():
    """A learned gate must be able to drive a blind agent's message toward 0,
    so mean-pooling is not dominated by uninformative broadcasts (the EPO fix)."""
    torch.manual_seed(0)
    enc = M.MessageEncoder(hidden=16, msg_dim=8, mlp=True, gate=True)
    # Force the gate to "closed" by setting its bias very negative.
    with torch.no_grad():
        enc.gate.bias.fill_(-20.0)
    h = torch.randn(2, 4, 16)
    out = enc(h)
    assert out.abs().max() < 1e-3, "closed gate did not suppress the message"
    assert enc.last_gate.max() < 1e-3
    # And fully open (large positive bias) lets a non-trivial message through.
    with torch.no_grad():
        enc.gate.bias.fill_(20.0)
    out2 = enc(h)
    assert out2.abs().max() > 1e-3
    assert enc.last_gate.min() > 0.99


def test_gating_preserves_signal_through_mean():
    """With every gate closed, each emitted message is ~0, so the neighbour mean
    is ~0 -- i.e. blind agents contribute nothing to the pool (the EPO fix).
    We check the emitted messages and their mean directly, which is the precise
    invariant, rather than the post-transform output (transform is zero-init, so
    testing through it would only re-test that)."""
    torch.manual_seed(0)
    fc = M.build_comm("fc", hidden=16, msg_dim=8, heads=2, layers=1,
                      mlp=True, gate=True)
    with torch.no_grad():
        fc.encoder.gate.weight.zero_()
        fc.encoder.gate.bias.fill_(-20.0)          # all gates closed
    h = torch.randn(1, 5, 16)
    alive = torch.ones(1, 5)
    fc(h, alive)
    # Every emitted message must be ~0 ...
    assert fc.last_message.abs().max() < 1e-3, "closed gates did not zero messages"
    # ... so the mean-excluding-self of the messages is ~0 too.
    agg = M._mean_excluding_self(fc.last_message, alive)
    assert agg.abs().max() < 1e-3, "blind agents still contributed to the pool"

    # And with one gate forced open, that agent DOES contribute.
    with torch.no_grad():
        fc.encoder.gate.bias.fill_(-20.0)
    h2 = h.clone()
    # make agent 0 produce a large gate via a big activation on a chosen dim
    with torch.no_grad():
        fc.encoder.gate.weight.zero_()
        fc.encoder.gate.weight[0, 0] = 50.0        # gate opens when h[...,0] > 0
        h2[0, 0, 0] = 1.0                           # agent 0 open
        h2[0, 1:, 0] = -1.0                         # others closed
    fc(h2, alive)
    per_agent_norm = fc.last_message.abs().sum(-1)[0]   # (5,)
    assert per_agent_norm[0] > per_agent_norm[1:].max(), \
        "the open agent should emit a larger message than the closed ones"


def test_mlp_encoder_has_more_capacity_than_linear():
    """The MLP encoder must be able to separate two neighbour sets with the same
    mean; the single-linear encoder cannot (that is why Upgrade 2 exists)."""
    torch.manual_seed(0)
    mlp = M.MessageEncoder(8, 8, mlp=True, gate=False)
    lin = M.MessageEncoder(8, 8, mlp=False, gate=False)
    for enc in (mlp, lin):
        for p in enc.parameters():
            if p.dim() == 2:
                torch.nn.init.xavier_uniform_(p)
    a = torch.tensor([[[1.0] * 8, [3.0] * 8]])          # mean 2
    b = torch.tensor([[[2.0] * 8, [2.0] * 8]])          # mean 2
    # After mean over the two neighbours, linear encoder gives identical pooled
    # vectors; MLP encoder does not.
    lin_pooled_a = lin(a).mean(1)
    lin_pooled_b = lin(b).mean(1)
    mlp_pooled_a = mlp(a).mean(1)
    mlp_pooled_b = mlp(b).mean(1)
    assert torch.allclose(lin_pooled_a, lin_pooled_b, atol=1e-5)
    assert not torch.allclose(mlp_pooled_a, mlp_pooled_b, atol=1e-4)


def test_all_conditions_identical_at_init_with_upgrades():
    """Gating (starts mostly open) + MLP encoder must STILL leave every
    condition numerically identical at initialisation, because the output
    projection is zero-init. This protects the clean experimental baseline."""
    torch.manual_seed(0)
    h = torch.randn(2, 5, 16)
    alive = torch.ones(2, 5)
    ref = None
    for name in ("none", "fc", "gnn", "attn"):
        torch.manual_seed(0)
        out = M.build_comm(name, 16, 8, 2, 2, mlp=True, gate=True)(h, alive)
        if ref is None:
            ref = out
        else:
            assert torch.allclose(ref, out, atol=1e-6), \
                f"{name} differs at init even with upgrades on"


def test_aux_head_only_when_enabled():
    torch.manual_seed(0)
    on = M.RecurrentPolicy(12, 5, 16, "attn", 8, 2, 2, aux_sighting=True)
    off = M.RecurrentPolicy(12, 5, 16, "attn", 8, 2, 2, aux_sighting=False)
    none = M.RecurrentPolicy(12, 5, 16, "none", 8, 2, 2, aux_sighting=True)
    obs = torch.randn(1, 4, 12)
    h0 = torch.zeros(1, 4, 16)
    on(obs, h0, torch.ones(1, 4)); off(obs, h0, torch.ones(1, 4))
    assert on.aux_logits() is not None
    assert off.aux_logits() is None
    # NoComm has no message, so aux must be disabled even if requested.
    assert none.aux_sighting is False


@pytest.mark.parametrize("comm", ["fc", "gnn", "attn"])
def test_aux_gradients_reach_encoder(comm):
    """The auxiliary loss must actually push gradients into the message
    encoder, so it can shape the messages."""
    torch.manual_seed(0)
    pol = M.RecurrentPolicy(12, 5, 16, comm, 8, 2, 2, aux_sighting=True)
    obs = torch.randn(3, 4, 12)
    pol(obs, torch.zeros(3, 4, 16), torch.ones(3, 4))
    aux = pol.aux_logits()                       # (3,4)
    target = torch.randint(0, 2, (3, 4)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(aux, target)
    loss.backward()
    enc_grads = [p.grad for p in pol.comm.encoder.parameters()
                 if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads), \
        f"{comm}: aux loss did not reach the message encoder"


@pytest.mark.parametrize("gate,mlp,aux", [
    (True, True, 0.1),      # all upgrades on (the new default)
    (False, False, 0.0),    # everything off (old behaviour)
    (True, False, 0.0),     # gate only
    (False, True, 0.1),     # mlp + aux, no gate
])
def test_end_to_end_with_upgrade_combos(gate, mlp, aux, tmp_path):
    a = _args(comm="attn")
    a.total_steps = 80
    a.rollout_len = 40
    a.chunk_len = 20
    a.ppo_epochs = 2
    a.hidden = 16
    a.msg_dim = 8
    a.num_heads = 2
    a.num_layers = 2
    a.eval_episodes = 0
    a.final_eval_episodes = 0
    a.eval_interval = 40
    a.ckpt_interval = 40
    a.gate = gate
    a.mlp_encoder = mlp
    a.aux_coef = aux
    a.out_dir = str(tmp_path / f"exp_{gate}_{mlp}_{aux}")
    a._env_factory = lambda _a: FakeSMAC(seed=0)
    res = M.train(a)
    assert 0.0 <= res["train_win_rate"] <= 1.0 or res["train_win_rate"] != res["train_win_rate"]


# =============================================================================
# 14. --pool-type switch: mean vs max aggregation for FC.
# =============================================================================
def test_pool_type_flag_parses():
    assert M.build_args(["--comm", "fc", "--no-wandb"]).pool_type == "max"   # default changed to max
    assert M.build_args(["--comm", "fc", "--pool-type", "max",
                         "--no-wandb"]).pool_type == "max"


def test_fc_mean_vs_max_differ():
    """Mean and max FC must produce different outputs on the same input, or the
    switch isn't actually doing anything."""
    torch.manual_seed(0)
    h = torch.randn(2, 4, 16)
    alive = torch.ones(2, 4)
    fc_mean = M.build_comm("fc", 16, 8, 2, 1, mlp=True, gate=False, pool_type="mean")
    fc_max = M.build_comm("fc", 16, 8, 2, 1, mlp=True, gate=False, pool_type="max")
    for mod in (fc_mean, fc_max):
        torch.nn.init.normal_(mod.transform.weight, std=0.5)
    assert not torch.allclose(fc_mean(h, alive), fc_max(h, alive), atol=1e-4)


def test_max_pool_ignores_dilution():
    """The point of max under EPO: one strong signal survives regardless of how
    many near-zero (blind) messages surround it."""
    fc_max = M.build_comm("fc", 8, 8, 2, 1, mlp=False, gate=False, pool_type="max")
    alive = torch.ones(1, 5)
    # agent 1 sends a strong positive signal; the rest send ~0.
    m = torch.zeros(1, 5, 8)
    m[0, 1] = 9.0
    agg = M._max_excluding_self(m, alive)
    # every OTHER agent should see agent 1's 9 in their aggregate.
    for i in [0, 2, 3, 4]:
        assert torch.allclose(agg[0, i], torch.full((8,), 9.0), atol=1e-4), \
            f"agent {i} did not receive the strong signal via max"


def test_max_pool_no_neg_inf_leak():
    """Dead/self masking uses -1e9, but real negative messages must NOT be
    replaced by -1e9 in the output."""
    fc_max_fn = M._max_excluding_self
    alive = torch.ones(1, 3)
    m = torch.tensor([[[0.0] * 4, [-5.0] * 4, [-3.0] * 4]])   # all negative
    agg = fc_max_fn(m, alive)
    # agent 0 sees max(-5,-3) = -3, NOT -1e9
    assert torch.allclose(agg[0, 0], torch.full((4,), -3.0), atol=1e-4), \
        "-1e9 mask leaked into a real negative message"


def test_max_pool_sole_survivor_zero():
    """A lone survivor (no valid sender) must get 0, not -1e9."""
    alive = torch.tensor([[1.0, 0.0, 0.0]])
    m = torch.randn(1, 3, 4)
    agg = M._max_excluding_self(m, alive)
    assert torch.allclose(agg[0, 0], torch.zeros(4), atol=1e-6)


def test_pool_type_preserves_init_invariance():
    """Even with pool_type=max, all conditions stay identical at init (the
    output projection is zero-init)."""
    torch.manual_seed(0)
    h = torch.randn(2, 5, 16)
    alive = torch.ones(2, 5)
    torch.manual_seed(0)
    none = M.build_comm("none", 16, 8, 2, 2)(h, alive)
    torch.manual_seed(0)
    fc_max = M.build_comm("fc", 16, 8, 2, 2, mlp=True, gate=True, pool_type="max")(h, alive)
    assert torch.allclose(none, fc_max, atol=1e-6), \
        "FC with max-pooling is not identical to baseline at init"