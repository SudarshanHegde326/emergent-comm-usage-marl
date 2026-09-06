"""Channel constraints for the emergent-communication study.

Implements the four constraint classes named in the Terms of Reference as a
SINGLE module that transforms a per-agent message tensor of shape (B, N, msg_dim):

  1. Bandwidth quantisation   -- deterministic, straight-through estimator (PPO-safe)
  2. KL information bottleneck -- variational message, KL term exposed as .last_kl
  3. Gaussian channel noise   -- additive noise (primarily a TEST-TIME robustness knob)
  4. Random agent dropout     -- whole senders muted at random (TEST-TIME robustness knob)

DESIGN CONTRACT
---------------
With every knob at its default (quant_bits=0, noise_std=0, dropout_p=0,
kl_bottleneck=False) this module is the IDENTITY and exposes last_kl=0. So a
trainer that constructs it with defaults behaves numerically exactly as it did
before the module existed. Nothing changes until you turn a knob on.

PPO NOTE (read before training WITH a constraint on)
----------------------------------------------------
* Quantisation is a deterministic function of the message, so the stored action
  log-probs and the replayed ones match -> PPO ratio stays valid. Safe to train with.
* KL bottleneck SAMPLES the message in training mode. That injects stochasticity
  into the message, so like any stochastic regulariser under PPO it introduces a
  small mismatch between the sampled message at collection and at replay. Keep the
  weight (--kl-coef) modest and treat it as a regulariser; evaluation uses the mean
  (deterministic), so your reported win rates are clean. State this in the write-up.
* Noise and agent-dropout are stochastic and are intended mainly as DEPLOYMENT
  perturbations for the robustness sweep (apply them to a trained model at eval via
  set_channel_cfg, then measure % of performance retained). You *can* switch them on
  during training, but then the same PPO caveat as the bottleneck applies.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def quantize_ste(x: torch.Tensor, bits: int, value_range: float) -> torch.Tensor:
    """Uniformly quantise x to 2**bits levels over [-value_range, value_range],
    with a straight-through estimator so gradients pass unchanged.

    The forward output is the quantised value; the backward gradient is identity.
    """
    if bits <= 0:
        return x
    levels = 2 ** bits
    # Clamp into the representable range, then round to the nearest level.
    xc = x.clamp(-value_range, value_range)
    # Map [-r, r] -> [0, levels-1], round, map back.
    scaled = (xc + value_range) / (2 * value_range) * (levels - 1)
    q = torch.round(scaled) / (levels - 1) * (2 * value_range) - value_range
    # Straight-through: value of q, gradient of x.
    return x + (q - x).detach()


class ChannelConstraint(nn.Module):
    """Transforms a message tensor (B, N, msg_dim) under up to four constraints.

    Parameters
    ----------
    msg_dim : int
        Message width (needed only when kl_bottleneck is on, for the logvar head).
    quant_bits : int
        Bits per message dimension. 0 disables quantisation.
    noise_std : float
        Std of additive Gaussian channel noise. 0 disables.
    dropout_p : float
        Probability that a whole sender's message is muted. 0 disables.
    kl_bottleneck : bool
        If True, treat the incoming message as the mean of a Gaussian, learn a
        per-dimension log-variance, sample in training mode, and expose the KL to
        a unit Gaussian prior as .last_kl for the trainer to add to the loss.
    quant_range : float
        Symmetric clip range for quantisation (messages pass through tanh-like
        magnitudes; 2.0 comfortably covers a LayerNorm/gated encoder output).
    """

    def __init__(self, msg_dim: int, quant_bits: int = 0, noise_std: float = 0.0,
                 dropout_p: float = 0.0, kl_bottleneck: bool = False,
                 quant_range: float = 2.0):
        super().__init__()
        if quant_bits < 0:
            raise ValueError(f"quant_bits must be >= 0, got {quant_bits}")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.quant_bits = quant_bits
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.quant_range = quant_range
        self.kl_bottleneck = kl_bottleneck
        if kl_bottleneck:
            # mu is the incoming message; only the log-variance is learned.
            self.logvar_head = nn.Linear(msg_dim, msg_dim)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)   # start at sigma=1 (KL small)
        self.last_kl = None

    @property
    def is_identity(self) -> bool:
        return (self.quant_bits == 0 and self.noise_std == 0.0
                and self.dropout_p == 0.0 and not self.kl_bottleneck)

    def forward(self, msg: torch.Tensor) -> torch.Tensor:
        m = msg
        kl = msg.new_zeros(())

        # (2) KL information bottleneck ------------------------------------
        if self.kl_bottleneck:
            mu = m
            logvar = self.logvar_head(m).clamp(-8.0, 8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                m = mu + std * torch.randn_like(std)
            else:
                m = mu                                   # deterministic eval
            # KL(N(mu, sigma) || N(0, 1)), summed over dims, averaged over agents.
            kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()

        # (1) Bandwidth quantisation (deterministic, PPO-safe) -------------
        if self.quant_bits > 0:
            m = quantize_ste(m, self.quant_bits, self.quant_range)

        # (3) Gaussian channel noise ---------------------------------------
        if self.noise_std > 0.0:
            m = m + self.noise_std * torch.randn_like(m)

        # (4) Random agent dropout (mute whole senders) --------------------
        if self.dropout_p > 0.0:
            keep = (torch.rand(m.shape[:-1], device=m.device) >= self.dropout_p)
            m = m * keep.unsqueeze(-1).to(m.dtype)

        self.last_kl = kl
        return m


def set_channel_cfg(policy: nn.Module, quant_bits=None, noise_std=None,
                    dropout_p=None) -> int:
    """Override channel-constraint knobs on every ChannelConstraint in a policy,
    in place, for a test-time robustness sweep. Returns how many modules were set.

    Only the arguments you pass are changed; pass None to leave a knob untouched.
    KL bottleneck is a training-time architectural choice and is deliberately NOT
    settable here.

    Usage (robustness sweep on a trained checkpoint):
        for s in (0.1, 0.5, 1.0):
            set_channel_cfg(policy, noise_std=s)
            wr, _ = evaluate(env, policy, ...)   # % retained vs the clean win rate
    """
    n = 0
    for mod in policy.modules():
        if isinstance(mod, ChannelConstraint):
            if quant_bits is not None:
                mod.quant_bits = int(quant_bits)
            if noise_std is not None:
                mod.noise_std = float(noise_std)
            if dropout_p is not None:
                mod.dropout_p = float(dropout_p)
            n += 1
    return n
