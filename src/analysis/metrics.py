"""Sparsity metrics and effect-size statistics for the analysis layer."""

from __future__ import annotations
import numpy as np
import torch


# ----------------------------- sparsity ----------------------------------

def hoyer_sparsity(m: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean Hoyer sparsity of message vectors, in [0, 1].

    m : (S, msg_dim) messages (living agents only, already flattened over agents).
    1.0 = maximally sparse (one active dim); 0.0 = fully dense/uniform.
    """
    m = m.float()
    d = m.shape[-1]
    if d <= 1:
        return float("nan")
    l1 = m.abs().sum(-1)
    l2 = m.pow(2).sum(-1).sqrt()
    ratio = l1 / (l2 + eps)                        # in [1, sqrt(d)]
    hoyer = (np.sqrt(d) - ratio) / (np.sqrt(d) - 1.0)
    return float(hoyer.clamp(0.0, 1.0).mean())


def activation_sparsity(m: torch.Tensor, thresh: float = 1e-3) -> float:
    """Fraction of message dimensions whose magnitude is below `thresh`."""
    m = m.float()
    return float((m.abs() < thresh).float().mean())


def gate_open_rate(gate: torch.Tensor, thresh: float = 0.5) -> float:
    """Fraction of timesteps a gate is open (> thresh). 1.0 if gating is off."""
    if gate is None:
        return float("nan")
    return float((gate.float() > thresh).float().mean())


# --------------------------- effect sizes --------------------------------

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g (small-sample-corrected Cohen's d) for group a vs group b.

    Positive g means a > b. Uses the pooled SD and the (1 - 3/(4N-9)) correction,
    which matters at the tiny seed counts used here (e.g. n=3 per group).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-8:            # both groups ~constant: effect size undefined
        return float("nan")
    d = (a.mean() - b.mean()) / pooled
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * correction)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap (1-alpha) CI for Hedges' g of a vs b, resampling seeds."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    gs = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        gs[i] = hedges_g(ra, rb)
    gs = gs[np.isfinite(gs)]
    if gs.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(gs, 100 * alpha / 2))
    hi = float(np.percentile(gs, 100 * (1 - alpha / 2)))
    return (lo, hi)
