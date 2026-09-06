"""Mutual-information estimators for the communication-effectiveness analysis.

Two estimators over paired samples (x_i, y_i):

  * MINE  -- Donsker-Varadhan lower bound (Belghazi et al., 2018), with the
             exponential-moving-average bias correction on the second term.
  * InfoNCE -- the noise-contrastive lower bound (Oord et al., 2018); lower
             variance, upper-bounded by log(batch_size). Used as a cross-check.

Both return an estimate in NATS. Report both; if they disagree strongly, trust
InfoNCE and say so. These operate on any (x, y): use them for I(message; obs)
and, more importantly, I(message; enemy_visible).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class _StatNet(nn.Module):
    """T(x, y): concatenate and score with a small MLP."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)


def _as_2d(a: torch.Tensor) -> torch.Tensor:
    return a.float().unsqueeze(-1) if a.dim() == 1 else a.float()


def estimate_mine(x: torch.Tensor, y: torch.Tensor, *, hidden: int = 128,
                  iters: int = 2000, lr: float = 1e-3, batch: int = 512,
                  ema_alpha: float = 0.01, seed: int = 0,
                  device: str = "cpu") -> float:
    """MINE lower bound on I(X; Y) in nats. x, y are (S, .) or (S,) aligned by row."""
    torch.manual_seed(seed)
    x = _as_2d(x).to(device)
    y = _as_2d(y).to(device)
    S = x.shape[0]
    if S < 8:
        return float("nan")
    T = _StatNet(x.shape[1], y.shape[1], hidden).to(device)
    opt = torch.optim.Adam(T.parameters(), lr=lr)
    ema = None
    last = []
    for it in range(iters):
        idx = torch.randint(0, S, (min(batch, S),), device=device)
        xb, yb = x[idx], y[idx]
        yb_shuf = yb[torch.randperm(yb.shape[0], device=device)]  # marginal
        t_joint = T(xb, yb)
        t_marg = T(xb, yb_shuf)
        # DV bound estimate.
        exp_marg = torch.exp(t_marg)
        mi = t_joint.mean() - torch.log(exp_marg.mean() + 1e-8)
        # Bias-corrected gradient for the log-mean-exp term (Belghazi et al.).
        ema = exp_marg.mean().detach() if ema is None \
            else (1 - ema_alpha) * ema + ema_alpha * exp_marg.mean().detach()
        loss = -(t_joint.mean() - exp_marg.mean() / (ema + 1e-8))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it >= iters - 100:
            last.append(mi.item())
    return float(np.mean(last)) if last else float("nan")


def estimate_infonce(x: torch.Tensor, y: torch.Tensor, *, hidden: int = 128,
                     iters: int = 2000, lr: float = 1e-3, batch: int = 512,
                     seed: int = 0, device: str = "cpu") -> float:
    """InfoNCE lower bound on I(X; Y) in nats (<= log(batch))."""
    torch.manual_seed(seed)
    x = _as_2d(x).to(device)
    y = _as_2d(y).to(device)
    S = x.shape[0]
    if S < 8:
        return float("nan")
    T = _StatNet(x.shape[1], y.shape[1], hidden).to(device)
    opt = torch.optim.Adam(T.parameters(), lr=lr)
    last = []
    for it in range(iters):
        b = min(batch, S)
        idx = torch.randint(0, S, (b,), device=device)
        xb, yb = x[idx], y[idx]
        # Score matrix: scores[i, j] = T(x_i, y_j). Diagonal = positive pairs.
        xe = xb.unsqueeze(1).expand(b, b, xb.shape[1])
        ye = yb.unsqueeze(0).expand(b, b, yb.shape[1])
        scores = T.net(torch.cat([xe, ye], dim=-1)).squeeze(-1)   # (b, b)
        labels = torch.arange(b, device=device)
        loss = nn.functional.cross_entropy(scores, labels)
        mi = np.log(b) - loss.item()          # InfoNCE bound
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it >= iters - 100:
            last.append(mi)
    return float(np.mean(last)) if last else float("nan")
