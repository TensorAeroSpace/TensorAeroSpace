"""Quantile-regression distributional critic with twin design.

References:
    Dabney et al. (2018) Distributional RL with Quantile Regression, AAAI.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CriticConfig:
    n_state: int
    n_action: int
    n_quantiles: int = 32
    hidden_sizes: tuple[int, ...] = (256, 256)
    huber_kappa: float = 1.0


class QRDistCritic(nn.Module):
    """MLP that emits N quantiles of the return distribution Z(s,a)."""

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(cfg.n_state) + int(cfg.n_action)
        layers: list[nn.Module] = []
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, int(cfg.n_quantiles)))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)


def qr_huber_loss(z_pred: torch.Tensor, z_target: torch.Tensor,
                  kappa: float) -> torch.Tensor:
    """Asymmetric Huber-quantile loss (Dabney 2018, eq. 10)."""
    n = int(z_pred.shape[-1])
    tau = (torch.arange(n, device=z_pred.device).float() + 0.5) / n  # (N,)
    delta = z_target.detach().unsqueeze(1) - z_pred.unsqueeze(2)      # (B, N_pred, N_tgt)
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa,
                        0.5 * delta ** 2,
                        kappa * (abs_delta - 0.5 * kappa))
    rho = (tau.view(1, n, 1) - (delta < 0).float()).abs() * huber / kappa
    return rho.mean(dim=2).sum(dim=1).mean()


def soft_update(*, target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.mul_(1.0 - tau).add_(p_s, alpha=tau)
