"""Squashed-Gaussian actor with reparameterisation (Haarnoja 2018 SAC style)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ActorConfig:
    n_state: int
    n_action: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class GaussianActor(nn.Module):
    """π(a|s) = tanh(N(μ_θ(s), σ_θ(s)))."""

    def __init__(self, cfg: ActorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = int(cfg.n_state)
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head_mean = nn.Linear(in_dim, int(cfg.n_action))
        self.head_log_std = nn.Linear(in_dim, int(cfg.n_action))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(s)
        mean = self.head_mean(h)
        log_std = self.head_log_std(h).clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std

    def rsample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(s)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        z = mean + std * eps
        a = torch.tanh(z)

        log_prob_z = (
            -0.5 * ((z - mean) / std) ** 2 - log_std - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_prob = log_prob_z - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, log_prob

    def deterministic(self, s: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(s)
        return torch.tanh(mean)
