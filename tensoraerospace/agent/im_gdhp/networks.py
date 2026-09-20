"""Actor and scalar, analytically differentiated GDHP critic.

Sun & van Kampen (2021), Eqs. (51), (52) and (64). Autograd computes
both the input derivative and its mixed derivatives with network weights.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def _build_mlp(in_features, hidden_sizes, activation, *, bias=True):
    layers = []
    last = in_features
    for size in hidden_sizes:
        layers.extend([nn.Linear(last, size, bias=bias), activation()])
        last = size
    return nn.Sequential(*layers)


class GDHPActor(nn.Module):
    """Bounded policy for a tracking-error vector, Eq. (64).

    The optional input scale changes network coordinates, not physical units
    of actions. The constant ``bias_input`` (default 0.01) permits trim control at zero error,
    as in Eq. (64) and Section 5.2; the output layer has no extra bias.
    """

    input_scale: torch.Tensor

    def __init__(
        self,
        in_features: int,
        n_u: int,
        hidden_sizes: Sequence[int] = (32, 32),
        u_max: float = 1.0,
        activation: type[nn.Module] = nn.Tanh,
        input_scale=None,
        bias_input: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.n_u = int(n_u)
        self.u_max = float(u_max)
        self.bias_input = float(bias_input)
        if not math.isfinite(self.bias_input) or self.bias_input <= 0:
            raise ValueError("bias_input must be finite and positive")
        self.register_buffer(
            "input_scale",
            (
                torch.ones(in_features)
                if input_scale is None
                else torch.as_tensor(input_scale, dtype=torch.float32)
            ),
        )
        self.backbone = _build_mlp(
            in_features + 1, hidden_sizes, activation, bias=False
        )
        self.head = nn.Linear(
            hidden_sizes[-1] if hidden_sizes else in_features + 1, n_u, bias=False
        )
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.1, 0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the physical bounded control command."""
        scaled = obs * self.input_scale
        features = torch.cat(
            (scaled, torch.full_like(scaled[..., :1], self.bias_input)), dim=-1
        )
        return self.u_max * torch.tanh(self.head(self.backbone(features)))


class GDHPCritic(nn.Module):
    """Return J and its exact input gradient; no independent costate head.

    The bias-free tanh critic satisfies J(0)=0, as in Eqs. (51)--(52).
    ``n_y`` selects the first input coordinates when used independently;
    an IMGDHPAgent differentiates all tracked-error coordinates. Input
    scaling is inside the graph, so lambda retains physical error units.
    """

    input_scale: torch.Tensor

    def __init__(
        self,
        in_features: int,
        n_y: int,
        hidden_sizes: Sequence[int] = (32, 32),
        activation: type[nn.Module] = nn.Tanh,
        input_scale=None,
    ) -> None:
        super().__init__()
        self.in_features, self.n_y = int(in_features), int(n_y)
        if not 0 < self.n_y <= self.in_features:
            raise ValueError("n_y must be in 1..in_features")
        self.register_buffer(
            "input_scale",
            (
                torch.ones(in_features)
                if input_scale is None
                else torch.as_tensor(input_scale, dtype=torch.float32)
            ),
        )
        self.backbone = _build_mlp(in_features, hidden_sizes, activation, bias=False)
        self.j_head = nn.Linear(
            hidden_sizes[-1] if hidden_sizes else in_features, 1, bias=False
        )
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.1, 0.1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiate J even during inference; retain mixed gradients in training."""
        training_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            x = obs if obs.requires_grad else obs.detach().requires_grad_(True)
            j = self.j_head(self.backbone(x * self.input_scale))
            lam = torch.autograd.grad(
                j.sum(), x, create_graph=training_graph, retain_graph=training_graph
            )[0][..., : self.n_y]
        return (j, lam) if training_graph else (j.detach(), lam.detach())
