"""Actor and GDHP critic networks for IM-GDHP."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _build_mlp(
    in_features: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_features
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    return nn.Sequential(*layers)


class GDHPActor(nn.Module):
    """Deterministic policy network for IM-GDHP.

    Maps an augmented observation ``[y; y_ref; e_track]`` to a bounded
    control command ``u ∈ [-u_max, u_max]``. Bounding is enforced by a
    ``tanh`` output head scaled by ``u_max`` — this matches the
    "symmetrical sigmoid activation in the output layer of the actor"
    trick from the Sun/van Kampen incremental ADP papers.

    Args:
        in_features: Size of the augmented observation vector.
        n_u: Number of control channels.
        hidden_sizes: Sizes of the hidden layers.
        u_max: Per-channel absolute bound on the control command.
        activation: Hidden-layer activation factory.
    """

    def __init__(
        self,
        in_features: int,
        n_u: int,
        hidden_sizes: Sequence[int] = (32, 32),
        u_max: float = 1.0,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.n_u = int(n_u)
        self.u_max = float(u_max)
        self.backbone = _build_mlp(self.in_features, hidden_sizes, activation)
        last = hidden_sizes[-1] if hidden_sizes else self.in_features
        self.head = nn.Linear(last, self.n_u)

        # Small init on the output layer keeps initial actions near zero so
        # the agent starts close to the trim point instead of saturating.
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute ``u = u_max · tanh(head(backbone(obs)))``."""
        h = self.backbone(obs)
        raw = self.head(h)
        return self.u_max * torch.tanh(raw)


class GDHPCritic(nn.Module):
    """Dual-head critic for Global Dual Heuristic Programming.

    Outputs both the scalar cost-to-go ``J(o)`` and its vector derivative
    ``λ(o) = ∂J/∂y`` (same dimensionality as the observed state ``y``).
    The two heads share a common backbone so that the J-regression and
    the λ-regression reinforce each other during training.

    Args:
        in_features: Size of the augmented observation vector ``o``
            (typically ``n_y + n_ref + n_y`` when the tracking error is
            concatenated).
        n_y: Size of the observed state vector ``y``. Defines the number
            of outputs of the λ head.
        hidden_sizes: Sizes of the shared hidden layers.
        activation: Hidden-layer activation factory.
    """

    def __init__(
        self,
        in_features: int,
        n_y: int,
        hidden_sizes: Sequence[int] = (32, 32),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.n_y = int(n_y)
        self.backbone = _build_mlp(self.in_features, hidden_sizes, activation)
        last = hidden_sizes[-1] if hidden_sizes else self.in_features
        self.j_head = nn.Linear(last, 1)
        self.lambda_head = nn.Linear(last, self.n_y)

        nn.init.uniform_(self.j_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.j_head.bias)
        nn.init.uniform_(self.lambda_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.lambda_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(J, λ)`` for a batch of augmented observations."""
        h = self.backbone(obs)
        j = self.j_head(h)
        lam = self.lambda_head(h)
        return j, lam
