"""Distributional critics (IQN-based) for DSAC."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, layers: List[int], layer_norm: bool) -> nn.Sequential:
    seq = []
    last = input_dim
    for h in layers:
        seq.append(nn.Linear(last, h))
        if layer_norm:
            seq.append(nn.LayerNorm(h))
        seq.append(nn.ReLU())
        last = h
    return nn.Sequential(*seq)


class IQNCritic(nn.Module):
    """Implicit Quantile Network critic head.

    Mirrors HybridRL-FlightControl: psi(state, action), phi(quantile), merge -> quantile values.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layers: List[int],
        embedding_dim: int = 64,
        layer_norm: bool = True,
    ):
        super().__init__()
        if len(hidden_layers) < 2:
            raise ValueError("hidden_layers must have at least two layers")
        self.embedding_dim = int(embedding_dim)
        self.constant_vector = torch.from_numpy(
            np.arange(1, 1 + self.embedding_dim, dtype=np.float32)
        )  # (E,)

        # psi: state-action feature extractor (all but last hidden layer)
        self.psi = _build_mlp(obs_dim + act_dim, hidden_layers[:-1], layer_norm)

        # phi: quantile embedding -> last hidden layer size -1
        self.phi = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_layers[-2]),
            nn.LayerNorm(hidden_layers[-2]) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )

        # merge: combines psi * phi -> output 1
        self.merge = nn.Sequential(
            nn.Linear(hidden_layers[-2], hidden_layers[-1]),
            nn.LayerNorm(hidden_layers[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, quantile: torch.Tensor
    ) -> torch.Tensor:
        # state: (B,S), action: (B,A), quantile: (B,Q,1)
        state = state.view(state.size(0), -1)
        action = action.view(action.size(0), -1)
        quantile = quantile.view(quantile.size(0), quantile.size(1), 1)

        B, Q, _ = quantile.shape
        H = self.embedding_dim

        # psi
        sa = torch.cat([state, action], dim=1)  # (B,S+A)
        psi = self.psi(sa)  # (B, hidden[-2])

        # phi
        const = self.constant_vector.to(quantile.device)  # (E,)
        cos_tau = torch.cos(quantile * const * torch.pi)  # (B,Q,E)
        phi = self.phi(cos_tau)  # (B,Q,H')

        # merge
        psi_exp = psi.view(B, 1, -1)
        psi_phi = phi * psi_exp  # (B,Q,H')
        out = self.merge(psi_phi)  # (B,Q,1)
        return out.squeeze(-1)  # (B,Q)


class QuantileTwin(nn.Module):
    """Twin IQN critics."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layers: List[int],
        embedding_dim: int = 64,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.q1 = IQNCritic(obs_dim, act_dim, hidden_layers, embedding_dim, layer_norm)
        self.q2 = IQNCritic(obs_dim, act_dim, hidden_layers, embedding_dim, layer_norm)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, quantile: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action, quantile), self.q2(state, action, quantile)
