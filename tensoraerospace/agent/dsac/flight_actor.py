"""Actor network copied from dsac-flight (NormalPolicyNet)."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Independent, Normal

from .flight_mlp import make_mlp

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class NormalPolicyNet(nn.Module):
    """Outputs an Independent Normal distribution for continuous actions."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        n_hidden_layers: int,
        n_hidden_units: int,
    ) -> None:
        super().__init__()
        self.shared_net = make_mlp(
            num_in=int(obs_dim),
            num_out=int(n_hidden_units),
            n_hidden_layers=int(n_hidden_layers),
            n_hidden_units=int(n_hidden_units),
            final_activation=nn.ReLU(),
        )
        self.mu_layer = nn.Linear(int(n_hidden_units), int(action_dim))
        self.log_std_layer = nn.Linear(int(n_hidden_units), int(action_dim))

    def forward(self, states: torch.Tensor) -> Independent:
        out = self.shared_net(states)
        means = self.mu_layer(out)
        log_stds = self.log_std_layer(out)
        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))
        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

    def get_mean(self, states: torch.Tensor) -> torch.Tensor:
        out = self.shared_net(states)
        return self.mu_layer(out)

    def get_std(self, states: torch.Tensor) -> torch.Tensor:
        out = self.shared_net(states)
        log_std = self.log_std_layer(out)
        return torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
