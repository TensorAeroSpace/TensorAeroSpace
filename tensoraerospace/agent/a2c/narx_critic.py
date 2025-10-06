"""NARX-based critic for A2C.

This module provides a simple NARX (Nonlinear AutoRegressive with exogenous
inputs) critic network and a helper to build fixed-size history features from
sequences of observations and actions.

Usage pattern:
    - Build features with ``build_narx_features`` from (states, actions).
    - Feed features to ``NARXCritic`` to get state-value estimates.

We keep the design explicit: the critic expects pre-built feature tensors, so
integration with an existing agent is straightforward and does not require the
critic to carry recurrent state.
"""

from typing import List, Sequence, Union

import numpy as np
import torch
from torch import nn


def _to_float_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert input to float32 tensor.

    Always returns a CPU tensor; the caller should move it to the device
    used by the model/optimizer.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.from_numpy(np.asarray(x)).to(dtype=torch.float32)


def build_narx_features(
    states: Union[np.ndarray, torch.Tensor],
    actions: Union[np.ndarray, torch.Tensor],
    history_length: int,
) -> torch.Tensor:
    """Build NARX features for a single sequence.

    Features at time t are the concatenation of:
      [x_t, x_{t-1}, ..., x_{t-h+1}, u_{t-1}, u_{t-2}, ..., u_{t-h}]

    The first steps are zero-padded where history is missing.

    Args:
        states: Array/Tensor of shape (T, obs_dim).
        actions: Array/Tensor of shape (T, act_dim).
        history_length: Number of past steps to include (h >= 1).

    Returns:
        Tensor of shape (T, h*obs_dim + h*act_dim).
    """
    assert history_length >= 1, "history_length must be >= 1"

    x = _to_float_tensor(states)
    u = _to_float_tensor(actions)

    if x.dim() != 2 or u.dim() != 2:
        raise ValueError("states and actions must be 2D tensors: (T, dim)")

    if x.shape[0] != u.shape[0]:
        raise ValueError("states and actions must have the same T length")

    T = x.shape[0]
    obs_dim = x.shape[1]
    act_dim = u.shape[1]

    # Zero padding for missing history at the sequence start (on same device)
    zeros_obs = torch.zeros(
        (history_length - 1, obs_dim), dtype=x.dtype, device=x.device
    )
    zeros_act = torch.zeros((history_length, act_dim), dtype=u.dtype, device=u.device)

    # Build stacked observations [x_t, x_{t-1}, ..., x_{t-h+1}]
    # Pad at the beginning so missing history is zero; use only PAST values
    obs_stack = []
    x_padded = torch.cat([zeros_obs, x], dim=0)
    for k in range(history_length):
        # For k=0 take current x_t, for k=1 take x_{t-1}, etc.
        start = history_length - 1 - k
        obs_stack.append(x_padded[start : start + T])
    obs_feats = torch.cat(obs_stack, dim=1)

    # Build stacked actions [u_{t-1}, u_{t-2}, ..., u_{t-h}]
    act_stack = []
    u_padded = torch.cat([zeros_act, u], dim=0)
    for k in range(1, history_length + 1):
        act_stack.append(u_padded[history_length - k : history_length - k + T])
    act_feats = torch.cat(act_stack, dim=1)

    return torch.cat([obs_feats, act_feats], dim=1)


class NARXCritic(nn.Module):
    """A simple MLP-based NARX critic.

    The network maps fixed-size history features produced by
    ``build_narx_features`` to a scalar value estimate V(s_t).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history_length: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.history_length = history_length

        input_dim = history_length * (observation_dim + action_dim)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hs))
            layers.append(activation())
            prev_dim = hs
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value estimates.

        Args:
            features: Tensor of shape (batch, feature_dim) where
                feature_dim == history_length * (obs_dim + act_dim).

        Returns:
            Tensor of shape (batch, 1) with value estimates.
        """
        out: torch.Tensor = self.model(features)
        return out
