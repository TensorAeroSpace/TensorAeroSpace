"""NARX-style dynamics models for MPC.

This module provides utilities for using NARX (autoregressive) representations
within MPC components.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class NARX(nn.Module):
    """NARX neural network for learning system dynamics.

    The model uses lagged (historical) state and control inputs to predict the
    next state.

    Args:
        input_size: Input size (concatenated lagged states and controls).
        hidden_size: Hidden layer size.
        output_size: Output size (predicted state dimension).
        num_layers: Number of hidden layers.
        state_lags: Number of state lags used as input.
        control_lags: Number of control lags used as input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        state_lags: int,
        control_lags: int,
    ) -> None:
        """Initialize the NARX network.

        Args:
            input_size: Input size.
            hidden_size: Hidden layer size.
            output_size: Output size.
            num_layers: Number of hidden layers.
            state_lags: Number of state lags.
            control_lags: Number of control lags.
        """
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_lags = state_lags
        self.control_lags = control_lags

        # Define the input layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Define the hidden layers using ModuleList
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Define the output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            state: Lagged state tensor.
            control: Lagged control tensor.

        Returns:
            torch.Tensor: Predicted next state.
        """
        # Concatenate lagged states and controls
        x = torch.cat((state, control), dim=1)

        # Pass through the first fully connected layer
        x = self.activation(self.fc1(x))

        # Pass through the hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Pass through the output layer
        x = self.fc_out(x)

        return x


class NARXDynamicsModel(nn.Module):
    """MPCAgent-compatible NARX dynamics model.

    `MPCAgent` expects learned dynamics modules with signature:

        y = model(xu), where xu = concat([x, u])

    This wrapper builds an internal :class:`~tensoraerospace.agent.mpc.narx.NARX`
    and provides the required `forward(xu)` interface.

    Notes:
        - For the current MPC pipeline, this is typically used with
          ``state_lags=1`` and ``control_lags=1`` (one-step model).
        - If you want true NARX with lags > 1, you must provide an augmented
          state/action history vector as input to MPC (not handled implicitly).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        state_lags: int = 1,
        control_lags: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state_lags = int(state_lags)
        self.control_lags = int(control_lags)

        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim/action_dim must be positive")
        if self.state_lags <= 0 or self.control_lags <= 0:
            raise ValueError("state_lags/control_lags must be positive")

        input_size = (
            self.state_dim * self.state_lags + self.action_dim * self.control_lags
        )
        self.net = NARX(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            output_size=int(self.state_dim),
            num_layers=int(num_layers),
            state_lags=self.state_lags,
            control_lags=self.control_lags,
        )

    def forward(self, xu: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass.

        Args:
            xu: Concatenated input of shape (B, state_dim*state_lags + action_dim*control_lags).

        Returns:
            Predicted next-state (or delta-state) of shape (B, state_dim).
        """
        if xu.ndim != 2:
            xu = xu.view(xu.shape[0], -1)
        s_end = int(self.state_dim * self.state_lags)
        state = xu[:, :s_end]
        control = xu[:, s_end:]
        return self.net(state, control)
