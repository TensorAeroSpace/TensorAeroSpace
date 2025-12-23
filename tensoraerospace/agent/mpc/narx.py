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
        self, input_size, hidden_size, output_size, num_layers, state_lags, control_lags
    ):
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
