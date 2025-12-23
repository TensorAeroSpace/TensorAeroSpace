"""NARX neural network model.

This module defines a NARX (Nonlinear AutoRegressive with eXogenous inputs)
neural network model used for time-series prediction in some agents.
"""

import torch
import torch.nn as nn


class NARX(nn.Module):
    """NARX (Nonlinear AutoRegressive with eXogenous inputs) MLP model.

    This is a simple fully-connected NARX-style network for time-series
    prediction that uses the current input and the previous output.

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden layer size.
        output_size (int): Output feature dimension.

    Attributes:
        hidden_size (int): Hidden layer size.
        input_layer (nn.Linear): Linear layer applied to concatenated input and
            last output.
        output_layer (nn.Linear): Output projection layer.
        criterion (nn.MSELoss): Mean squared error loss.
        optimizer (torch.optim.Adam): Adam optimizer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize NARX MLP.

        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden layer width.
            output_size: Output feature dimension.
        """
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + output_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input_tensor, last_output):
        """Compute one forward step.

        Args:
            input_tensor (Tensor): Current input tensor.
            last_output (Tensor): Previous output tensor.

        Returns:
            Tensor: Model output tensor.
        """
        combined = torch.cat((input_tensor, last_output), 0)
        hidden = torch.tanh(self.input_layer(combined))
        output = self.output_layer(hidden)
        return output

    def train(self, predcit_tensor, target_tensor):
        """Perform one gradient update step.

        Note:
            This method name shadows ``torch.nn.Module.train``.

        Args:
            predcit_tensor (Tensor): Predicted tensor.
            target_tensor (Tensor): Target tensor.

        Returns:
            float: Loss value after one update step.
        """
        loss = self.criterion(predcit_tensor, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
