import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class NARX(nn.Module):
    """
    Нейронная сеть NARX (Nonlinear AutoRegressive with eXogenous inputs) для моделирования динамических систем.

    NARX сеть использует предыдущие значения состояний и управляющих воздействий для предсказания
    будущих состояний системы.

    Args:
        input_size (int): Размер входного слоя (общий размер лагированных состояний и управлений).
        hidden_size (int): Размер скрытых слоев.
        output_size (int): Размер выходного слоя (размерность предсказываемого состояния).
        num_layers (int): Количество скрытых слоев.
        state_lags (int): Количество лагов для состояний.
        control_lags (int): Количество лагов для управляющих воздействий.
    """

    def __init__(
        self, input_size, hidden_size, output_size, num_layers, state_lags, control_lags
    ):
        """
        Инициализация NARX нейронной сети.

        Args:
            input_size (int): Размер входного слоя.
            hidden_size (int): Размер скрытых слоев.
            output_size (int): Размер выходного слоя.
            num_layers (int): Количество скрытых слоев.
            state_lags (int): Количество лагов для состояний.
            control_lags (int): Количество лагов для управляющих воздействий.
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

    def forward(self, state, control):
        """
        Прямое распространение через NARX сеть.

        Args:
            state (torch.Tensor): Тензор лагированных состояний.
            control (torch.Tensor): Тензор лагированных управляющих воздействий.

        Returns:
            torch.Tensor: Предсказанное следующее состояние системы.
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
