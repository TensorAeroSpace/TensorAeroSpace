from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn

class NARX(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, state_lags, control_lags):
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
