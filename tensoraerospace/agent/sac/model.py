"""Neural network models for Soft Actor-Critic (SAC).

This module defines policy and Q-network architectures used by the SAC agent.
"""

from typing import Tuple, Union

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m: torch.nn.Module) -> None:
    """Initialize policy weights.

    Applies Xavier initialization for weights and constant initialization for biases
    of linear layers.

    Args:
        m (nn.Module): Neural network module to initialize.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """Neural network for value function approximation.

    Args:
        num_inputs (int): Number of input features.
        hidden_dim (int): Hidden layers dimension.

    Attributes:
        linear1 (nn.Linear): First linear layer.
        linear2 (nn.Linear): Second linear layer.
        linear3 (nn.Linear): Third linear layer.
    """

    def __init__(self, num_inputs: int, hidden_dim: int):
        """Initialize ValueNetwork.

        Args:
            num_inputs (int): Number of input features.
            hidden_dim (int): Hidden layers dimension.
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of neural network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output value tensor.

        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    """Neural network for Q function evaluation.

    Args:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Hidden layers dimension.

    Attributes:
        linear1 (nn.Linear): First linear layer for Q1.
        linear2 (nn.Linear): Second linear layer for Q1.
        linear3 (nn.Linear): Third linear layer for Q1.
        linear4 (nn.Linear): First linear layer for Q2.
        linear5 (nn.Linear): Second linear layer for Q2.
        linear6 (nn.Linear): Third linear layer for Q2.

    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """Initialize twin Q-network architecture.

        Args:
            num_inputs: State dimension.
            num_actions: Action dimension.
            hidden_dim: Hidden layer width.
        """
        super(QNetwork, self).__init__()

        # Q1 арха
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 арха
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Q-value estimation.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(Q1, Q2)`` tensors.
        """
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    """Gaussian policy used by SAC.

    Args:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Hidden layer dimension.
        action_space (Optional[gym.Space]): Action space. Defaults to None.

    Attributes:
        linear1 (nn.Linear): First linear layer.
        linear2 (nn.Linear): Second linear layer.
        mean_linear (nn.Linear): Linear layer for mean value.
        log_std_linear (nn.Linear): Linear layer for log standard deviation.
        action_scale (torch.Tensor): Action scale.
        action_bias (torch.Tensor): Action bias.

    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_dim: int,
        action_space: gym.Space | None = None,
    ):
        """Initialize Gaussian policy network and action scaling.

        Args:
            num_inputs: State dimension.
            num_actions: Action dimension.
            hidden_dim: Hidden layer width.
            action_space: Optional gym space to derive scaling/bias.
        """
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log standard deviation for a batch of states."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action using the reparameterization trick.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(action, log_prob, mean)``.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # для трюка репараметризации (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        #  Применение ограничения на действия
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device: Union[str, torch.device]) -> "GaussianPolicy":
        """Move internal tensors to the given device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """Deterministic policy used by SAC (e.g., for evaluation).

    Args:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Hidden layer dimension.
        action_space (Optional[gym.Space]): Action space used to scale/bias actions.

    Attributes:
        linear1 (nn.Linear): First linear layer.
        linear2 (nn.Linear): Second linear layer.
        mean (nn.Linear): Linear layer producing the mean action.
        noise (torch.Tensor): Noise tensor for exploration.
        action_scale (torch.Tensor | float): Action scaling factor.
        action_bias (torch.Tensor | float): Action bias.
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_dim: int,
        action_space: gym.Space | None = None,
    ):
        """Initialize deterministic policy network for SAC evaluation.

        Args:
            num_inputs: State dimension.
            num_actions: Action dimension.
            hidden_dim: Hidden layer width.
            action_space: Optional gym space to derive scaling/bias.
        """
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # Масштабирование действий
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the deterministic action mean for a batch of states."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action by adding bounded Gaussian noise to the mean.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(action, log_prob, mean)``.
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device: Union[str, torch.device]) -> "DeterministicPolicy":
        """Move internal tensors to the given device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
