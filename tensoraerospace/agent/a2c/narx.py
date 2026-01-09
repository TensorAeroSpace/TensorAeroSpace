"""A2C training utilities with NARX-style features.

This module provides helper classes and functions used to train A2C agents with
NARX (Nonlinear AutoRegressive with eXogenous inputs) representations.
"""

from collections.abc import Sequence
from typing import Iterable

import numpy as np
import torch
from gymnasium import Env
from torch import nn
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard optional at runtime

    class SummaryWriter:  # type: ignore
        """Fallback SummaryWriter when tensorboard is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_histogram(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass


from ..metrics import create_metric_writer


def clip_grad_norm_(module: torch.optim.Optimizer, max_grad_norm: float) -> None:
    """Clip gradients to prevent exploding gradients.

    Args:
        module (torch.optim.Optimizer): Optimizer whose parameter gradients will
            be clipped.
        max_grad_norm (float): Maximum gradient norm.
    """
    nn.utils.clip_grad_norm_(
        [p for g in module.param_groups for p in g["params"]], max_grad_norm
    )


def mish(input: torch.Tensor) -> torch.Tensor:
    """Apply the Mish activation function."""
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """PyTorch module implementing the Mish activation."""

    def __init__(self):
        """Initialize Mish activation module."""
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return mish(input)


def t(x, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert input to a float PyTorch tensor.

    Args:
        x: Array-like input.
        device: Target device. If None, tensor stays on default device (CPU).
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    out = torch.from_numpy(x).float()
    if device is not None:
        out = out.to(device)
    return out


class Actor(nn.Module):
    """Actor network for an actor-critic algorithm.

    Args:
        state_dim (int): State dimension.
        n_actions (int): Action dimension.
        activation (torch.nn.Module): Activation class (e.g., ``nn.Tanh``).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        activation: type[nn.Module] = nn.Tanh,
    ):
        """Initialize the actor network."""
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions),
        )

        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X: torch.Tensor) -> torch.distributions.Normal:
        """Compute an action distribution for a batch of states."""
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)

        return torch.distributions.Normal(means, stds)


class Critic(nn.Module):
    """Critic network for an actor-critic algorithm.

    Args:
        state_dim (int): State dimension.
        activation (torch.nn.Module): Activation class (e.g., ``nn.Tanh``).
    """

    def __init__(self, state_dim: int, activation: type[nn.Module] = nn.Tanh):
        """Build critic network layers.

        Args:
            state_dim: State dimension.
            activation: Activation module class.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate state value for input features."""
        return self.model(X)


def discounted_rewards(
    rewards: Sequence[float], dones: Sequence[float], gamma: float
) -> list[float]:
    """Compute discounted returns for a sequence of rewards."""
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory_narx(
    memory: Sequence[tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]],
    gamma: float = 0.99,
    discount_rewards: bool = True,
    device: torch.device | str | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert collected transitions into tensors suitable for training.

    The function also builds an augmented critic input that concatenates the
    current state with a lagged state (previous step) to mimic a NARX-style
    representation.

    Args:
        memory (list[tuple]): Tuples ``(action, reward, state, next_state, done)``.
        gamma (float): Discount factor. Defaults to ``0.99``.
        discount_rewards (bool): If True, uses discounted returns. Defaults to True.

    Returns:
        tuple: ``(actions, rewards, states, next_states, dones, critic_states)``.
    """
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    critic_states = []  # Инициализация для хранения состояний и предыдущих действий

    # Используем None или 0 как заполнитель для предыдущего действия первого состояния
    prev_state = np.zeros(memory[0][2].shape)
    prev_next_state = np.zeros(memory[0][2].shape)
    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(np.concatenate((next_state.flatten(), prev_next_state)))
        dones.append(done)
        # Добавляем текущее состояние и предыдущее действие в hist_values
        critic_states.append(np.concatenate((state.flatten(), prev_state)))
        prev_state = (
            state.flatten()
        )  # Обновляем предыдущее действие для следующей итерации
        prev_next_state = next_state.flatten()  #
    if discount_rewards:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions, device=device).view(-1, 1)
    states = t(states, device=device)
    next_states = t(next_states, device=device)
    rewards = t(rewards, device=device).view(-1, 1)
    dones = t(dones, device=device).view(-1, 1)
    critic_states = t(critic_states, device=device)  # Преобразование списка в тензор

    return actions, rewards, states, next_states, dones, critic_states


class A2CLearner:
    """Learner implementing Advantage Actor-Critic (A2C) updates."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        gamma: float = 0.9,
        entropy_beta: float = 0.01,
        actor_lr: float = 4e-4,
        critic_lr: float = 4e-3,
        max_grad_norm: float = 0.5,
        device: torch.device | str | None = None,
    ):
        """Initialize learner with optimizers and hyperparameters.

        Args:
            actor: Policy network.
            critic: Value network.
            gamma: Discount factor.
            entropy_beta: Entropy regularization weight.
            actor_lr: Learning rate for actor.
            critic_lr: Learning rate for critic.
            max_grad_norm: Gradient clipping norm.
        """
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.device = (
            torch.device(device)
            if device is not None
            else next(self.actor.parameters()).device
        )
        # Ensure networks are on the same device as the learner
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.writer = create_metric_writer()

    def learn(
        self,
        memory: Sequence[tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]],
        steps: int,
        discount_rewards: bool = True,
    ) -> None:
        """Update actor/critic using a batch of collected transitions.

        Args:
            memory (list): Collected transitions.
            steps (int): Global step index used for logging.
            discount_rewards (bool): If True, uses discounted returns as TD target.
        """
        (
            actions,
            rewards,
            states,
            next_states,
            dones,
            critic_states,
        ) = process_memory_narx(
            memory, self.gamma, discount_rewards, device=self.device
        )

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        value = self.critic(critic_states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (
            -logs_probs * advantage.detach()
        ).mean() - entropy * self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()

        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        actor_grads = [
            p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None
        ]
        if actor_grads:
            self.writer.add_histogram(
                "gradients/actor",
                torch.cat(actor_grads).detach().cpu(),
                global_step=steps,
            )
        self.writer.add_histogram(
            "parameters/actor",
            torch.cat([p.data.view(-1) for p in self.actor.parameters()])
            .detach()
            .cpu(),
            global_step=steps,
        )
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        critic_grads = [
            p.grad.view(-1) for p in self.critic.parameters() if p.grad is not None
        ]
        if critic_grads:
            self.writer.add_histogram(
                "gradients/critic",
                torch.cat(critic_grads).detach().cpu(),
                global_step=steps,
            )
        self.writer.add_histogram(
            "parameters/critic",
            torch.cat([p.data.view(-1) for p in self.critic.parameters()])
            .detach()
            .cpu(),
            global_step=steps,
        )
        self.critic_optim.step()

        # reports
        self.writer.add_scalar(
            "losses/log_probs", -logs_probs.mean(), global_step=steps
        )
        self.writer.add_scalar("losses/entropy", entropy, global_step=steps)
        self.writer.add_scalar(
            "losses/entropy_beta", self.entropy_beta, global_step=steps
        )
        self.writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        self.writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        self.writer.add_scalar("losses/critic", critic_loss, global_step=steps)


class Runner:
    """Environment interaction loop used to collect training data."""

    def __init__(self, env: Env, actor: nn.Module, writer: SummaryWriter):
        """Create runner for data collection.

        Args:
            env: Environment instance.
            actor: Policy network used to select actions.
            writer: TensorBoard writer for logging rewards.
        """
        self.env = env
        self.actor = actor
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        # Initialize previous action as zeros; adjust the size based on your action space
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.writer = writer
        self.device = next(self.actor.parameters()).device

    @staticmethod
    def _flatten_observation(observation: np.ndarray | Iterable[float]) -> np.ndarray:
        """Flatten environment observations to shape ``(n,)``."""
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        """Reset environment and episode state."""
        self.episode_reward = 0
        self.done = False
        self.state, info = self.env.reset()
        self.state = self._flatten_observation(self.state)
        # Reset previous action at the start of each episode
        self.prev_action = np.zeros(self.env.action_space.shape)

    def run(
        self,
        max_steps: int,
        memory: (
            list[tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]] | None
        ) = None,
    ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]]:
        """Run the environment for a fixed number of steps and collect transitions.

        Args:
            max_steps (int): Number of environment steps to execute.
            memory (list, optional): Existing list to append transitions to.

        Returns:
            list: Collected transitions.
        """
        if not memory:
            memory = []

        for i in range(max_steps):
            if self.done:
                self.reset()

            state_t = torch.as_tensor(
                self.state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dists = self.actor(state_t)
            actions = dists.sample().detach().cpu().numpy()
            actions_clipped = np.clip(
                actions, self.env.action_space.low, self.env.action_space.high
            )

            next_state, reward, terminated, truncated, info = self.env.step(
                actions_clipped[0]
            )
            self.done = terminated or truncated
            next_state = self._flatten_observation(next_state)

            # Here, instead of just the state, we store the state concatenated with the previous action
            memory.append((actions, reward, self.state, next_state, self.done))

            self.prev_action = actions_clipped[0]  # Update the previous action
            self.state = next_state
            self.steps += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)
                # Assuming writer is defined and configured globally
                self.writer.add_scalar(
                    "episode_reward", self.episode_reward, global_step=self.steps
                )

        return memory
