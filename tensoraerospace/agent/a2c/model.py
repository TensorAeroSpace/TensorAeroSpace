"""Advantage Actor-Critic (A2C) algorithm implementation module.

This module contains the A2C algorithm implementation for reinforcement learning,
including actor and critic neural networks, memory processing functions and the main
A2C agent class for aerospace system control.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from ..metrics import MetricWriter, create_metric_writer, schema
from .narx_critic import build_narx_features


def mish(input):
    """Mish activation function.

    Mish is a smooth, continuous activation function defined as:
    f(x) = x * tanh(softplus(x))

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Result of applying Mish activation function.
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """PyTorch module for Mish activation function.

    This class wraps the Mish activation function in a PyTorch module,
    allowing it to be used in neural networks.
    """

    def __init__(self):
        """Initialize Mish module."""
        super().__init__()

    def forward(self, input):
        """Forward pass through Mish activation function.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of applying Mish activation function.
        """
        return mish(input)


# Helper function to convert numpy arrays to tensors
def to_tensor(x, device="cpu", dtype=torch.float32):
    """Convert numpy array to PyTorch tensor on specified device.

    Args:
        x: Input data (numpy array or other type).
        device (str or torch.device): Device to place tensor on. Defaults to 'cpu'.
        dtype: Data type of tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: PyTorch tensor on specified device.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def t(x):
    """Convert input to a float32 torch tensor on CPU.

    This helper mirrors the behavior expected by tests and is equivalent to
    ``torch.from_numpy(np.array(x)).float()`` for array-like inputs.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(x).float()


class Actor(nn.Module):
    """Actor neural network for A2C algorithm.

    Actor generates policy - probability distribution of actions
    for each state. Uses normal distribution for
    continuous actions.

    Args:
        state_dim (int): State space dimension.
        n_actions (int): Number of actions.
        activation: Activation function for hidden layers. Defaults to nn.Tanh.

    Attributes:
        n_actions (int): Number of actions.
        model (nn.Sequential): Main neural network.
        logstds (nn.Parameter): Logarithms of standard deviations for actions.
    """

    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        """Initialize actor.

        Args:
            state_dim (int): State space dimension.
            n_actions (int): Number of actions.
            activation: Activation function for hidden layers.
        """
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions),
        )

        self.logstds = nn.Parameter(torch.full((n_actions,), -1.0))

    def forward(self, X):
        """Forward pass through actor network.

        Args:
            X (torch.Tensor): Input states.

        Returns:
            torch.distributions.Normal: Normal distribution of actions.
        """
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        return torch.distributions.Normal(means, stds)


class Critic(nn.Module):
    """Critic neural network for A2C algorithm.

    Critic evaluates state values, predicting expected
    cumulative reward from given state.

    Args:
        state_dim (int): State space dimension.
        activation: Activation function for hidden layers. Defaults to nn.Tanh.

    Attributes:
        model (nn.Sequential): Main neural network.
    """

    def __init__(self, state_dim, activation=nn.Tanh):
        """Initialize critic.

        Args:
            state_dim (int): State space dimension.
            activation: Activation function for hidden layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )

    def forward(self, X):
        """Forward pass through critic network.

        Args:
            X (torch.Tensor): Input states.

        Returns:
            torch.Tensor: State value estimates.
        """
        return self.model(X)


def discounted_rewards(rewards, dones, gamma):
    """Calculate discounted rewards for episode.

    Args:
        rewards (list): List of rewards for each step.
        dones (list): List of episode termination flags.
        gamma (float): Discount coefficient.

    Returns:
        list: List of discounted rewards.
    """
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


class RolloutTransition(tuple):
    """Five-item transition with termination and physical history metadata.

    The first item is the sampled Gaussian action. ``executed_action`` holds
    the clipped command used by the plant and NARX input history.
    """

    terminated: bool
    executed_action: np.ndarray
    previous_states: tuple[np.ndarray, ...]
    previous_actions: tuple[np.ndarray, ...]

    def __new__(
        cls,
        action,
        reward,
        state,
        next_state,
        done,
        *,
        terminated,
        executed_action,
        previous_states=(),
        previous_actions=(),
    ):
        obj = super().__new__(cls, (action, reward, state, next_state, done))
        obj.terminated = bool(terminated)
        obj.executed_action = np.array(executed_action, copy=True)
        obj.previous_states = tuple(np.array(x, copy=True) for x in previous_states)
        obj.previous_actions = tuple(np.array(x, copy=True) for x in previous_actions)
        return obj

    def __getnewargs_ex__(self):
        return tuple(self), {
            "terminated": self.terminated,
            "executed_action": self.executed_action,
            "previous_states": self.previous_states,
            "previous_actions": self.previous_actions,
        }


def process_memory(memory, gamma=0.99, discount_rewards=True, device="cpu"):
    """Process experience memory for training.

    Args:
        memory (list): List of tuples (action, reward, state,
            next_state, done).
        gamma (float): Discount coefficient. Defaults to 0.99.
        discount_rewards (bool): Whether to apply reward discounting.
            Defaults to True.
        device (str or torch.device): Device to place tensors on.
            Defaults to 'cpu'.

    Returns:
        tuple: Tuple of tensors (actions, rewards, states, next_states, dones).
    """
    if not memory:
        raise ValueError("memory must contain at least one transition")
    actions, states, next_states, rewards, dones = [], [], [], [], []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)

    if discount_rewards:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = to_tensor(actions, device=device)
    states = to_tensor(states, device=device)
    next_states = to_tensor(next_states, device=device)
    rewards = to_tensor(rewards, device=device).view(-1, 1)
    dones = to_tensor(dones, device=device).view(-1, 1)

    return actions, rewards, states, next_states, dones


def clip_grad_norm_(module, max_grad_norm):
    """Clip gradients by norm for training stabilization.

    Args:
        module: PyTorch optimizer.
        max_grad_norm (float): Maximum gradient norm.
    """
    nn.utils.clip_grad_norm_(
        [p for g in module.param_groups for p in g["params"]], max_grad_norm
    )


class A2C(BaseRLModel):
    """Implementation of Advantage Actor-Critic (A2C) algorithm.

    A2C is a reinforcement learning algorithm that uses
    actor for action selection and critic for state evaluation.
    Algorithm minimizes actor and critic losses simultaneously.

    Args:
        env: Training environment.
        actor: Actor neural network.
        critic: Critic neural network.
        gamma (float): Discount coefficient. Defaults to 0.99.
        entropy_beta (float): Entropy bonus coefficient. Defaults to 0.01.
        actor_lr (float): Actor learning rate. Defaults to 1e-4.
        critic_lr (float): Critic learning rate. Defaults to 3e-4.
        max_grad_norm (float): Maximum gradient norm. Defaults to 0.5.
        seed (int, optional): Seed for reproducible results.

    Attributes:
        env: Training environment.
        state: Current environment state.
        done (bool): Episode termination flag.
        steps (int): Total number of steps.
        episode_reward (float): Reward for current episode.
        episode_rewards (list): Episode reward history.
        actor: Actor neural network.
        critic: Critic neural network.
        gamma (float): Discount coefficient.
        entropy_beta (float): Entropy bonus coefficient.
        actor_optim: Actor optimizer.
        critic_optim: Critic optimizer.
        writer: TensorBoard writer for logging.
    """

    def __init__(
        self,
        env,
        actor,
        critic,
        gamma=0.99,
        entropy_beta=0.01,
        actor_lr=1e-4,
        critic_lr=3e-4,
        max_grad_norm=0.5,
        seed=None,
        device=None,
        log_dir=None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize A2C agent.

        Args:
            env: Training environment.
            actor: Actor neural network.
            critic: Critic neural network.
            gamma (float): Discount factor. Defaults to 0.99.
            entropy_beta (float): Entropy bonus coefficient. Defaults to 0.01.
            actor_lr (float): Actor learning rate. Defaults to 1e-4.
            critic_lr (float): Critic learning rate. Defaults to 3e-4.
            max_grad_norm (float): Maximum gradient norm for clipping.
                Defaults to 0.5.
            seed (int, optional): Random seed for reproducibility.
            device (str or torch.device, optional): Device to use
                ('cpu' or 'cuda'). If None, auto-selects CUDA if available.
            log_dir (str, optional): TensorBoard log directory. If None,
                the default ``runs/`` directory of ``SummaryWriter`` is used.
        """
        self.env = env
        self.state: Optional[np.ndarray] = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_rewards: list[float] = []
        self.update_count = 0

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set seed for reproducibility
        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

        # Move models to device
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_beta = entropy_beta
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        self.log_dir = log_dir
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags
        self.wandb_config = wandb_config
        self.writer: MetricWriter = create_metric_writer(
            tb_log_dir=log_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            algo="a2c",
        )

        print(f"A2C initialized on device: {self.device}")

    def _set_seed(self, seed):
        """Set random seeds for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set seed for environment if supported
        if hasattr(self.env, "seed"):
            try:
                self.env.seed(seed)
            except TypeError:
                # For Gymnasium envs
                pass

    def reset(self):
        """Reset agent and environment state for new episode."""
        self.episode_reward = 0
        self.done = False
        self.state, _ = self.env.reset()
        self.state = np.array(self.state, copy=True)
        self._history_states = []
        self._history_actions = []
        self.episode_length = 0

    def predict(self, state, deterministic=True):
        """Predict action for given state.

        Args:
            state: Environment state (numpy array or list).
            deterministic (bool): If True, returns mean of distribution.
                If False, samples from distribution. Defaults to True.

        Returns:
            numpy.ndarray: Action, clipped to action space bounds.

        Example:
            >>> state = env.reset()
            >>> action = agent.predict(state, deterministic=True)
            >>> next_state, reward, done, info = env.step(action)
        """
        self.actor.eval()

        with torch.no_grad():
            state_tensor = to_tensor(state, device=self.device)

            # Add batch dimension if needed
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Get action distribution
            dist = self.actor(state_tensor)

            # Select action
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            # Convert to numpy and remove batch dimension
            action = action.squeeze(0).cpu().numpy()

            # Clip to action space bounds
            action = np.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )

        self.actor.train()
        return action

    def set_eval_mode(self):
        """Set models to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self):
        """Set models to training mode."""
        self.actor.train()
        self.critic.train()

    def run_episode(self, max_steps):
        """Collect experience from environment interaction for a fixed number of steps.

        The method always collects exactly max_steps steps, automatically starting
        new episodes if previous ones ended. This ensures a constant batch size
        for stable A2C training.

        Args:
            max_steps (int): Number of steps to collect experience.

        Returns:
            list: List of tuples (action, reward, state, next_state, done)
                  representing experience from environment interaction.
        """
        memory = []

        for _ in range(max_steps):
            if self.done:
                self.reset()

            with torch.no_grad():
                state_tensor = to_tensor(self.state, device=self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                dist = self.actor(state_tensor)
                action = dist.sample().squeeze(0).cpu().numpy()
            actions_clipped = np.clip(
                action,
                self.env.action_space.low,
                self.env.action_space.high,
            )

            state_snapshot = np.array(self.state, copy=True)
            next_state, reward, terminated, truncated, info = self.env.step(
                actions_clipped
            )
            self.done = terminated or truncated

            final_state = info.get(
                "final_observation", info.get("terminal_observation")
            )
            target_state = (
                final_state if self.done and final_state is not None else next_state
            )
            memory.append(
                RolloutTransition(
                    action.copy(),
                    float(reward),
                    state_snapshot,
                    np.array(target_state, copy=True),
                    self.done,
                    terminated=terminated,
                    executed_action=actions_clipped,
                    previous_states=self._history_states,
                    previous_actions=self._history_actions,
                )
            )
            h = getattr(self, "history_length", 1)
            self._history_states = (
                (self._history_states + [state_snapshot])[-(h - 1) :] if h > 1 else []
            )
            self._history_actions = (self._history_actions + [actions_clipped.copy()])[
                -h:
            ]
            self.state = np.array(next_state, copy=True)
            self.steps += 1
            self.episode_length += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)
                self.writer.log_episode(
                    reward=float(self.episode_reward),
                    length=int(self.episode_length),
                    env_step=int(self.steps),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                self.episode_reward = 0
                self.episode_length = 0

        return memory

    def _critic_inputs(self, memory, states, next_states):
        """Return current/next value-network inputs for a rollout."""
        return states, next_states

    def _prepare_update(self, memory, discount_rewards):
        actions, rewards, states, next_states, dones = process_memory(
            memory, self.gamma, discount_rewards=False, device=self.device
        )
        features, next_features = self._critic_inputs(memory, states, next_states)
        with torch.no_grad():
            terminals = to_tensor(
                [getattr(item, "terminated", item[4]) for item in memory],
                device=self.device,
            ).view(-1, 1)
            next_values = self.critic(next_features) * (1 - terminals)
            targets = rewards + self.gamma * next_values
            if discount_rewards:
                running = next_values[-1]
                for i in reversed(range(len(memory))):
                    if bool(dones[i]):
                        running = next_values[i]
                    running = rewards[i] + self.gamma * running
                    targets[i] = running
        return actions, states, features, targets

    def learn(self, memory, steps, discount_rewards=True):
        """Train the agent based on collected experience.

        Performs one training step for actor and critic using the
        Advantage Actor-Critic algorithm.

        Args:
            memory (list): List of experience from environment interaction.
            steps (int): Current step number for logging.
            discount_rewards (bool): Whether to apply reward discounting.
                                   Defaults to True.
        """
        actions, states, features, td_target = self._prepare_update(
            memory, discount_rewards
        )

        # Critic learning FIRST
        value = self.critic(features)
        critic_loss = F.mse_loss(value, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()

        # Recalculate value with updated critic (no grad for advantage)
        with torch.no_grad():
            value_updated = self.critic(features)
            advantage = td_target - value_updated

            # Normalize advantage for stable learning (critical for A2C!)
            advantage_normalized = advantage
            if advantage.numel() > 1:
                advantage_normalized = (advantage - advantage.mean()) / (
                    advantage.std(unbiased=False) + 1e-8
                )

        # Actor learning with fresh advantage estimates
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        if logs_probs.dim() > 1:
            logs_probs = logs_probs.sum(dim=-1, keepdim=True)
        entropy = norm_dists.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1).mean()

        # Policy gradient with entropy bonus
        actor_loss = (
            -(logs_probs * advantage_normalized).mean() - self.entropy_beta * entropy
        )

        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.actor_optim.step()

        # Reporting (canonical TB metrics).
        # ``entropy`` and ``actor_loss`` still hold autograd graph references
        # at this point; detach before casting to plain floats so we never
        # accidentally extend the graph through the writer.
        self.writer.add_scalar(
            schema.LOSS_ENTROPY, float(entropy.detach()), env_step=steps
        )
        self.writer.add_scalar(
            schema.A2C.ENTROPY_BETA, float(self.entropy_beta), env_step=steps
        )
        self.writer.add_scalar(
            schema.LOSS_ACTOR, float(actor_loss.detach()), env_step=steps
        )
        self.writer.add_scalar(
            schema.LOSS_CRITIC, float(critic_loss.detach()), env_step=steps
        )

        # Advantage diagnostics
        self.writer.add_scalar(
            schema.A2C.ADVANTAGE_MEAN, float(advantage.mean()), env_step=steps
        )
        self.writer.add_scalar(
            schema.A2C.ADVANTAGE_STD,
            float(advantage.std(unbiased=False)),
            env_step=steps,
        )
        self.writer.add_scalar(
            schema.A2C.ADVANTAGE_NORMALIZED_MEAN,
            float(advantage_normalized.mean()),
            env_step=steps,
        )

        # Value and TD target metrics
        self.writer.add_scalar(
            schema.VALUE_MEAN, float(value_updated.mean()), env_step=steps
        )
        self.writer.add_scalar(
            schema.VALUE_TD_TARGET, float(td_target.mean()), env_step=steps
        )
        self.writer.add_scalar(
            schema.A2C.VALUE_BEFORE_UPDATE,
            float(value.detach().mean().item()),
            env_step=steps,
        )

        # Policy statistics
        self.writer.add_scalar(
            schema.POLICY_ACTION_STD,
            float(norm_dists.stddev.detach().mean()),
            env_step=steps,
        )

        # Training counters (mandatory minimum tier).
        self.update_count += 1
        self.writer.add_scalar(
            schema.TRAIN_UPDATES, int(self.update_count), env_step=steps
        )
        self.writer.add_scalar(
            schema.TRAIN_LR,
            float(self.actor_optim.param_groups[0]["lr"]),
            env_step=steps,
        )

    def train(
        self,
        num_episodes=None,
        *,
        max_steps=None,
        save_best: bool = False,
        verbose: bool = True,
        steps_on_memory=128,
        episodes=2000,
        episode_length=300,
        discount_rewards=True,
        log_freq=10,
        save_freq=None,
        save_path=None,
        **kwargs,
    ):
        """Train the A2C agent (unified interface with legacy kwargs).

        The method accepts both the canonical unified-API parameters
        (``num_episodes``, ``max_steps``, ``save_best``, ``save_path``,
        ``verbose``) and A2C's historical keyword-only options
        (``steps_on_memory``, ``episodes``, ``episode_length`` …) so
        existing notebooks continue to work unchanged.

        Args:
            num_episodes (int, optional): Unified-API alias for
                ``episodes``. When provided, it overrides the legacy
                ``episodes`` keyword.
            max_steps (int, optional): Unified-API alias for
                ``episode_length``. When provided, it overrides the
                legacy ``episode_length`` keyword.
            save_best (bool): Kept for API consistency – A2C always
                saves the best model if ``save_path`` is provided.
            verbose (bool): Reserved for symmetry with other agents.
            steps_on_memory (int): Number of steps to collect before
                learning. Defaults to 128.
            episodes (int): Total number of training episodes. Defaults
                to 2000.
            episode_length (int): Maximum episode length. Defaults to
                300.
            discount_rewards (bool): Whether to use Monte Carlo returns
                (True) or TD(0) (False). Defaults to True (recommended
                for stability).
            log_freq (int): Frequency of console logging (in
                iterations). Defaults to 10.
            save_freq (int, optional): Frequency of saving checkpoints
                (in iterations). If None, does not save during
                training.
            save_path (str, optional): Base path for saving checkpoints.
                If None, uses current directory / ``checkpoints``.
            **kwargs: Ignored; present for forward-compat with the
                unified interface.

        Returns:
            dict: Training statistics including episode rewards.
        """
        _ = (save_best, verbose, kwargs)
        # Unified-API overrides: translate to legacy names.
        if num_episodes is not None:
            episodes = int(num_episodes)
        if max_steps is not None:
            episode_length = int(max_steps)
        total_steps = (episodes * episode_length) // steps_on_memory
        best_reward = -np.inf

        try:
            for i in tqdm(range(total_steps), desc="Training"):
                memory = self.run_episode(steps_on_memory)
                self.learn(memory, self.steps, discount_rewards=discount_rewards)

                # Console logging
                if i % log_freq == 0 and len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-10:]
                    avg_reward = float(np.mean(recent_rewards))
                    print(
                        f"Step {self.steps} | "
                        f"Episodes: {len(self.episode_rewards)} | "
                        f"Avg Reward (last 10): {avg_reward:.2f}"
                    )

                    # Save best model
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        if save_path:
                            best_path = Path(save_path) / "best_model"
                            best_path.mkdir(parents=True, exist_ok=True)
                            self.save(best_path)

                # Periodic checkpoint saving
                if save_freq and i % save_freq == 0 and i > 0:
                    if save_path is None:
                        save_path = Path.cwd() / "checkpoints"
                    checkpoint_path = Path(save_path) / f"checkpoint_step_{self.steps}"
                    self.save(checkpoint_path)
        finally:
            # Flush + assert canonical metrics contract before returning. Wrap
            # in try/finally so the contract check runs even when the training
            # loop exits early via exception.
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.flush()
                self.writer.assert_contract_satisfied()

        return {
            "episode_rewards": self.episode_rewards,
            "total_steps": self.steps,
            "best_reward": best_reward,
        }

    def close(self):
        """Close TensorBoard writer and cleanup resources."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_param_env(self):
        """Get environment and agent parameters for saving.

        Returns:
            dict: Dictionary with environment and agent policy parameters.
        """
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params = {}
        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        # Получение информации о сигнале справки, если она доступна
        try:
            ref_signal = self.env.unwrapped.ref_signal.__class__
            env_params["ref_signal"] = f"{ref_signal.__module__}.{ref_signal.__name__}"
        except AttributeError:
            pass

        # Добавление информации о пространстве действий и пространстве состояний
        try:
            action_space = str(self.env.action_space)
            env_params["action_space"] = action_space
        except AttributeError:
            pass

        try:
            observation_space = str(self.env.observation_space)
            env_params["observation_space"] = observation_space
        except AttributeError:
            pass

        policy_params = {
            "gamma": self.gamma,
            "entropy_beta": self.entropy_beta,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
        }
        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(self, path=None):
        """Save model to specified directory.

        Creates a timestamped subdirectory to avoid overwriting existing models.
        Saves actor network, critic network, and configuration.

        Args:
            path (str, optional): Base directory path where model will be saved.
                If None, saves to 'checkpoints' directory in current working directory.

        Returns:
            Path: Path to the saved model directory.

        Example:
            >>> agent.save()  # Saves to ./checkpoints/20231005_143022_A2C/
            >>> agent.save('/path/to/models')  # Saves to /path/to/models/20231005_143022_A2C/
        """
        if path is None:
            path = Path.cwd() / "checkpoints"
        else:
            path = Path(path)

        # Create unique directory with timestamp format matching tests
        # Example: Oct06_12-34-56_A2C
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        save_dir = path / f"{date_str}_{self.__class__.__name__}"
        # Handle rare collisions when called multiple times within the same second
        while save_dir.exists():
            time.sleep(1)
            date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            save_dir = path / f"{date_str}_{self.__class__.__name__}"

        # Create directory - fail if it already exists to prevent accidental overwrites
        save_dir.mkdir(parents=True, exist_ok=False)

        # Define file paths
        config_path = save_dir / "config.json"
        actor_path = save_dir / "actor.pth"
        critic_path = save_dir / "critic.pth"

        # Save configuration
        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as outfile:
            json.dump(config, outfile, indent=2)

        # Save model weights
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        print(f"Model saved to: {save_dir}")
        return save_dir

    @classmethod
    def __load(cls, path):
        """Load A2C model from specified directory.

        Args:
            path (str or Path): Path to directory with saved model.

        Returns:
            A2C: Loaded A2C model instance.

        Raises:
            TheEnvironmentDoesNotMatch: If agent type doesn't match expected.
            FileNotFoundError: If required files are not found.
        """
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        actor_path = path / "actor.pth"

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Verify agent type
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch(
                f"Expected {agent_name}, but got {config['policy']['name']}"
            )

        # Recreate environment
        if "tensoraerospace" in config["env"]["name"]:
            env = get_class_from_string(config["env"]["name"])(
                **config["env"]["params"]
            )
        else:
            env = get_class_from_string(config["env"]["name"])()

        # Get dimensions
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        # Recreate networks
        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)

        # Load weights
        actor.load_state_dict(torch.load(actor_path, weights_only=False))
        critic.load_state_dict(torch.load(critic_path, weights_only=False))

        # Create agent
        new_agent = cls(
            env=env, actor=actor, critic=critic, **config["policy"]["params"]
        )

        return new_agent

    @classmethod
    def from_pretrained(cls, repo_name, access_token=None, version=None):
        """Load a pretrained model from a local path or Hugging Face Hub.

        Args:
            repo_name (str): Repository name or local path to the model.
            access_token (str, optional): Access token for Hugging Face Hub.
            version (str, optional): Model version to load.

        Returns:
            A2C: Loaded A2C model instance.
        """
        path = Path(repo_name)
        if path.exists():
            new_agent = cls.__load(path)
            return new_agent
        else:
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent

    def publish_to_hub(self, repo_name, folder_path, access_token=None):
        """Publish model to Hugging Face Hub.

        Args:
            repo_name (str): Repository name in Hub.
            folder_path (str): Path to model folder.
            access_token (str, optional): Access token for authentication.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )


class A2CWithNARXCritic(A2C):
    """A2C variant that uses a NARX critic with history-aware features."""

    def __init__(self, *args, history_length: int = 4, **kwargs):
        """Initialize NARX-enhanced A2C agent.

        Args:
            *args: Forwarded to base A2C constructor.
            history_length: Number of past steps to include in critic features.
            **kwargs: Forwarded to base A2C constructor.
        """
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        super().__init__(*args, **kwargs)

    def _build_narx_batch(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Create stacked NARX features from trajectories.

        Args:
            states: Tensor of shape (T, state_dim).
            actions: Tensor of shape (T, action_dim).

        Returns:
            Tensor with concatenated history features for critic input.
        """
        return build_narx_features(states, actions, self.history_length)

    def _critic_inputs(self, memory, states, next_states):
        """Use physical input histories, resetting them at episode boundaries."""

        def feature(state, past_states, past_actions):
            state = np.asarray(state).reshape(-1)
            action_zero = np.zeros_like(np.asarray(memory[0][0]).reshape(-1))
            xs = [np.asarray(x).reshape(-1) for x in reversed(past_states)]
            us = [np.asarray(u).reshape(-1) for u in reversed(past_actions)]
            xs = (xs + [np.zeros_like(state)] * self.history_length)[
                : self.history_length - 1
            ]
            us = (us + [action_zero] * self.history_length)[: self.history_length]
            return np.concatenate([state, *xs, *us])

        current, following = [], []
        history_states: list[np.ndarray] = []
        history_actions: list[np.ndarray] = []
        for item in memory:
            action, _, state, next_state, done = item
            past_states = list(getattr(item, "previous_states", history_states))
            past_actions = list(getattr(item, "previous_actions", history_actions))
            executed = getattr(item, "executed_action", action)
            current.append(feature(state, past_states, past_actions))
            following.append(
                feature(next_state, past_states + [state], past_actions + [executed])
            )
            history_states = (
                [] if done else (past_states + [state])[-self.history_length :]
            )
            history_actions = (
                [] if done else (past_actions + [executed])[-self.history_length :]
            )
        return to_tensor(current, device=self.device), to_tensor(
            following, device=self.device
        )

    def learn(self, memory, steps, discount_rewards=True):
        """Train actor and NARX critic on a batch of transitions.

        Args:
            memory: Replay buffer slice produced by runner.
            steps: Global step index for logging.
            discount_rewards: Whether to use discounted returns for TD target.
        """
        actions, states, features, td_target = self._prepare_update(
            memory, discount_rewards
        )

        value = self.critic(features)
        critic_loss = F.mse_loss(value, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()

        # Advantage with updated critic
        with torch.no_grad():
            value_updated = self.critic(features)
            advantage = td_target - value_updated
            advantage_normalized = advantage
            if advantage.numel() > 1:
                advantage_normalized = (advantage - advantage.mean()) / (
                    advantage.std(unbiased=False) + 1e-8
                )

        # Actor update (standard A2C)
        norm_dists = self.actor(states)
        log_probs = norm_dists.log_prob(actions)
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        entropy = norm_dists.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1).mean()

        actor_loss = (
            -(log_probs * advantage_normalized).mean() - self.entropy_beta * entropy
        )
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.actor_optim.step()

        # Logging (canonical TB metrics).
        self.writer.add_scalar(
            schema.LOSS_ACTOR, float(actor_loss.detach()), env_step=steps
        )
        self.writer.add_scalar(
            schema.LOSS_CRITIC, float(critic_loss.detach()), env_step=steps
        )
        self.writer.add_scalar(
            schema.A2C.ADVANTAGE_MEAN,
            float(advantage.detach().mean()),
            env_step=steps,
        )
        self.writer.add_scalar(
            schema.POLICY_ACTION_STD,
            float(norm_dists.stddev.detach().mean()),
            env_step=steps,
        )

        # Training counters (mandatory minimum tier).
        self.update_count += 1
        self.writer.add_scalar(
            schema.TRAIN_UPDATES, int(self.update_count), env_step=steps
        )
        self.writer.add_scalar(
            schema.TRAIN_LR,
            float(self.actor_optim.param_groups[0]["lr"]),
            env_step=steps,
        )
