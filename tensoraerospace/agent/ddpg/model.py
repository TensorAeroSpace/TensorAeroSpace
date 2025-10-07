from __future__ import annotations

import datetime
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:  # Prefer gymnasium typing when available
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

# Device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Optional tqdm progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # Fallback no-op tqdm if not available
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is None:

            class _Dummy:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, n=1):
                    pass

                def set_postfix(self, **kwargs):
                    pass

                def write(self, s):
                    print(s)

            return _Dummy()
        else:
            for x in iterable:
                yield x


# Optional TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:

    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass


from ..base import (  # noqa: E402
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)


class RunningMeanStd:
    """Online computation of running mean and standard deviation.

    Uses Welford's algorithm for numerical stability. This class maintains
    running statistics that can be updated incrementally with new data batches.

    Attributes:
        mean: Running mean of the data.
        var: Running variance of the data.
        count: Total number of samples processed.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        """Initialize RunningMeanStd.

        Args:
            shape: Shape of the data to normalize. Default is scalar (empty tuple).
            epsilon: Small value added to count to prevent division by zero.
                Default is 1e-4.
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update statistics based on a new batch of data.

        Args:
            x: New data batch with shape (batch_size, *data_shape).
                The first dimension is the batch dimension.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update statistics using batch moments.

        This method implements Welford's online algorithm for computing
        running mean and variance in a numerically stable way.

        Args:
            batch_mean: Mean of the batch.
            batch_var: Variance of the batch.
            batch_count: Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        delta_sq_term = np.square(delta) * self.count * batch_count
        M2 = m_a + m_b + delta_sq_term / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Normalize data using current statistics.

        Args:
            x: Data to normalize with shape matching the initialized shape.
            epsilon: Small value added to variance to prevent division by zero.
                Default is 1e-8.

        Returns:
            Normalized data: (x - mean) / sqrt(var + epsilon).
        """
        return (x - self.mean) / np.sqrt(self.var + epsilon)

    def state_dict(self) -> Dict[str, Union[List[float], float, np.ndarray]]:
        """Serialize the current state for checkpointing.

        Returns:
            Dictionary containing mean, variance, and count.
        """
        return {
            "mean": self.mean.tolist() if hasattr(self.mean, "tolist") else self.mean,
            "var": self.var.tolist() if hasattr(self.var, "tolist") else self.var,
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a checkpoint dictionary.

        Args:
            state: Dictionary containing 'mean', 'var', and 'count' keys.
        """
        self.mean = np.array(state.get("mean", self.mean))
        self.var = np.array(state.get("var", self.var))
        self.count = float(state.get("count", self.count))


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms.

    Stores transitions (state, action, reward, next_state, done) and provides
    random sampling for training. Uses a circular buffer for memory efficiency.

    Attributes:
        capacity: Maximum number of transitions to store.
        buffer: List of stored transitions.
        position: Current position in the circular buffer.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize ReplayBuffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Whether the episode terminated.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.

        Raises:
            ValueError: If batch_size is larger than the buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {len(self.buffer)} transitions. "
                f"Wait for more data collection or reduce batch size."
            )
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer state for checkpointing.

        Returns:
            Dictionary containing capacity, buffer contents, and position.
        """
        return {
            "capacity": self.capacity,
            "buffer": self.buffer,
            "position": self.position,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay buffer state from a checkpoint dictionary.

        Args:
            state: Dictionary containing 'capacity', 'buffer', and 'position' keys.
        """
        self.capacity = int(state.get("capacity", self.capacity))
        self.buffer = list(state.get("buffer", []))
        self.position = int(state.get("position", 0))


class OUNoise(object):
    """Ornstein-Uhlenbeck process for action exploration noise.

    Generates temporally correlated noise suitable for continuous control tasks.
    The noise follows a mean-reverting stochastic process that adds smoothness
    to exploration compared to uncorrelated Gaussian noise.

    Attributes:
        mu: Mean of the OU process (typically 0.0).
        theta: Rate of mean reversion (higher = faster decay).
        sigma: Current noise scale (decays from max_sigma to min_sigma).
        max_sigma: Initial noise scale.
        min_sigma: Final noise scale after decay.
        decay_period: Number of steps over which to decay sigma.
        action_dim: Dimensionality of the action space.
        low: Lower bounds of the action space.
        high: Upper bounds of the action space.
        state: Current state of the OU process.
    """

    def __init__(
        self,
        action_space: gym.Space,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.3,
        min_sigma: float = 0.3,
        decay_period: int = 100000,
    ) -> None:
        """Initialize Ornstein-Uhlenbeck noise process.

        Args:
            action_space: Gym action space with .shape, .low, and .high attributes.
            mu: Mean of the OU process. Default is 0.0.
            theta: Rate of mean reversion. Default is 0.15.
            max_sigma: Initial noise scale. Default is 0.3.
            min_sigma: Final noise scale after decay. Default is 0.3.
            decay_period: Number of steps to decay from max_sigma to min_sigma.
                Default is 100000.
        """
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> None:
        """Reset the OU process state to the mean."""
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self) -> np.ndarray:
        """Evolve the OU process by one timestep.

        Returns:
            Updated state of the OU process.
        """
        x = self.state
        random_part = self.sigma * np.random.randn(self.action_dim)
        dx = self.theta * (self.mu - x) + random_part
        self.state = x + dx
        return self.state

    def get_action(self, action: np.ndarray, t: int = 0) -> np.ndarray:
        """Add OU noise to an action and clip to action space bounds.

        Args:
            action: Base action from the policy network.
            t: Current timestep for noise decay. Default is 0.

        Returns:
            Action with added noise, clipped to [low, high].
        """
        ou_state = self.evolve_state()
        # Linearly decay sigma from max_sigma to min_sigma
        self.sigma = self.max_sigma - (
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        )
        return np.clip(action + ou_state, self.low, self.high)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize OU noise state for checkpointing.

        Returns:
            Dictionary containing all OU process parameters and current state.
        """
        return {
            "mu": self.mu,
            "theta": self.theta,
            "sigma": self.sigma,
            "max_sigma": self.max_sigma,
            "min_sigma": self.min_sigma,
            "decay_period": self.decay_period,
            "action_dim": self.action_dim,
            "low": self.low,
            "high": self.high,
            "state": self.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore OU noise state from a checkpoint dictionary.

        Args:
            state: Dictionary containing OU process parameters and state.
        """
        self.mu = float(state.get("mu", self.mu))
        self.theta = float(state.get("theta", self.theta))
        self.sigma = float(state.get("sigma", self.sigma))
        self.max_sigma = float(state.get("max_sigma", self.max_sigma))
        self.min_sigma = float(state.get("min_sigma", self.min_sigma))
        self.decay_period = int(state.get("decay_period", self.decay_period))
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.low = state.get("low", self.low)
        self.high = state.get("high", self.high)
        self.state = np.array(state.get("state", self.state))


class ValueNetwork(nn.Module):
    """Critic network (Q-function) for DDPG.

    Estimates the action-value function Q(s, a). Takes both state and action
    as input and outputs a scalar Q-value. Uses a two-layer fully connected
    architecture with ReLU activations.
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_size: int,
        init_w: float = 3e-3,
    ) -> None:
        """Initialize the value network.

        Args:
            num_inputs: Dimension of the state space.
            num_actions: Dimension of the action space.
            hidden_size: Number of units in each hidden layer.
            init_w: Weight initialization range for the final layer.
                Smaller values help stabilize early training. Default is 3e-3.
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Initialize final layer with small weights for stable Q-value estimates
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q(s, a).

        Args:
            state: Batch of state observations, shape (batch_size, num_inputs).
            action: Batch of actions, shape (batch_size, num_actions).

        Returns:
            Q-values for each (state, action) pair, shape (batch_size, 1).
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class PolicyNetwork(nn.Module):
    """Actor network (policy) for DDPG.

    Deterministic policy that maps states to actions. Uses a two-layer fully
    connected architecture with ReLU activations and tanh output. Automatically
    scales the tanh output from [-1, 1] to the actual action space bounds.

    Attributes:
        action_scale: Scale factor to map tanh output to action range.
        action_bias: Bias to center the action range.
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_size: int,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        init_w: float = 3e-3,
    ) -> None:
        """Initialize the policy network.

        Args:
            num_inputs: Dimension of the state space.
            num_actions: Dimension of the action space.
            hidden_size: Number of units in each hidden layer.
            action_low: Lower bounds of the action space. If None, defaults to [-1].
            action_high: Upper bounds of the action space. If None, defaults to [1].
            init_w: Weight initialization range for the final layer.
                Smaller values help stabilize early training. Default is 3e-3.
        """
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # Initialize final layer with small weights for stable policy updates
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        # Store action space bounds for automatic scaling
        if action_low is not None and action_high is not None:
            scale = (action_high - action_low) / 2.0
            self.action_scale = torch.FloatTensor(scale).to(device)
            bias = (action_high + action_low) / 2.0
            self.action_bias = torch.FloatTensor(bias).to(device)
        else:
            # Default to [-1, 1] if bounds not provided
            self.action_scale = torch.FloatTensor([1.0]).to(device)
            self.action_bias = torch.FloatTensor([0.0]).to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the action for a given state.

        Args:
            state: Batch of state observations, shape (batch_size, num_inputs).

        Returns:
            Actions scaled to [action_low, action_high], shape (batch_size, num_actions).
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # Scale tanh output [-1, 1] to [action_low, action_high]
        x = x * self.action_scale + self.action_bias
        return x

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for a single state (inference mode).

        Args:
            state: Single state observation as numpy array.

        Returns:
            Action as numpy array, scaled to action space bounds.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state)
        return action.squeeze(0).cpu().numpy()


class DDPG:
    """Deep Deterministic Policy Gradient (DDPG) agent.

    DDPG is an off-policy actor-critic algorithm for continuous control.
    It combines DPG with Deep Q-Learning techniques like experience replay
    and target networks for stability.

    Key features:
        - Deterministic policy (actor) and Q-function (critic)
        - Target networks with soft updates (Polyak averaging)
        - Experience replay buffer for sample efficiency
        - Ornstein-Uhlenbeck noise for exploration
        - Optional observation normalization for faster convergence
        - Automatic action scaling to environment bounds

    Reference:
        Lillicrap et al. "Continuous control with deep reinforcement learning" (2015)
        https://arxiv.org/abs/1509.02971
    """

    def __init__(
        self,
        env: gym.Env,
        value_lr: float,
        policy_lr: float,
        replay_buffer_size: int,
        normalize_observations: bool = True,
    ) -> None:
        """Initialize DDPG agent.

        Args:
            env: Gym environment with continuous action space. Must have
                .observation_space, .action_space, .reset(), and .step() methods.
            value_lr: Learning rate for the critic (Q-function) network.
            policy_lr: Learning rate for the actor (policy) network.
            replay_buffer_size: Maximum number of transitions to store in replay buffer.
            normalize_observations: Whether to normalize observations using running
                mean and standard deviation. Recommended for faster convergence.
                Default is True.
        """
        self.env = env
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        self.replay_buffer_size = replay_buffer_size
        self.normalize_observations = normalize_observations

        self.ou_noise = OUNoise(self.env.action_space)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = 256

        # Initialize observation normalizer
        self.obs_rms: Optional[RunningMeanStd]
        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=(self.state_dim,))
        else:
            self.obs_rms = None

        # Get action space bounds for proper scaling
        action_low = env.action_space.low
        action_high = env.action_space.high

        self.value_net = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)
        self.policy_net = PolicyNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        self.target_value_net = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)
        self.target_policy_net = PolicyNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.policy_lr
        )

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # TensorBoard writer (lazy init in learn to include run-time params)
        self.writer = None

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics.

        Args:
            obs: Raw observation from the environment.

        Returns:
            Normalized observation if normalization is enabled, otherwise
            returns the original observation unchanged.
        """
        if self.normalize_observations and self.obs_rms is not None:
            return self.obs_rms.normalize(obs)
        return obs

    def ddpg_update(
        self,
        batch_size: int,
        gamma: float = 0.99,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        soft_tau: float = 1e-2,
    ) -> None:
        """Perform one DDPG update step on both actor and critic networks.

        This method implements the core DDPG algorithm:
        1. Sample a minibatch from replay buffer
        2. Compute target Q-values using target networks
        3. Update critic by minimizing TD error
        4. Update actor using deterministic policy gradient
        5. Soft update target networks (Polyak averaging)

        Args:
            batch_size: Number of transitions to sample from replay buffer.
            gamma: Discount factor for future rewards. Default is 0.99.
            min_value: Minimum value for Q-value clipping. Default is -inf.
            max_value: Maximum value for Q-value clipping. Default is inf.
            soft_tau: Soft update coefficient (Polyak averaging). Values close to 0
                mean slower updates. Default is 1e-2.
        """
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = batch

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        except Exception:
            pass
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        except Exception:
            pass
        self.value_optimizer.step()

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        # Log training metrics if writer is available
        if self.writer is not None:
            try:
                self.writer.add_scalar(
                    "loss/policy", float(policy_loss.item()), self.frame_idx
                )
                self.writer.add_scalar(
                    "loss/value", float(value_loss.item()), self.frame_idx
                )
                with torch.no_grad():
                    mean_action = self.policy_net(state).mean().item()
                self.writer.add_scalar(
                    "policy/mean_action", float(mean_action), self.frame_idx
                )
            except Exception:
                pass

    def learn(
        self,
        max_frames: int,
        max_steps: int,
        batch_size: int,
        gamma: float = 0.995,
        soft_tau: float = 5e-3,
        warmup_frames: int = 10_000,
        updates_per_step: int = 1,
        target_value_clip: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ) -> None:
        """Train the DDPG agent.

        Runs the main training loop: collect experience, update networks, and
        log metrics. Supports warmup period for initial exploration and multiple
        updates per environment step for sample efficiency.

        Args:
            max_frames: Maximum number of environment steps to train for.
            max_steps: Maximum steps per episode before truncation.
            batch_size: Minibatch size for network updates.
            gamma: Discount factor for future rewards. Default is 0.995.
            soft_tau: Soft update coefficient for target networks (Polyak averaging).
                Smaller values mean slower updates. Default is 5e-3.
            warmup_frames: Number of steps to collect before starting updates.
                Allows the replay buffer to fill with diverse experience. Default is 10_000.
            updates_per_step: Number of gradient updates per environment step.
                Higher values improve sample efficiency but slow down training. Default is 1.
            target_value_clip: Tuple of (min, max) for Q-value clipping. Helps prevent
                overestimation. Set to None to disable clipping. Default is (-10.0, 10.0).
        """
        self.max_frames = max_frames
        self.max_steps = max_steps
        self.frame_idx = 0
        self.rewards = []
        self.batch_size = batch_size

        # Lazy init TensorBoard writer with a sensible logdir
        if self.writer is None:
            try:
                logdir = os.path.join("runs", "ddpg")
                os.makedirs(logdir, exist_ok=True)
                self.writer = SummaryWriter()
            except Exception:
                self.writer = None

        with tqdm(total=max_frames, desc="DDPG Training") as pbar:
            while self.frame_idx < max_frames:
                state = self.env.reset()[0]
                self.ou_noise.reset()
                episode_reward = 0
                # Collect states for batch normalization update
                episode_states = []

                for step in range(max_steps):
                    # Store raw state for normalization update
                    episode_states.append(state)

                    # Normalize state before passing to policy
                    normalized_state = self._normalize_observation(state)
                    action = self.policy_net.get_action(normalized_state)
                    action = self.ou_noise.get_action(action, step)
                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = self.env.step(action)
                    done = terminated or truncated

                    # Store normalized states in replay buffer
                    norm_next = self._normalize_observation(next_state)
                    self.replay_buffer.push(
                        normalized_state,
                        action,
                        reward,
                        norm_next,
                        done,
                    )
                    # Warmup: collect transitions without updates
                    if (
                        self.frame_idx > warmup_frames
                        and len(self.replay_buffer) > batch_size
                    ):
                        for _ in range(max(1, int(updates_per_step))):
                            if target_value_clip is None:
                                mn, mx = -np.inf, np.inf
                            else:
                                mn, mx = float(target_value_clip[0]), float(
                                    target_value_clip[1]
                                )
                            self.ddpg_update(
                                batch_size,
                                gamma=gamma,
                                min_value=mn,
                                max_value=mx,
                                soft_tau=soft_tau,
                            )

                    state = next_state
                    episode_reward += reward
                    self.frame_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        frame=self.frame_idx,
                        ep_reward=float(episode_reward),
                    )

                    if done:
                        break

                self.rewards.append(episode_reward)

                # Update observation normalization statistics with episode data
                if (
                    self.normalize_observations
                    and self.obs_rms is not None
                    and episode_states
                ):
                    episode_states_array = np.array(episode_states)
                    self.obs_rms.update(episode_states_array)

                # Log per-episode reward
                if self.writer is not None:
                    try:
                        self.writer.add_scalar(
                            "Performance/Reward",
                            float(episode_reward),
                            len(self.rewards),
                        )
                    except Exception:
                        pass

    def _collect_grads(self, model: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
        """Collect parameter gradients of a model as CPU tensors.

        Helper method for saving gradients in checkpoints for debugging
        or continued training from exact gradient state.

        Args:
            model: PyTorch model whose gradients to collect.

        Returns:
            Dictionary mapping parameter names to gradient tensors (or None).
        """
        # Use typing compatible with Python <3.10 to satisfy linters
        from typing import Dict, Optional

        grads: Dict[str, Optional[torch.Tensor]] = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                grads[name] = None
            else:
                grads[name] = param.grad.detach().cpu()
        return grads

    def save(self, filepath: Union[str, Path], include_grads: bool = False) -> None:
        """Save training state (checkpoint) or full model folder.

        Supports two save formats:

        1. **Single file checkpoint** (.pt/.pth extension):
           - Backward compatible format
           - Contains all networks, optimizers, buffers, and training state
           - Suitable for resuming training

        2. **Directory format** (no extension or other extensions):
           - HuggingFace Hub compatible structure
           - Separate files for config and each network
           - Suitable for model sharing and deployment

        Args:
            filepath: Path to save location. If ends with .pt/.pth, saves as
                single file. Otherwise saves as directory.
            include_grads: Whether to save optimizer states and gradients.
                Only applicable for single file checkpoints. Default is False.

        Examples:
            >>> agent.save("checkpoint.pt", include_grads=True)  # Single file
            >>> agent.save("my_model")  # Directory with config.json, etc.
        """
        # Directory-style save (HF-style)
        ext = os.path.splitext(str(filepath))[1].lower()
        if ext not in (".pt", ".pth"):
            folder = Path(str(filepath)).expanduser()
            folder.mkdir(parents=True, exist_ok=True)

            # 1) Save config (env + policy params)
            config = self.get_param_env()
            with open(folder / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            # 2) Save networks
            torch.save(self.policy_net, folder / "policy.pth")
            torch.save(self.value_net, folder / "value.pth")
            torch.save(self.target_policy_net, folder / "target_policy.pth")
            torch.save(self.target_value_net, folder / "target_value.pth")

            # 3) Optionally save optimizers
            if include_grads:
                p_opt_path = folder / "policy_optim.pth"
                v_opt_path = folder / "value_optim.pth"
                torch.save(self.policy_optimizer.state_dict(), p_opt_path)
                torch.save(self.value_optimizer.state_dict(), v_opt_path)
            return

        # File checkpoint (original behavior)
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        ckpt = {
            "value_net": self.value_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
            "target_policy_net": self.target_policy_net.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "ou_noise": self.ou_noise.state_dict(),
            "normalize_observations": self.normalize_observations,
            "frame_idx": getattr(self, "frame_idx", 0),
            "rewards": getattr(self, "rewards", []),
            "max_frames": getattr(self, "max_frames", None),
            "max_steps": getattr(self, "max_steps", None),
            "batch_size": getattr(self, "batch_size", None),
        }

        # Save observation normalizer if it exists
        if self.obs_rms is not None:
            ckpt["obs_rms"] = self.obs_rms.state_dict()

        if include_grads:
            ckpt["value_net_grads"] = self._collect_grads(self.value_net)
            ckpt["policy_net_grads"] = self._collect_grads(self.policy_net)

        torch.save(ckpt, filepath)

    def load(
        self,
        filepath: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        load_optimizer: bool = True,
        load_targets: bool = True,
        load_replay: bool = True,
        load_noise: bool = True,
        load_grads: bool = False,
        strict: bool = True,
    ) -> None:
        """Load training state from a checkpoint file.

        Restores networks, optimizers, replay buffer, OU noise, and observation
        normalization statistics. Provides granular control over which components
        to restore.

        Args:
            filepath: Path to checkpoint file (.pt or .pth).
            map_location: Device to load tensors to. If None, uses current device.
                Can be 'cpu', 'cuda', 'cuda:0', etc.
            load_optimizer: Whether to restore optimizer states (momentum, etc.).
                Set to False for inference only. Default is True.
            load_targets: Whether to restore target network weights.
                Set to False for inference only. Default is True.
            load_replay: Whether to restore replay buffer contents.
                Set to False to start with fresh buffer. Default is True.
            load_noise: Whether to restore OU noise state.
                Set to False to reset exploration. Default is True.
            load_grads: Whether to restore parameter gradients.
                Useful for debugging gradient flow. Default is False.
            strict: Whether to strictly match state dict keys.
                Set to False for partial loading. Default is True.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        ckpt = torch.load(filepath, map_location=map_location, weights_only=False)

        self.value_net.load_state_dict(ckpt["value_net"], strict=strict)
        self.policy_net.load_state_dict(ckpt["policy_net"], strict=strict)
        if load_targets:
            self.target_value_net.load_state_dict(
                ckpt["target_value_net"], strict=strict
            )
            self.target_policy_net.load_state_dict(
                ckpt["target_policy_net"], strict=strict
            )

        if load_optimizer and "value_optimizer" in ckpt:
            self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        if load_optimizer and "policy_optimizer" in ckpt:
            self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])

        if load_replay and "replay_buffer" in ckpt:
            self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
        if load_noise and "ou_noise" in ckpt:
            self.ou_noise.load_state_dict(ckpt["ou_noise"])

        # Restore observation normalizer if it exists
        if "obs_rms" in ckpt and self.obs_rms is not None:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])

        default_frame = getattr(self, "frame_idx", 0)
        self.frame_idx = int(ckpt.get("frame_idx", default_frame))
        self.rewards = list(ckpt.get("rewards", []))
        default_max_f = getattr(self, "max_frames", None)
        self.max_frames = ckpt.get("max_frames", default_max_f)
        default_max_s = getattr(self, "max_steps", None)
        self.max_steps = ckpt.get("max_steps", default_max_s)
        default_batch = getattr(self, "batch_size", None)
        self.batch_size = ckpt.get("batch_size", default_batch)

        if load_grads:
            vgrads = ckpt.get("value_net_grads")
            pgrads = ckpt.get("policy_net_grads")
            if vgrads is not None:
                for name, param in self.value_net.named_parameters():
                    grad = vgrads.get(name)
                    if grad is None:
                        param.grad = None
                    else:
                        param.grad = grad.to(param.device).clone()
            if pgrads is not None:
                for name, param in self.policy_net.named_parameters():
                    grad = pgrads.get(name)
                    if grad is None:
                        param.grad = None
                    else:
                        param.grad = grad.to(param.device).clone()

    # ====== HuggingFace-style API (mirror of SAC) ======
    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Collect environment and policy parameters for saving.

        Creates a configuration dictionary containing all information needed
        to reconstruct the agent and environment. Compatible with HuggingFace
        Hub format for model sharing.

        Returns:
            Dictionary with 'env' and 'policy' keys, each containing:
                - 'name': Fully qualified class name
                - 'params': Initialization parameters
        """
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        try:
            if "tensoraerospace" in env_name:
                env_params = serialize_env(self.env)
        except Exception:
            env_params = {}

        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        policy_params = {
            "value_lr": self.value_lr,
            "policy_lr": self.policy_lr,
            "hidden_dim": self.hidden_dim,
            "replay_buffer_size": self.replay_buffer_size,
            "normalize_observations": self.normalize_observations,
            "device": device.type,
            "ou_noise": {
                "theta": getattr(self.ou_noise, "theta", 0.15),
                "max_sigma": getattr(self.ou_noise, "max_sigma", 0.3),
                "min_sigma": getattr(self.ou_noise, "min_sigma", 0.3),
                "decay_period": getattr(self.ou_noise, "decay_period", 100000),
            },
        }

        # Add obs_rms state if available
        if self.obs_rms is not None:
            policy_params["obs_rms"] = self.obs_rms.state_dict()

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    @classmethod
    def __load(
        cls,
        path: Union[str, Path],
        load_gradients: bool = False,
    ) -> "DDPG":
        path = Path(path)
        config_path = path / "config.json"
        policy_path = path / "policy.pth"
        value_path = path / "value.pth"
        target_policy_path = path / "target_policy.pth"
        target_value_path = path / "target_value.pth"
        policy_optim_path = path / "policy_optim.pth"
        value_optim_path = path / "value_optim.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch

        # Recreate env
        if "tensoraerospace" in config["env"]["name"]:
            env = get_class_from_string(config["env"]["name"])(
                **config["env"]["params"]
            )
        else:
            env = get_class_from_string(config["env"]["name"])()

        p = config["policy"]["params"]
        new_agent = cls(
            env=env,
            value_lr=float(p.get("value_lr", 1e-3)),
            policy_lr=float(p.get("policy_lr", 1e-3)),
            replay_buffer_size=int(p.get("replay_buffer_size", 100000)),
            normalize_observations=bool(p.get("normalize_observations", True)),
        )

        # Load networks
        new_agent.policy_net = torch.load(
            policy_path, map_location=device, weights_only=False
        )
        new_agent.value_net = torch.load(
            value_path, map_location=device, weights_only=False
        )
        new_agent.target_policy_net = torch.load(
            target_policy_path, map_location=device, weights_only=False
        )
        new_agent.target_value_net = torch.load(
            target_value_path, map_location=device, weights_only=False
        )

        # Reinit optimizers to match new params
        policy_lr = float(p.get("policy_lr", 1e-3))
        new_agent.policy_optimizer = optim.Adam(
            new_agent.policy_net.parameters(), lr=policy_lr
        )
        value_lr = float(p.get("value_lr", 1e-3))
        new_agent.value_optimizer = optim.Adam(
            new_agent.value_net.parameters(), lr=value_lr
        )

        # Restore obs_rms if available
        if "obs_rms" in p and new_agent.obs_rms is not None:
            new_agent.obs_rms.load_state_dict(p["obs_rms"])

        if load_gradients:
            if policy_optim_path.exists():
                st = torch.load(
                    policy_optim_path, map_location=device, weights_only=False
                )
                new_agent.policy_optimizer.load_state_dict(st)
            if value_optim_path.exists():
                st = torch.load(
                    value_optim_path, map_location=device, weights_only=False
                )
                new_agent.value_optimizer.load_state_dict(st)
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "DDPG":
        """Load a pretrained DDPG model from local directory or Hugging Face Hub.

        Automatically detects whether the path is local or a Hub repository.
        Downloads model if necessary and reconstructs the complete agent with
        environment, networks, and all parameters.

        Args:
            repo_name: Local path or HuggingFace Hub repository name.
                Examples: './my_model', 'username/ddpg-b747', '/abs/path/to/model'.
            access_token: HuggingFace API token for private repositories.
                Not needed for public repos or local paths. Default is None.
            version: Specific version/commit/tag to load from Hub.
                Default is None (latest version).
            load_gradients: Whether to restore optimizer states with gradients.
                Useful for continuing training. Default is False.

        Returns:
            Initialized DDPG agent with loaded weights and configuration.

        Raises:
            FileNotFoundError: If local path doesn't exist.
            TheEnvironmentDoesNotMatch: If config specifies wrong agent class.

        Examples:
            >>> # Load from local directory
            >>> agent = DDPG.from_pretrained("./my_saved_model")
            >>>
            >>> # Load from Hugging Face Hub
            >>> agent = DDPG.from_pretrained("username/ddpg-b747-v1")
            >>>
            >>> # Load private model with token
            >>> agent = DDPG.from_pretrained(
            ...     "username/private-model",
            ...     access_token="hf_..."
            ... )
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                msg = f"Local directory not found: '{repo_name}'. "
                msg += "Please check the path."
                raise FileNotFoundError(msg)
            return cls.__load(p, load_gradients=load_gradients)

        folder_path = BaseRLModel.from_pretrained(
            repo_name, access_token=access_token, version=version
        )
        return cls.__load(folder_path, load_gradients=load_gradients)

    def push_to_hub(
        self,
        repo_name: str,
        access_token: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        include_gradients: bool = False,
    ) -> str:
        """Save the model and upload to Hugging Face Hub.

        Saves the model in HuggingFace-compatible format (config.json + separate
        network files) and uploads to the specified repository. Creates the
        repository if it doesn't exist.

        Args:
            repo_name: Name of the HuggingFace Hub repository.
                Format: 'username/repo-name' or just 'repo-name' (uses your username).
            access_token: HuggingFace API token with write access.
                Required for pushing. Get from https://huggingface.co/settings/tokens
            save_path: Local directory to save model before uploading.
                If None, creates a timestamped directory. Default is None.
            include_gradients: Whether to save optimizer states.
                Useful for sharing training checkpoints. Default is False.

        Returns:
            Path to the local saved folder.

        Raises:
            ValueError: If access_token is not provided or invalid.

        Examples:
            >>> agent.push_to_hub(
            ...     repo_name="my-awesome-ddpg",
            ...     access_token="hf_your_token_here"
            ... )
            'Oct05_14-23-45_DDPG'

        Note:
            The uploaded model can be loaded by anyone using:
            >>> agent = DDPG.from_pretrained("username/my-awesome-ddpg")
        """
        if save_path is None:
            date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            save_path = Path.cwd() / f"{date_str}_{self.__class__.__name__}"
        else:
            save_path = Path(str(save_path))
        save_path.mkdir(parents=True, exist_ok=True)

        # Save in folder-style format for hub
        self.save(save_path, include_grads=include_gradients)

        # Upload
        BaseRLModel().publish_to_hub(
            repo_name=repo_name,
            folder_path=str(save_path),
            access_token=access_token,
        )
        return str(save_path)
