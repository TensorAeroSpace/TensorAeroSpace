"""Proximal Policy Optimization (PPO) algorithm implementation module.

This module contains the PPO algorithm implementation for reinforcement learning,
including actor and critic neural networks, batch iteration functions
and the main PPO agent class for aerospace system control.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)


class RunningMeanStd:
    """Tracks the running mean and standard deviation of observations for normalization.

    This class implements Welford's online algorithm for computing running mean and variance,
    which is numerically stable and memory efficient.

    Attributes:
        mean: Running mean of the data.
        var: Running variance of the data.
        count: Number of samples processed.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """Initialize running statistics.

        Args:
            epsilon: Small value to avoid division by zero. Defaults to 1e-4.
            shape: Shape of the data to track. Defaults to scalar (empty tuple).
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data.

        Args:
            x: New data batch.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update statistics from batch moments.

        Args:
            batch_mean: Mean of the batch.
            batch_var: Variance of the batch.
            batch_count: Number of samples in batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Initialize layer weights and biases using uniform distribution.

    Args:
        layer (nn.Linear): Neural network layer to be initialized.
        init_w (float, optional): Half interval for uniform distribution. Defaults to 3e-3.

    Returns:
        nn.Linear: Layer with initialized weights and biases.

    Examples:
        >>> layer = nn.Linear(10, 5)
        >>> init_layer_uniform(layer)
        Linear(in_features=10, out_features=5, bias=True)
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Critic(nn.Module):
    """Value function network for PPO algorithm.

    The critic estimates the expected return (value) from a given state,
    which is used to compute advantages for policy updates.

    Architecture:
        - Two hidden layers with ReLU activation
        - Final linear layer outputs scalar value estimate

    Attributes:
        d1: First hidden layer.
        d2: Second hidden layer.
        v: Value output layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """Initialize critic network.

        Args:
            input_dim: Dimension of input observations.
            hidden_dim: Number of units in hidden layers. Defaults to 256.
        """
        super(Critic, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.v = init_layer_uniform(self.v)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform forward pass to compute state value.

        Args:
            input_data: Input observation tensor of shape (batch_size, input_dim).

        Returns:
            Value estimates of shape (batch_size, 1).
        """
        x = F.relu(self.d1(input_data))
        x = F.relu(self.d2(x))
        v = self.v(x)
        return v


class Actor(nn.Module):
    """Policy network for PPO algorithm with continuous action spaces.

    The actor implements a Gaussian policy that outputs mean and standard deviation
    for continuous action distributions.

    Architecture:
        - Two hidden layers with ReLU activation
        - Separate output heads for mean (mu) and log std (delta)
        - Tanh activation on outputs to bound actions

    Attributes:
        d1: First hidden layer.
        d2: Second hidden layer.
        mu: Mean output layer for action distribution.
        delta: Log standard deviation output layer.
        log_std_min: Minimum allowed log std value.
        log_std_max: Maximum allowed log std value.
    """

    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 256):
        """Initialize actor network.

        Args:
            input_dim: Dimension of input observations.
            out_dim: Dimension of action space.
            hidden_dim: Number of units in hidden layers. Defaults to 256.
        """
        super(Actor, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        # Mean of the action distribution
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.mu = init_layer_uniform(self.mu)
        # Log std of the action distribution
        self.delta = nn.Linear(hidden_dim, out_dim)
        self.delta = init_layer_uniform(self.delta)
        self.log_std_min = -20
        self.log_std_max = 0

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.distributions.Normal]:
        """Perform forward pass to compute action distribution and sample action.

        Args:
            input_data: Input observation tensor of shape (batch_size, input_dim).

        Returns:
            Tuple containing:
                - Sampled action tensor of shape (batch_size, action_dim).
                - Normal distribution object representing the action distribution.
        """
        x = F.relu(self.d1(input_data))
        x = F.relu(self.d2(x))

        # Continuous action space: Gaussian policy
        mu = torch.tanh(self.mu(x))
        log_std = torch.tanh(self.delta(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action, dist


def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
) -> Generator[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    None,
    None,
]:
    """Create mini-batch iterator for PPO training with shuffled indices.

    This function generates mini-batches by randomly shuffling the data indices
    for each epoch, which helps prevent overfitting and improves generalization.

    Args:
        epoch: Number of epochs to iterate over the data.
        mini_batch_size: Size of each mini-batch.
        states: State tensor of shape (batch_size, state_dim).
        actions: Action tensor of shape (batch_size, action_dim).
        log_probs: Log probability tensor of shape (batch_size, 1).
        returns: Return tensor of shape (batch_size, 1).
        advantages: Advantage tensor of shape (batch_size, 1).
        rewards: Reward tensor of shape (batch_size, 1).
        values: Old value estimates of shape (batch_size, 1).

    Yields:
        Tuple containing mini-batches of (states, actions, log_probs, returns,
        advantages, rewards, values) for each iteration.
    """
    batch_size = states.size(0)
    for _ in range(epoch):
        # Shuffle indices for each epoch
        indices = np.random.permutation(batch_size)
        # Iterate over mini-batches without replacement
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            if end > batch_size:
                end = batch_size
            batch_indices = indices[start:end]
            yield states[batch_indices, :], actions[batch_indices], log_probs[
                batch_indices
            ], returns[batch_indices], advantages[batch_indices], rewards[
                batch_indices
            ], values[
                batch_indices
            ]


class PPO(BaseRLModel):
    """Proximal Policy Optimization (PPO) reinforcement learning agent.

    PPO is a policy gradient method that uses a clipped objective function to ensure
    stable and monotonic policy improvements. This implementation includes:
        - Actor-Critic architecture with separate networks
        - Generalized Advantage Estimation (GAE)
        - Observation and reward normalization
        - Value function clipping
        - Gradient clipping for stability
        - KL divergence early stopping
        - TensorBoard logging

    The agent is designed for continuous control tasks in aerospace applications.

    Attributes:
        actor: Policy network that outputs action distributions.
        critic: Value network that estimates state values.
        gamma: Discount factor for future rewards.
        clip_pram: PPO clipping parameter epsilon.
        gae_lambda: GAE lambda for advantage estimation.
        max_grad_norm: Maximum gradient norm for clipping.
        target_kl: Target KL divergence for early stopping.
        normalize_obs: Whether to normalize observations.
        normalize_reward: Whether to normalize rewards.
        obs_rms: Running statistics for observation normalization.
        ret_rms: Running statistics for return normalization.
        writer: TensorBoard summary writer.
    """

    def __init__(
        self,
        env: Any,
        gamma: float = 0.99,
        max_episodes: int = 30,
        rollout_len: int = 2048,
        clip_pram: float = 0.2,
        num_epochs: int = 64,
        batch_size: int = 64,
        entropy_coef: float = 0.005,
        actor_lr: float = 0.001,
        critic_lr: float = 0.005,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        eval_freq: int = 10,
        seed: int = 336699,
    ) -> None:
        """Initialize agent with given environment and discount coefficient.

        Args:
            env: Environment object with which agent will interact.
            gamma: Discount coefficient. Defaults to 0.99.
            max_episodes: Maximum number of training episodes.
            rollout_len: Number of steps per rollout.
            clip_pram: PPO clipping parameter epsilon.
            num_epochs: Number of optimization epochs per rollout.
            batch_size: Mini-batch size for SGD.
            entropy_coef: Entropy bonus coefficient.
            actor_lr: Learning rate for actor network.
            critic_lr: Learning rate for critic network.
            gae_lambda: GAE lambda parameter for advantage estimation.
            max_grad_norm: Maximum gradient norm for clipping.
            target_kl: Target KL divergence for early stopping.
            normalize_obs: Whether to normalize observations.
            normalize_reward: Whether to normalize rewards.
            actor_hidden_dim: Hidden layer size for actor network.
            critic_hidden_dim: Hidden layer size for critic network.
            eval_freq: Frequency (in episodes) for evaluation.
            seed: Random seed.
        """
        self.gamma = gamma
        self.env = env
        self.actor = Actor(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_dim=actor_hidden_dim,
        )
        self.critic = Critic(
            env.observation_space.shape[0], hidden_dim=critic_hidden_dim
        )
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.clip_pram = clip_pram
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.eval_freq = eval_freq
        torch.manual_seed(seed)
        self.rollout_len = rollout_len
        self.max_episodes = max_episodes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.ep_reward: list = []
        self.total_avgr: list = []
        self.target = False
        self.best_reward = float("-inf")
        self.avg_rewards_list: list = []
        self.writer = SummaryWriter()

        # Observation and reward normalization
        if self.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        if self.normalize_reward:
            self.ret_rms = RunningMeanStd(shape=())

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics.

        Args:
            obs: Raw observation.

        Returns:
            Normalized observation.
        """
        if self.normalize_obs:
            return np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                -10.0,
                10.0,
            )
        return obs

    def act(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Select action for given state.

        Args:
            state: Current environment state.
            deterministic: If True, use mean action (no sampling).

        Returns:
            tuple: Tuple containing action, mean action and log probability.
        """
        if self.normalize_obs:
            state = self._normalize_obs(state)
        state_t = torch.as_tensor(np.array([state]), dtype=torch.float32)
        with torch.no_grad():
            action, dist = self.actor(state_t)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            mean_action = dist.mean
            if deterministic:
                action = mean_action
        return (
            action.detach(),
            mean_action.detach().cpu().numpy(),
            log_prob.detach(),
        )

    def actor_loss(
        self,
        probs: torch.Tensor,
        entropy: torch.Tensor,
        actions: torch.Tensor,
        adv: torch.Tensor,
        old_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate actor losses.

        Args:
            probs: Action probabilities of new policy.
            entropy: Action entropy.
            actions: Actions taken.
            adv: Advantages.
            old_probs: Action probabilities of old policy.

        Returns:
            Tensor: Actor loss function value.
        """
        ratios = torch.exp(probs - old_probs)
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * adv
        # Encourage higher entropy (exploration)
        loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        return loss

    def learn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv: torch.Tensor,
        old_probs: torch.Tensor,
        discnt_rewards: torch.Tensor,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
    ) -> Dict[str, float]:
        """Agent training procedure.

        Args:
            states: States experienced by agent.
            actions: Actions taken by agent.
            adv: Advantages.
            old_probs: Log probabilities of previous actions.
            discnt_rewards: Discounted rewards.
            rewards: Actual received rewards.
            old_values: Previous value function estimates.

        Returns:
            dict: Dictionary with training metrics.
        """
        self.a_opt.zero_grad()
        self.c_opt.zero_grad()
        new_actions, new_distr = self.actor(states)
        # Sum log-probabilities across action dimensions
        new_probs = new_distr.log_prob(actions).sum(dim=-1, keepdim=True)
        # Entropy summed across action dimensions, averaged across batch
        entropy = new_distr.entropy().sum(dim=-1).mean()

        # Calculate approximate KL divergence
        with torch.no_grad():
            log_ratio = new_probs - old_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()

        # Calculate value function with clipping
        v = self.critic(states)
        v_clipped = old_values + torch.clamp(
            v - old_values, -self.clip_pram, self.clip_pram
        )
        # Unclipped value loss
        v_loss_unclipped = (v.squeeze() - discnt_rewards.squeeze()).pow(2)
        # Clipped value loss
        v_loss_clipped = (v_clipped.squeeze() - discnt_rewards.squeeze()).pow(2)
        # Take maximum for conservative updates
        c_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        # Calculate clip fraction (for diagnostics)
        with torch.no_grad():
            ratio = torch.exp(new_probs - old_probs)
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_pram).float().mean()

        # Actor loss
        a_loss = self.actor_loss(new_probs, entropy, actions, adv.detach(), old_probs)

        # Backward passes
        a_loss.backward()
        c_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        # Optimizer steps
        self.a_opt.step()
        self.c_opt.step()

        return {
            "actor_loss": a_loss.item(),
            "critic_loss": c_loss.item(),
            "entropy": float(entropy.detach().cpu().item()),
            "approx_kl": float(approx_kl.cpu().item()),
            "clip_fraction": float(clip_fraction.cpu().item()),
        }

    def test_reward(self) -> float:
        """Test model by executing one episode with deterministic actions.

        Returns:
            float: Total reward per episode.
        """
        total_reward = 0
        reset_return = self.env.reset()
        if type(reset_return) is tuple:
            state, info = reset_return
        else:
            state = reset_return
        done = False
        while not done:
            # Use deterministic actions for evaluation (no sampling)
            action, mean_action, delta = self.act(state, deterministic=True)
            step_return = self.env.step(mean_action[0])
            if len(step_return) > 4:
                next_state, reward, terminated, trunkated, info = step_return
                done = terminated or trunkated
            else:
                next_state, reward, terminated, info = step_return
                done = terminated
            state = next_state
            total_reward += reward
        return total_reward

    def preprocess1(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        rewards: list[torch.Tensor],
        dones: list[torch.Tensor],
        values: list[torch.Tensor],
        probs: list[torch.Tensor],
        gamma: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Preprocess transitions for buffer.

        Args:
            states: List of states.
            actions: List of actions.
            rewards: List of rewards.
            dones: List of boolean values indicating episode termination.
            values: State values.
            probs: Log probabilities of actions.
            gamma: Discount coefficient.

        Returns:
            tuple: Tuple containing processed states, actions, rewards, advantages and probabilities.
        """

        # Use environment observation dimension instead of a hardcoded value
        states2 = torch.cat(states).view(-1, self.env.observation_space.shape[0])
        actions2 = torch.cat(actions).detach()
        rewards2 = torch.cat(rewards)
        dones2 = torch.cat(dones)
        values2 = torch.cat(values).flatten()
        probs2 = torch.cat(probs).detach()

        returns2 = []
        g2 = 0
        for i in reversed(range(len(rewards))):
            delta2 = rewards2[i] + gamma * values2[i + 1] * (1 - dones2[i]) - values2[i]
            g2 = delta2 + gamma * self.gae_lambda * (1 - dones2[i]) * g2
            returns2.insert(0, g2 + values2[i].view(-1, 1))

        # Compute advantages without recreating a tensor from a list of tensors
        returns_tensor = torch.cat(returns2).detach().squeeze()
        adv2 = returns_tensor - values2[:-1]
        # adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return states2, actions2, returns2, adv2, rewards2, probs2

    def train(self) -> None:
        """Train the PPO agent through interaction with the environment.

        This method implements the complete PPO training loop:
            1. Collect rollout data by interacting with environment
            2. Compute advantages using GAE
            3. Update policy and value function using mini-batch SGD
            4. Log metrics to TensorBoard
            5. Periodically evaluate policy performance

        The training loop continues for max_episodes, with each episode consisting
        of rollout_len environment steps. Policy updates are performed using
        num_epochs of optimization over mini-batches of size batch_size.

        Training can be stopped early using KL divergence thresholds (target_kl)
        or by setting self.target = True.

        Note:
            All metrics are logged to TensorBoard including actor/critic losses,
            rewards, entropy, KL divergence, clip fraction, and explained variance.
        """
        for episode in tqdm(range(self.max_episodes)):
            # print("Episode", episode)
            if self.target:
                break

            reset_return = self.env.reset()
            if type(reset_return) is tuple:
                state, info = reset_return
            else:
                state = reset_return
            done = False
            all_aloss = []
            all_entropies = []
            episode_lengths = []
            all_closs = []
            rewards = []
            states = []
            actions = []
            probs = []
            # mus = []
            # deltas = []
            dones = []
            values = []
            scores = []
            score = 0

            curr_ep_len = 0
            rollout_states = []  # For obs normalization update
            for step in range(self.rollout_len):
                rollout_states.append(state)
                action, mu, prob = self.act(state)
                # Normalize state for value function if needed
                state_normalized = (
                    self._normalize_obs(state) if self.normalize_obs else state
                )
                with torch.no_grad():
                    value = self.critic(torch.FloatTensor(np.array([state_normalized])))
                # Clip action to environment bounds to avoid invalid controls
                env_action = action.detach().cpu().numpy()[0]
                try:
                    low, high = (
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )
                    env_action = np.clip(env_action, low, high)
                except Exception:
                    pass
                step_return = self.env.step(env_action)
                if len(step_return) > 4:
                    next_state, reward, terminated, trunkated, info = step_return
                    done = terminated or trunkated
                else:
                    next_state, reward, terminated, info = step_return
                    done = terminated
                score += reward
                curr_ep_len += 1
                dones.append(
                    torch.FloatTensor(np.reshape(done, (1, -1)).astype(np.float64))
                )
                rewards.append(
                    torch.FloatTensor(np.reshape(reward, (1, -1)).astype(np.float64))
                )
                states.append(torch.FloatTensor(state_normalized))
                actions.append(action[0])
                probs.append(prob)
                values.append(value)

                state = next_state
                if done:
                    scores.append(score)
                    episode_lengths.append(curr_ep_len)
                    score = 0
                    curr_ep_len = 0
                    reset_return = self.env.reset()
                    if type(reset_return) is tuple:
                        state, info = reset_return
                    else:
                        state = reset_return

            # Update observation normalization statistics
            if self.normalize_obs:
                self.obs_rms.update(np.array(rollout_states))

            # Calculate next state value for the terminal state
            next_state_normalized = (
                self._normalize_obs(next_state) if self.normalize_obs else next_state
            )
            with torch.no_grad():
                next_value = self.critic(
                    torch.FloatTensor(np.array([next_state_normalized]))
                )
            values.append(next_value)

            _, _, returns, _, _, _ = self.preprocess1(
                states, actions, rewards, dones, values, probs, self.gamma
            )
            states = torch.cat(states).view(-1, self.env.observation_space.shape[0])
            actions = torch.cat(actions).view(-1, 1)
            rewards = torch.cat(rewards)
            returns = torch.cat(returns).detach()
            values = torch.cat(values).detach()
            probs = torch.cat(probs).detach()

            # Reward normalization (normalize returns)
            if self.normalize_reward:
                returns_np = returns.cpu().numpy().flatten()
                self.ret_rms.update(returns_np)
                returns = torch.clamp(
                    (returns - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8),
                    -10.0,
                    10.0,
                )

            advantages = returns - values[:-1]
            # Store old values for clipped value loss
            old_values = values[:-1].clone()

            # Calculate explained variance (quality of value function)
            with torch.no_grad():
                y_pred = values[:-1]
                y_true = returns
                var_y = torch.var(y_true)
                explained_var = (1 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Train for a number of epochs with KL early stopping
            all_approx_kl = []
            all_clip_fractions = []
            for epoch in range(self.num_epochs):
                epoch_kls = []
                for (
                    state,
                    action,
                    old_log_prob,
                    return_,
                    adv,
                    reward,
                    old_val,
                ) in ppo_iter(
                    epoch=1,  # Inner loop already handles epochs
                    mini_batch_size=self.batch_size,
                    states=states,
                    actions=actions,
                    log_probs=probs,
                    returns=returns,
                    advantages=advantages,
                    rewards=rewards,
                    values=old_values,
                ):
                    metrics = self.learn(
                        state,
                        action,
                        adv,
                        old_log_prob,
                        return_,
                        reward,
                        old_val,
                    )
                    all_aloss.append(metrics["actor_loss"])
                    all_closs.append(metrics["critic_loss"])
                    all_entropies.append(metrics["entropy"])
                    all_approx_kl.append(metrics["approx_kl"])
                    all_clip_fractions.append(metrics["clip_fraction"])
                    epoch_kls.append(metrics["approx_kl"])

                # KL early stopping
                if self.target_kl is not None:
                    if np.mean(epoch_kls) > self.target_kl:
                        break

            avg_reward = np.mean(scores) if scores else 0.0
            avg_aloss = np.mean(all_aloss)
            avg_closs = np.mean(all_closs)
            avg_entropy = np.mean(all_entropies)
            avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
            avg_approx_kl = np.mean(all_approx_kl)
            avg_clip_fraction = np.mean(all_clip_fractions)

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Actor", avg_aloss, episode)
            self.writer.add_scalar("Loss/Critic", avg_closs, episode)
            self.writer.add_scalar("Performance/Reward", avg_reward, episode)
            self.writer.add_scalar("Performance/Entropy", avg_entropy, episode)
            self.writer.add_scalar(
                "Performance/Episode Length", avg_episode_length, episode
            )
            self.writer.add_scalar("Diagnostics/Approx KL", avg_approx_kl, episode)
            self.writer.add_scalar(
                "Diagnostics/Clip Fraction", avg_clip_fraction, episode
            )
            self.writer.add_scalar(
                "Diagnostics/Explained Variance", explained_var, episode
            )

            # Periodic evaluation
            if (episode + 1) % self.eval_freq == 0:
                eval_reward = self.test_reward()
                self.writer.add_scalar("Evaluation/Reward", eval_reward, episode)
                # Save best model
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    print(
                        f"\nNew best model! Reward: {eval_reward:.2f} "
                        f"(episode {episode + 1})"
                    )

        # print("Training completed. Average rewards list:", self.avg_rewards_list)

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Get environment and agent parameters for serialization.

        This method extracts all necessary information to reconstruct the agent
        and its environment, including hyperparameters, network architectures,
        and environment specifications.

        Returns:
            Dictionary with two main keys:
                - 'env': Dictionary containing environment name and parameters.
                - 'policy': Dictionary containing agent name and hyperparameters.

        Note:
            For TensorAeroSpace environments, full environment parameters are
            serialized. For other environments, only the class name is stored.
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

        # Get reference signal information if available
        try:
            ref_signal = self.env.ref_signal.__class__
            env_params["ref_signal"] = ref_signal
        except AttributeError:
            pass

        # Add information about action space and observation space
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
            "max_episodes": self.max_episodes,
            "rollout_len": self.rollout_len,
            "clip_pram": self.clip_pram,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "gae_lambda": self.gae_lambda,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "normalize_obs": self.normalize_obs,
            "normalize_reward": self.normalize_reward,
            "actor_hidden_dim": self.actor.d1.out_features,
            "critic_hidden_dim": self.critic.d1.out_features,
            "eval_freq": self.eval_freq,
            "seed": self.seed,
        }
        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(self, path: Union[str, Path, None] = None) -> None:
        """Save the PPO model to disk.

        This method saves all components needed to restore the agent:
            - Configuration file (config.json) with hyperparameters
            - Actor network weights (actor.pth)
            - Critic network weights (critic.pth)
            - Observation normalization statistics (obs_rms.npz, if enabled)
            - Return normalization statistics (ret_rms.npz, if enabled)

        The model is saved in a timestamped directory with format:
        {path}/{Month}{Day}_{Hour}-{Minute}-{Second}_PPO/

        Args:
            path: Directory where the model will be saved. If None, uses current
                working directory. Defaults to None.

        Example:
            >>> agent.save('/path/to/models')
            # Saves to: /path/to/models/Oct05_14-30-45_PPO/
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Current date and time in format 'MonthDay_Hour-Minute-Second'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Create path in current directory with date and time

        config_path = path / date_str / "config.json"
        actor_path = path / date_str / "actor.pth"
        critic_path = path / date_str / "critic.pth"
        obs_rms_path = path / date_str / "obs_rms.npz"
        ret_rms_path = path / date_str / "ret_rms.npz"

        # Create directory if it doesn't exist
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        # Save model
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        # Save normalization statistics
        if self.normalize_obs:
            np.savez(
                obs_rms_path,
                mean=self.obs_rms.mean,
                var=self.obs_rms.var,
                count=self.obs_rms.count,
            )
        if self.normalize_reward:
            np.savez(
                ret_rms_path,
                mean=self.ret_rms.mean,
                var=self.ret_rms.var,
                count=self.ret_rms.count,
            )

    @classmethod
    def __load(cls, path: Union[str, Path]) -> "PPO":
        """Load a PPO model from disk (internal method).

        This private method handles the complete restoration of a saved PPO agent,
        including network weights, configuration, and normalization statistics.

        Args:
            path: Directory containing the saved model files (config.json,
                actor.pth, critic.pth, and optional normalization files).

        Returns:
            Restored PPO agent instance with loaded weights and configuration.

        Raises:
            TheEnvironmentDoesNotMatch: If the agent type in the saved config
                does not match the current class.
            FileNotFoundError: If required model files are not found.

        Note:
            This is a private method. Use `from_pretrained()` for loading models.
        """
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        actor_path = path / "actor.pth"
        obs_rms_path = path / "obs_rms.npz"
        ret_rms_path = path / "ret_rms.npz"

        with open(config_path, "r") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            env = get_class_from_string(config["env"]["name"])(
                **config["env"]["params"]
            )
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])
        # Load weights
        critic_state = torch.load(critic_path)
        actor_state = torch.load(actor_path)
        new_agent.critic.load_state_dict(critic_state)
        new_agent.actor.load_state_dict(actor_state)

        # Load normalization statistics if they exist
        if new_agent.normalize_obs and obs_rms_path.exists():
            obs_rms_data = np.load(obs_rms_path)
            new_agent.obs_rms.mean = obs_rms_data["mean"]
            new_agent.obs_rms.var = obs_rms_data["var"]
            new_agent.obs_rms.count = float(obs_rms_data["count"])

        if new_agent.normalize_reward and ret_rms_path.exists():
            ret_rms_data = np.load(ret_rms_path)
            new_agent.ret_rms.mean = ret_rms_data["mean"]
            new_agent.ret_rms.var = ret_rms_data["var"]
            new_agent.ret_rms.count = float(ret_rms_data["count"])

        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "PPO":
        """Load a pretrained PPO model from local path or Hugging Face Hub.

        This method provides a unified interface for loading models from either:
            1. Local filesystem paths
            2. Hugging Face Hub repositories

        The method automatically detects whether repo_name is a local path or
        a remote repository and handles downloading/loading appropriately.

        Args:
            repo_name: Either a local directory path containing saved model files,
                or a Hugging Face Hub repository name (e.g., 'username/model-name').
            access_token: Hugging Face API token for accessing private repositories.
                Only required for private models. Defaults to None.
            version: Specific version/tag of the model to load from Hub.
                Defaults to None (loads latest version).

        Returns:
            Loaded PPO agent instance ready for inference or further training.

        Examples:
            Load from local path:
            >>> agent = PPO.from_pretrained('./saved_models/my_agent')

            Load from Hugging Face Hub:
            >>> agent = PPO.from_pretrained('username/ppo-pendulum-v1')

            Load specific version with auth:
            >>> agent = PPO.from_pretrained(
            ...     'username/private-model',
            ...     access_token='hf_xxx',
            ...     version='v1.0.0'
            ... )
        """
        path = Path(repo_name)
        if path.exists():
            new_agent = cls.__load(path)
            return new_agent
        else:
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent
