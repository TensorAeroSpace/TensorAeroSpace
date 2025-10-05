"""Proximal Policy Optimization (PPO) algorithm implementation module.

This module contains the PPO algorithm implementation for reinforcement learning,
including actor and critic neural networks, batch iteration functions
and the main PPO agent class for aerospace system control.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

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
    """Tracks the running mean and std of observations for normalization."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """Initialize running statistics.

        Args:
            epsilon: Small value to avoid division by zero.
            shape: Shape of the data to track.
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
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / tot_count
        )
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
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize critic module.

        Args:
            input_dim (int): Input data dimension.
            hidden_dim (int, optional): Hidden layer size. Defaults to 256.

        Performs the following operations:
        - Initialize first linear layer to transform input data to intermediate representation.
        - Initialize second linear layer to compute "value" from intermediate representation.
        - Initialize second linear layer using uniform distribution.
        """
        super(Critic, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.v = init_layer_uniform(self.v)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the network.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Output tensor representing "value" for each input example.

        Applies sequence of operations:
        - Pass input data through first linear layer and apply ReLU.
        - Pass through second hidden layer with ReLU.
        - Pass result through final linear layer to compute "value".
        """
        x = F.relu(self.d1(input_data))
        x = F.relu(self.d2(x))
        v = self.v(x)
        return v


class Actor(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 256):
        """
        Initialize Actor class, which is a subclass of nn.Module.

        Args:
            input_dim (int): Input layer size.
            out_dim (int): Output layer size.
            hidden_dim (int, optional): Hidden layer size. Defaults to 256.

        Initialize linear layers for calculating intermediate representations
        and action parameters.
        Uses custom init_layer_uniform functions to initialize `mu` and
        `delta` layers.
        """
        super(Actor, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Linear(hidden_dim, out_dim)
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.mu = init_layer_uniform(self.mu)
        self.delta = nn.Linear(hidden_dim, out_dim)
        self.delta = init_layer_uniform(self.delta)
        self.log_std_min = -20
        self.log_std_max = 0
        self.r = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_data: torch.Tensor,
        return_reward: bool = False,
        continous_actions: bool = False,
    ) -> Any:
        """
        Perform forward pass through model, computing agent actions
        based on input data.

        Args:
            input_data (Tensor): Input data for model.
            return_reward (bool, optional): Flag indicating whether to
                return reward. Defaults to False.
            continous_actions (bool, optional): Flag indicating whether
                actions should be continuous. Defaults to False.

        Returns:
            Union[Tuple[torch.Tensor, torch.distributions.Normal],
                  Tuple[torch.Tensor, torch.distributions.Normal, torch.Tensor],
                  torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            Depending on flags returns action, distribution
            (and reward if requested).
            If continous_actions True, returns either pair (action, dist)
            or triple (action, dist, r).
            Otherwise returns either action or pair (action, r).
        """
        x = F.relu(self.d1(input_data))
        x = F.relu(self.d2(x))

        if continous_actions:
            mu = torch.tanh(self.mu(x))
            log_std = torch.tanh(self.delta(x))
            log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            if return_reward:
                r = torch.flatten(F.relu(self.r(x)))
                return action, dist, r
            return action, dist
        a = F.softmax(self.a(x), dim=-1)
        if return_reward:
            r = torch.flatten(F.relu(self.r(x)))
            return a, r
        return a


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
):
    """Initialize iterator for PPO.

    Args:
        epoch (int): Number of epochs for iterations.
        mini_batch_size (int): Mini-batch size for each iteration.
        states (torch.Tensor): States tensor.
        actions (torch.Tensor): Actions tensor.
        log_probs (torch.Tensor): Action log probabilities tensor.
        returns (torch.Tensor): Expected returns tensor.
        advantages (torch.Tensor): Advantages tensor.
        rewards (torch.Tensor): Rewards tensor.
        values (torch.Tensor): Old value function estimates.
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
            yield states[batch_indices, :], actions[batch_indices], log_probs[batch_indices], returns[
                batch_indices
            ], advantages[batch_indices], rewards[batch_indices], values[batch_indices]


class PPO(BaseRLModel):
    """Class implementing PPO agent using PyTorch.

    Args:
        env: Environment object.
        gamma (float): Discount coefficient.
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
        self.c_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
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
            self.obs_rms = RunningMeanStd(
                shape=env.observation_space.shape
            )
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
            action, dist = self.actor(state_t, continous_actions=True)
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

    def auxillary_task(self, r: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate auxiliary task losses (reward prediction).

        Args:
            r: Predicted rewards.
            rewards: Real rewards.

        Returns:
            Tensor: MSE loss function value between predicted and real rewards.
        """
        return F.mse_loss(r, rewards)

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
        new_actions, new_distr, r = self.actor(
            states, return_reward=True, continous_actions=True
        )
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
            clip_fraction = (
                torch.abs(ratio - 1.0) > self.clip_pram
            ).float().mean()

        # Actor loss
        a_loss = self.actor_loss(
            new_probs, entropy, actions, adv.detach(), old_probs
        )

        # Backward passes
        a_loss.backward()
        c_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )

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
        """Test model by executing one episode.

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
            action, mean_action, delta = self.act(state)
            step_return = self.env.step(mean_action[0])
            if len(step_return) > 4:
                next_state, reward, terminated, trunkated, info = step_return
                done = terminated or trunkated
            else:
                next_state, reward, terminated, info = step_return
                done = terminated
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
        states2 = torch.cat(states).view(
            -1, self.env.observation_space.shape[0]
        )
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
        """Функция обучения агента.

        В процессе обучения агент проходит через заданное количество эпизодов, собирает данные,
        обрабатывает их и обновляет параметры модели.
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
                    self._normalize_obs(state)
                    if self.normalize_obs
                    else state
                )
                with torch.no_grad():
                    value = self.critic(
                        torch.FloatTensor(np.array([state_normalized]))
                    )
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
                    torch.FloatTensor(
                        np.reshape(done, (1, -1)).astype(np.float64)
                    )
                )
                rewards.append(
                    torch.FloatTensor(
                        np.reshape(reward, (1, -1)).astype(np.float64)
                    )
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
                self._normalize_obs(next_state)
                if self.normalize_obs
                else next_state
            )
            with torch.no_grad():
                next_value = self.critic(
                    torch.FloatTensor(np.array([next_state_normalized]))
                )
            values.append(next_value)

            _, _, returns, _, _, _ = self.preprocess1(
                states, actions, rewards, dones, values, probs, self.gamma
            )
            states = torch.cat(states).view(
                -1, self.env.observation_space.shape[0]
            )
            actions = torch.cat(actions).view(-1, 1)
            rewards = torch.cat(rewards)
            returns = torch.cat(returns).detach()
            values = torch.cat(values).detach()
            probs = torch.cat(probs).detach()
            advantages = returns - values[:-1]
            # Store old values for clipped value loss
            old_values = values[:-1].clone()
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

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
            avg_episode_length = (
                np.mean(episode_lengths) if episode_lengths else 0.0
            )
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
            self.writer.add_scalar(
                "Diagnostics/Approx KL", avg_approx_kl, episode
            )
            self.writer.add_scalar(
                "Diagnostics/Clip Fraction", avg_clip_fraction, episode
            )

            # Periodic evaluation
            if (episode + 1) % self.eval_freq == 0:
                eval_reward = self.test_reward()
                self.writer.add_scalar(
                    "Evaluation/Reward", eval_reward, episode
                )
                # Save best model
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    print(
                        f"\nNew best model! Reward: {eval_reward:.2f} "
                        f"(episode {episode + 1})"
                    )

        # print("Training completed. Average rewards list:", self.avg_rewards_list)

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Получает параметры среды и агента для сохранения.

        Returns:
            dict: Словарь с параметрами среды и политики агента.
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
            ref_signal = self.env.ref_signal.__class__
            env_params["ref_signal"] = ref_signal
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
        """Сохраняет модель PPO в указанной директории.

        Если путь не указан, создает директорию с текущей датой и временем.

        Args:
            path (str, optional): Путь, где будет сохранена модель. Если None,
                                создается директория с текущей датой и временем.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Текущая дата и время в формате 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Создание пути в текущем каталоге с датой и временем

        config_path = path / date_str / "config.json"
        actor_path = path / date_str / "actor.pth"
        critic_path = path / date_str / "critic.pth"

        # Создание директории, если она не существует
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    @classmethod
    def __load(cls, path: Union[str, Path]) -> "PPO":
        """Загружает модель PPO из указанной директории.

        Args:
            path (str or Path): Путь к директории с сохраненной моделью.

        Returns:
            PPO: Загруженный экземпляр модели PPO.

        Raises:
            TheEnvironmentDoesNotMatch: Если тип агента не соответствует ожидаемому.
        """
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        actor_path = path / "actor.pth"
        with open(config_path, "r") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            env = get_class_from_string(config["env"]["name"])(**config["env"]["params"])
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])
        # Load weights
        critic_state = torch.load(critic_path)
        actor_state = torch.load(actor_path)
        new_agent.critic.load_state_dict(critic_state)
        new_agent.actor.load_state_dict(actor_state)
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "PPO":
        """Загружает предобученную модель из локального пути или Hugging Face Hub.

        Args:
            repo_name (str): Имя репозитория или локальный путь к модели.
            access_token (str, optional): Токен доступа для Hugging Face Hub.
            version (str, optional): Версия модели для загрузки.

        Returns:
            PPO: Загруженный экземпляр модели PPO.
        """
        path = Path(repo_name)
        if path.exists():
            new_agent = cls.__load(path)
            return new_agent
        else:
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent
