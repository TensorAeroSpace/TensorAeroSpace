"""Generative Adversarial Imitation Learning (GAIL) agent.

This module contains the core GAIL implementation and supporting neural network
components used for imitation learning within TensorAeroSpace.
"""

import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def init_weights(m):
    """Initialize layer weights/biases with a normal distribution.

    Args:
        m (nn.Module): Layer/module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    """Combined policy/value network used by GAIL."""

    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        """Create actor-critic networks.

        Args:
            num_inputs (int): Input dimension.
            num_outputs (int): Output/action dimension.
            hidden_size (int): Hidden layer size.
            std (float, optional): Initial log-std scale. Defaults to 0.0.
        """
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: ``(action_distribution, value)``.
        """
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        next_value (Tensor): Value estimate for the last state.
        rewards (Tensor): Rewards.
        masks (Tensor): Terminal masks (0 for terminal, 1 otherwise).
        values (Tensor): Value estimates.
        gamma (float): Discount factor. Defaults to 0.99.
        tau (float): GAE parameter. Defaults to 0.95.

    Returns:
        list: Advantage-weighted returns (as a list of tensors).
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    """Mini-batch iterator used by PPO updates."""
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :
        ], returns[rand_ids, :], advantage[rand_ids, :]


class Discriminator(nn.Module):
    """Binary classifier distinguishing expert vs. policy trajectories."""

    def __init__(self, num_inputs, hidden_size):
        """Create the discriminator network.

        Args:
            num_inputs (int): Input dimension.
            hidden_size (int): Hidden layer size.
        """
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        """Compute discriminator probability for state-action pairs."""
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        prob = F.sigmoid(self.linear3(x))
        return prob


class GAIL:
    """Generative Adversarial Imitation Learning trainer."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        max_steps: int,
        mini_batch_size: int,
        epochs: int,
        data: np.ndarray,
    ):
        """Initialize the GAIL algorithm.

        Args:
            env: Environment instance.
            learning_rate (float): Learning rate.
            max_steps (int): Maximum steps per rollout/episode.
            mini_batch_size (int): Mini-batch size.
            epochs (int): Number of training epochs.
            data (Array): Expert demonstrations (states/actions).
        """
        self.env = env
        self.lr = learning_rate
        self.max_steps = max_steps
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.data = data

        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]

        self.model = ActorCritic(self.num_inputs, self.num_outputs, 256).to(device)
        self.discriminator = Discriminator(self.num_inputs + self.num_outputs, 128).to(
            device
        )

        self.discrim_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_discrim = optim.Adam(self.discriminator.parameters(), lr=self.lr)

    def expert_reward(self, state: torch.Tensor, action: np.ndarray) -> np.ndarray:
        """Compute imitation reward using the discriminator."""
        state = state.cpu().numpy()
        state_action = torch.FloatTensor(np.concatenate([state, action], 1)).to(device)
        return -np.log(self.discriminator(state_action).cpu().data.numpy())

    def test_env(self) -> float:
        """Run one evaluation rollout and return total reward."""
        state = self.env.reset()[0].reshape(1, -1)
        done = False
        total_reward = 0
        for _ in range(self.max_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = self.model(state)
            next_state, reward, done, _, _ = self.env.step(
                dist.sample().cpu().numpy()[0]
            )
            next_state = next_state.reshape(1, -1)
            state = next_state
            total_reward += reward
        return total_reward

    def ppo_update(
        self,
        ppo_epochs: int,
        mini_batch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        clip_param: float = 0.2,
    ) -> None:
        """PPO update function.

        Args:
            ppo_epochs (int): Number of epochs.
            mini_batch_size (int): Mini-batch size.
            states (Tensor): Batch of states.
            actions (Tensor): Batch of actions.
            log_probs (Tensor): Batch of action log probabilities.
            returns (Tensor): Batch of discounted rewards.
            advantages (Tensor): Batch of advantage function values.
            clip_param (float, optional): Clipping constant.
        """
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def learn(self, max_frames: int, max_reward: float) -> None:
        """Agent training function.

        Args:
            max_frames (int): Maximum number of steps in the environment.
            max_reward (int): Reward threshold for stopping training.
        """
        self.max_frames = max_frames

        test_rewards = []
        frame_idx = 0

        i_update = 0
        state = self.env.reset()[0].reshape(1, -1)
        early_stop = False

        while frame_idx < max_frames and not early_stop:
            i_update += 1

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(self.max_steps):
                state = torch.FloatTensor(state).to(device)
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done, _, _ = self.env.step(action.cpu().numpy())
                next_state = next_state.reshape(1, -1)
                reward = self.expert_reward(state, action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).to(device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

                states.append(state)
                actions.append(action)

                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env() for _ in range(10)])
                    print(test_reward)
                    test_rewards.append(test_reward)
                    if test_reward > max_reward:
                        early_stop = True

            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            if i_update % 3 == 0:
                self.ppo_update(
                    4,
                    self.mini_batch_size,
                    states,
                    actions,
                    log_probs,
                    returns,
                    advantage,
                )

            expert_state_action = self.data[
                np.random.randint(0, self.data.shape[0], 2 * self.max_steps * 16), :
            ]
            expert_state_action = torch.FloatTensor(expert_state_action).to(device)
            state_action = torch.cat([states, actions], 1)
            fake = self.discriminator(state_action)
            real = self.discriminator(expert_state_action)
            self.optimizer_discrim.zero_grad()
            discrim_loss = self.discrim_criterion(
                fake, torch.ones((states.shape[0], 1)).to(device)
            ) + self.discrim_criterion(
                real, torch.zeros((expert_state_action.size(0), 1)).to(device)
            )
            discrim_loss.backward()
            self.optimizer_discrim.step()
