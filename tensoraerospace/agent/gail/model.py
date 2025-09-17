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
    """
    Инициализирует веса и смещения (биасы) слоя с помощью нормального распределения.

    Args:
        m (nn.Linear): Слой нейронной сети, который будет инициализирован.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        """
        Инициализирует модуль актора - критика.

        Args:
            num_inputs (int): Размерность входных данных.
            num_outputs (int): Размерность выходных данных.
            hidden_size (int): Размер скрытого слоя.
            std (float, optional): стандартное отклонение
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
        """
        Производит прямой проход сети.

        Args:
            x (Tensor): Тензор входных данных.

        Returns:
            (Distribution, Tensor): распределение по действиям и значение функции критика
        """
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """
    Вычисляет значения advantage функции

    Args:
        next_value (Tensor): Значение функции критика для последнего состояния.
        rewards (Tensor): Значения наград.
        masks (Tensor): Маски для терминальных состояний.
        values (Tensor): Значение функции критика.
        gamma (float, optional): константа гамма.
        tau (foat, optional): константа тау.

    Returns:
        Tensor: значения advantage функции
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
    """
    Итератор для алгоритма PPO
    Args:
        mini_batch_size (int): Размер мини-батча.
        states (Tensor): Батч состояний.
        actions (Tensor): Батч действий.
        log_probs (Tensor): Батч логарифмов вероятностей действий.
        returns (Tensor): Батч отложенных наград.
        advantage (foat, optional): Батч значений advantage функции.

    Returns:
        Tensor: минибатч для итерации обновления PPO
    """
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :
        ], returns[rand_ids, :], advantage[rand_ids, :]


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        """
        Инициализирует модуль Дискриминатора.

        Args:
            num_inputs (int): Размерность входных данных.
            hidden_size (int): Размер скрытого слоя.
        """
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        prob = F.sigmoid(self.linear3(x))
        return prob


class GAIL:
    def __init__(self, env, learning_rate, max_steps, mini_batch_size, epochs, data):
        """
        Инициализация алгоритма GAIL
        Args:
            env: объект окружения, с которым будет взаимодействовать агент.
            learning_rate (float): learning rate.
            max_steps (int): максимальное количество шагов в среде.
            mini_batch_size (int): размер мини-батча.
            epochs (int): количество эпох обучения.
            data (Array): экспертные данные (состояния и выбранные действия).

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

    def expert_reward(self, state, action):
        """
        Награда на основе экспертных данных
        Args:
            state (Tensor): состояние среды.
            action (float): действие агента.
        Returns:
            int: награда дискриминатора.
        """
        state = state.cpu().numpy()
        state_action = torch.FloatTensor(np.concatenate([state, action], 1)).to(device)
        return -np.log(self.discriminator(state_action).cpu().data.numpy())

    def test_env(self):
        """
        Функция для тестирования алгоритма на основе одного эпизода в среде.
        Returns:
            int: Награда за эпизод.
        """
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
        ppo_epochs,
        mini_batch_size,
        states,
        actions,
        log_probs,
        returns,
        advantages,
        clip_param=0.2,
    ):
        """
        Функция для обновления PPO.
        Args:
            ppo_epochs (int): количество эпох.
            mini_batch_size (int): размер мини-батча.
            states (Tensor): батч состояний.
            actions (Tensor): батч действий.
            log_probs (Tensor): батч логарифмов вероятностей действий.
            returns (Tensor): батч отложенных наград.
            advantages (Tensor): батч значений advantage функции.
            clip_param (float, optional): константа для клиппинга.
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

    def learn(self, max_frames, max_reward):
        """
        Функция обучения агента
        Args:
            max_frames (int): максимальное количество шагов в среде.
            max_reward (int): награда для прекращения обучения.
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
