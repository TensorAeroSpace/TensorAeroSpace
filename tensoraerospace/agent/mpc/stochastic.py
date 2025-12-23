"""Stochastic MPC agents and training utilities.

This module contains components used for stochastic MPC approaches in
TensorAeroSpace, including training/evaluation helpers and logging utilities.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import uniform
from torch.distributions import Uniform
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import BaseRLModel


class Net(nn.Module):
    """Simple feed-forward dynamics model.

    The network is a small MLP with dropout that predicts the next state from a
    concatenation of the current state and action.

    Args:
        num_action: Action dimension.
        num_states: State dimension.
    """

    def __init__(self, num_action, num_states):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(
            num_action + num_states, 16
        )  # 3 состояния + 1 действие = 4
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, num_states)  # Предсказание следующего состояния
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer with a probability of 0.5

    def forward(self, x):
        """Run a forward pass.

        Args:
            x (torch.Tensor): Input tensor (state-action features).

        Returns:
            torch.Tensor: Predicted next state.
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class MPCAgent(BaseRLModel):
    """Stochastic MPC agent using a learned dynamics model.

    The agent samples candidate actions (or action sequences) from a distribution,
    rolls them out through a learned system model, and selects the best action
    according to a provided cost function.

    Args:
        gamma: Discount factor.
        action_dim: Action space dimension.
        observation_dim: Observation/state dimension.
        model: Dynamics model approximator.
        cost_function: Cost function used to evaluate rollouts.
        env: Environment instance used for data collection/testing.
        min_max_action_value: Tuple ``(min_action, max_action)`` used for sampling
            actions in ``choose_action_ref``.
        lr: Learning rate for the dynamics model optimizer.
        criterion: Loss function used to train the dynamics model.
    """

    def __init__(
        self,
        gamma,
        action_dim,
        observation_dim,
        model,
        cost_function,
        env,
        min_max_action_value=(-0.5, 0.5),
        lr=1e-3,
        criterion=torch.nn.MSELoss(),
    ):
        self.gamma = gamma
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.system_model = model
        self.lr = lr
        self.system_model_optimizer = optim.Adam(self.system_model.parameters(), lr=lr)
        self.cost_function = cost_function
        self.writer = SummaryWriter()
        self.criterion = criterion
        self.env = env
        self.min_action, self.max_action = min_max_action_value

    def from_pretrained(self, repo_name, access_token=None, version=None):
        """Load a pretrained dynamics model from the Hugging Face Hub.

        Args:
            repo_name: Repository name on the Hub.
            access_token: Optional access token.
            version: Optional revision/tag/commit.

        Raises:
            ValueError: If the environment name stored in the config does not match
                the agent's current environment.
        """
        folder_path = super().from_pretrained(repo_name, access_token, version)
        self.system_model = torch.load(
            os.path.join(folder_path, "model.pth"), weights_only=False
        )
        config_path = Path(folder_path)
        config_path = config_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        if config["env"]["name"] != self.env.unwrapped.__class__.__name__:
            raise ValueError(
                "Environment name in config.json does not match the environment passed to the model."
            )

    def train_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        """Train the dynamics model on (s, a) -> s' transitions.

        Args:
            states (np.ndarray): Current states.
            actions (np.ndarray): Actions taken in those states.
            next_states (np.ndarray): Next states observed after actions.
            epochs (int): Number of training epochs. Defaults to ``100``.
            batch_size (int): Mini-batch size. Defaults to ``64``.
        """
        for epoch in (pbar := tqdm(range(epochs))):
            permutation = np.random.permutation(states.shape[0])
            for i in range(0, states.shape[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_states, batch_actions, batch_next_states = (
                    states[indices],
                    actions[indices],
                    next_states[indices],
                )
                inputs = np.hstack((batch_states, batch_actions.reshape(-1, 1)))
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(batch_next_states, dtype=torch.float32)
                self.system_model_optimizer.zero_grad()
                outputs = self.system_model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.system_model_optimizer.step()

            self.writer.add_scalar("Loss/train", loss.item(), epoch)
            pbar.set_description(f"Loss {loss.item()}")

    def collect_data(
        self, num_episodes: int = 1000, control_exploration_signal=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect transition data by executing a policy in the environment.

        If ``control_exploration_signal`` is provided, actions are taken from that
        sequence; otherwise actions are sampled from the environment's action space.

        Args:
            num_episodes (int): Number of episodes to collect.
            control_exploration_signal: Optional sequence of actions to replay.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(states, actions, next_states)``.
        """
        if control_exploration_signal is not None:
            states, actions, next_states = [], [], []
            for _ in tqdm(range(num_episodes)):
                state, info = self.env.reset()
                done = False
                index_exp_signal = 0
                while not done:
                    action = control_exploration_signal[index_exp_signal]
                    # action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, info = self.env.step(
                        [action]
                    )
                    done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    state = next_state
                    index_exp_signal += 1
            return np.array(states), np.array(actions), np.array(next_states)
        else:
            states, actions, next_states = [], [], []
            for _ in tqdm(range(num_episodes)):
                state, info = self.env.reset()
                done = False
                while not done:
                    action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    state = next_state
            return np.array(states), np.array(actions), np.array(next_states)

    def choose_action(
        self, state: np.ndarray, rollout: int, horizon: int
    ) -> np.ndarray:
        """Choose an action by Monte-Carlo rollout using the learned model.

        Args:
            state (np.ndarray): Current environment state.
            rollout (int): Number of sampled trajectories to evaluate.
            horizon (int): Planning horizon (number of steps to roll out).

        Returns:
            np.ndarray: Selected action (as numpy array).
        """
        # state = torch.from_numpy(state, dtype=torch.float32)
        initial_state = torch.as_tensor(np.array([state]), dtype=torch.float32)
        best_action = None
        max_trajectory_value = -float("inf")
        action_distribution = uniform(
            loc=-2, scale=4
        )  # Assuming a continuous action space for simplicity

        for trajectory in range(rollout):
            state = initial_state
            trajectory_value = 0
            for h in range(horizon):
                action = torch.Tensor([[action_distribution.rvs()]])
                if h == 0:
                    first_action = action
                next_state = self.system_model(torch.cat([state, action], dim=-1))
                costs = self.cost_function(next_state, action)
                trajectory_value += -costs

                state = next_state
            if trajectory_value > max_trajectory_value:
                max_trajectory_value = trajectory_value
                best_action = first_action
        return best_action.numpy()

    def choose_action_ref(
        self,
        state: np.ndarray,
        rollout: int,
        horizon: int,
        reference_signals: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, float]:
        """Choose an action using reference signals in the cost function.

        Args:
            state (np.ndarray): Current environment state.
            rollout (int): Number of sampled trajectories to evaluate.
            horizon (int): Planning horizon.
            reference_signals (np.ndarray): Reference signals used by the cost function.
            step (int): Current time step index.

        Returns:
            Tuple[np.ndarray, float]: ``(best_action, best_value)``.
        """
        initial_state = torch.tensor([state], dtype=torch.float32)
        best_action = None
        max_trajectory_value = float("inf")
        action_distribution = Uniform(self.min_action, self.max_action)
        for trajectory in range(rollout):
            state = initial_state
            trajectory_value = 0
            for h in range(horizon):
                action = torch.Tensor([[action_distribution.sample()]])
                if h == 0:
                    first_action = action
                next_state = self.system_model(torch.cat([state, action], dim=-1))
                costs = self.cost_function(next_state, action, reference_signals, step)
                trajectory_value += -costs

                state = next_state
            if trajectory_value < max_trajectory_value:
                max_trajectory_value = trajectory_value
                best_action = first_action
        return best_action.numpy(), max_trajectory_value

    def test_model(
        self, num_episodes: int = 100, rollout: int = 10, horizon: int = 1
    ) -> List[float]:
        """Evaluate the agent in the environment for a number of episodes.

        Args:
            num_episodes (int): Number of evaluation episodes.
            rollout (int): Number of rollouts per decision step.
            horizon (int): Planning horizon for action selection.

        Returns:
            List[float]: Total reward per episode.
        """
        total_rewards = (
            []
        )  # Список для хранения суммарных вознаграждений за каждый эпизод
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state, rollout, horizon)
                state, reward, terminated, truncated, info = self.env.step(action[0])
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
            print(f"Episode {episode+1}: Total Reward = {total_reward}")
            total_rewards.append(total_reward)

        average_reward = sum(total_rewards) / num_episodes
        self.writer.add_scalar("Test/AverageReward", average_reward, num_episodes)
        return total_rewards

    def test_network(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> None:
        """Evaluate dynamics model prediction accuracy on a dataset.

        Args:
            states (np.ndarray): Current states.
            actions (np.ndarray): Actions.
            next_states (np.ndarray): Ground-truth next states.
        """
        self.system_model.eval()  # Перевести модель в режим оценки
        with torch.no_grad():  # Отключить вычисление градиентов
            # Подготовка данных
            inputs = np.hstack((states, actions.reshape(-1, 1)))
            inputs = torch.tensor(inputs, dtype=torch.float32)
            true_next_states = torch.tensor(next_states, dtype=torch.float32)

            # Получение предсказаний от модели
            predicted_next_states = self.system_model(inputs)

            # Вычисление потерь (среднеквадратичная ошибка)
            mse_loss = torch.nn.functional.mse_loss(
                predicted_next_states, true_next_states
            )
            print(f"Test MSE Loss: {mse_loss.item()}")

            # Логирование потерь в TensorBoard
            self.writer.add_scalar("Test/MSE_Loss", mse_loss.item(), 0)

        self.system_model.train()  # Вернуть модель в режим обучения

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Return a serializable dictionary describing env/agent parameters."""
        env_name = self.env.unwrapped.__class__.__name__
        agent_name = self.__class__.__name__
        env_params = {}

        # Получение информации о сигнале справки, если она доступна
        try:
            ref_signal = self.env.ref_signal.__class__.__name__
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
            "lr": self.lr,
            "gamma": self.gamma,
            "cost_function": self.cost_function.__name__,
            "model": self.system_model.__class__.__name__,
        }
        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(self, path: str | os.PathLike | None = None) -> None:
        """Save the dynamics model and configuration to disk.

        If ``path`` is not provided, uses the current working directory and
        creates a timestamped subdirectory.

        Args:
            path: Directory to save into. If None, uses ``Path.cwd()``.
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
        path = path / date_str / "model.pth"

        # Создание директории, если она не существует
        path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)
        torch.save(self.system_model, path)

    def load(self, path: str | os.PathLike) -> None:
        """Load a saved dynamics model from disk.

        Args:
            path: Directory containing ``model.pth``.
        """
        path = Path(path)
        path = path / "model.pth"
        self.system_model = torch.load(path, weights_only=False)
        self.system_model.eval()
