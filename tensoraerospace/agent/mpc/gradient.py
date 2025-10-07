import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import BaseRLModel


def initialize_tensor(
    size: torch.Size = torch.Size([1, 1]),
    min_val: float | None = None,
    max_val: float | None = None,
) -> torch.Tensor:
    # Handle None values and prevent division by zero
    if min_val is None and max_val is None:
        # Default initialization with standard normal distribution
        tensor = torch.randn(size, requires_grad=True)
    elif min_val is None:
        # Only max_val is specified
        mean = max_val - 1.0  # Arbitrary offset
        std_dev = 1.0
        tensor = torch.normal(mean, std_dev, size, requires_grad=True)
        tensor = torch.clamp(tensor, max=max_val)
    elif max_val is None:
        # Only min_val is specified
        mean = min_val + 1.0  # Arbitrary offset
        std_dev = 1.0
        tensor = torch.normal(mean, std_dev, size, requires_grad=True)
        tensor = torch.clamp(tensor, min=min_val)
    else:
        # Both values are specified
        if abs(max_val - min_val) < 1e-8:  # Prevent division by zero
            # If min and max are essentially equal, return constant tensor
            tensor = torch.full(size, (min_val + max_val) / 2, requires_grad=True)
        else:
            mean = (max_val + min_val) / 2
            std_dev = (
                max_val - min_val
            ) / 4  # 4 standard deviations to cover ~95% of values
            tensor = torch.normal(mean, std_dev, size, requires_grad=True)
            tensor = torch.clamp(tensor, min=min_val, max=max_val)

    return tensor


class Net(nn.Module):
    """Create neural network for system dynamics modeling.

    Network consists of three linear layers and ReLU activation functions between them.
    Input layer accepts vector of 3 elements representing system states.
    Second and third layers are hidden layers with 128 neurons.
    Output layer generates vector of 2 elements representing prediction of next system state.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # 3 states + 1 action = 4
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Next state prediction

    def forward(self, x):
        """Perform forward propagation of input data through network.

        Args:
            x (torch.Tensor): Input data representing system states.

        Returns:
            torch.Tensor: Prediction of next system state.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MPCOptimizationAgent(BaseRLModel):
    """
    Agent using Model Predictive Control (MPC) method for action optimization in environment.

    Attributes:
        gamma (float): Discount coefficient.
        action_dim (int): Action space dimension.
        observation_dim (int): Observation space dimension.
        model (torch.nn.Module): Model for environment dynamics approximation.
        cost_function (callable): Cost function used for action evaluation.
        lr (float): Learning rate for model optimizer.
        criterion (torch.nn.modules.loss): Loss criterion for model training.
    """

    def __init__(
        self,
        gamma,
        action_dim,
        observation_dim,
        model,
        cost_function,
        env,
        lr=1e-3,
        criterion=torch.nn.MSELoss(),
        optimization_lr=1,
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
        self.optimization_lr = optimization_lr

    def from_pretrained(self, repo_name, access_token=None, version=None):
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

    def train_transformers_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        """
        Train transformer model of system dynamics using data about states, actions and next states.

        Args:
            states (numpy.ndarray): Array of current states.
            actions (numpy.ndarray): Array of actions performed in these states.
            next_states (numpy.ndarray): Array of next states after performing actions.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            None
        """
        for epoch in (pbar := tqdm(range(epochs))):
            permutation = np.random.permutation(states.shape[0])
            epoch_loss = 0.0
            for i in range(0, states.shape[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_states, batch_actions, batch_next_states = (
                    states[indices],
                    actions[indices],
                    next_states[indices],
                )

                # Combine states and actions into one tensor
                inputs = np.hstack((batch_states, batch_actions.reshape(-1, 1)))
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(batch_next_states, dtype=torch.float32)

                # Transform input data to form (batch_size, sequence_length, embedding_dim)
                inputs = inputs.unsqueeze(1)  # (batch_size, 1, embedding_dim)
                inputs = inputs.transpose(
                    0, 1
                )  # (sequence_length, batch_size, embedding_dim)

                # Zero gradients
                self.system_model_optimizer.zero_grad()

                # Forward propagation through model
                outputs = self.system_model(inputs)

                # Transform output data back
                outputs = outputs.transpose(0, 1).squeeze(1)  # (batch_size, 2)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Backward propagation
                loss.backward()

                # Update model parameters
                self.system_model_optimizer.step()

                # Aggregate loss across batches
                epoch_loss += loss.item()

            # Log average loss value per epoch
            avg_epoch_loss = epoch_loss / (states.shape[0] // batch_size)
            self.writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
            pbar.set_description(f"Avg Loss {avg_epoch_loss:.4f}")

    def train_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        """
        Train environment dynamics model using data about states, actions and next states.

        Args:
            states (numpy.ndarray): Array of current states.
            actions (numpy.ndarray): Array of actions performed in these states.
            next_states (numpy.ndarray): Array of next states after performing actions.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            None
        """
        for epoch in (pbar := tqdm(range(epochs))):
            permutation = np.random.permutation(states.shape[0])
            epoch_loss = 0.0
            num_batches = 0

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

                # Accumulate loss for epoch average
                epoch_loss += loss.item()
                num_batches += 1

            # Log average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
            pbar.set_description(f"Avg Loss {avg_epoch_loss:.4f}")

    def collect_data(
        self, num_episodes: int = 1000, control_exploration_signal=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect data about states, actions and next states by executing random policy in environment.

        Args:
            num_episodes (int): Number of episodes for data collection.

        Returns:
            tuple: Returns tuple of three arrays (states, actions, next_states).
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
                    actions.append(action[0][0])
                    next_states.append(next_state)
                    state = next_state
            return np.array(states), np.array(actions), np.array(next_states)

    def choose_action(
        self, state: np.ndarray, rollout: int, horizon: int
    ) -> np.ndarray:
        """
        Select optimal action using model for prediction and evaluation of action consequences.

        Args:
            state (numpy.ndarray): Current environment state.
            rollout (int): Number of predicted trajectories for evaluation.
            horizon (int): Planning horizon (number of steps ahead for evaluation).

        Returns:
            numpy.ndarray: Returns array containing selected action.
        """
        initial_state = torch.tensor(np.array([state]), dtype=torch.float32)
        best_cost = float("inf")
        best_action_sequence = None

        for _ in range(rollout):
            action_sequence = torch.randn(
                horizon, 1, requires_grad=True
            )  # Initialize action sequence
            optimizer = optim.Adam([action_sequence], lr=1)

            for optimization_step in range(rollout):  # Number of optimization steps
                optimizer.zero_grad()
                state = initial_state
                total_cost = 0
                for h in range(horizon):
                    action = action_sequence[h].unsqueeze(0)
                    next_state = self.system_model(torch.cat([state, action], dim=-1))
                    cost = self.cost_function(next_state, action)
                    total_cost += cost
                    state = next_state

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_action_sequence = action_sequence.detach().clone()

                total_cost.backward()
                optimizer.step()

        return (
            best_action_sequence[0].detach().numpy()
        )  # Возвращаем первое действие из наилучшей последовательности

    def choose_action_ref(
        self,
        state: np.ndarray,
        rollout: int,
        horizon: int,
        reference_signals: np.ndarray,
        step: int,
        optimization_steps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Выбирает оптимальное действие с учетом эталонных сигналов.

        Args:
            state (numpy.ndarray): Текущее состояние среды.
            rollout (int): Количество прогнозируемых траекторий для оценки.
            horizon (int): Горизонт планирования.
            reference_signals (numpy.ndarray): Эталонные сигналы для оценки действий.
            step (int): Текущий временной шаг в среде.

        Returns:
            Tuple[np.ndarray, float]: Пара (действие, значение функции стоимости для лучшего действия).
        """

        initial_state = torch.as_tensor(np.array([state]), dtype=torch.float32)
        for _ in range(rollout):
            best_cost = float("inf")
            best_action_sequence = None
            action_sequence = torch.FloatTensor(horizon, 1).uniform_(-0.43, 0.43)
            # Создание тензора с require_grad=True
            action_sequence = torch.tensor(action_sequence, requires_grad=True)
            optimizer = optim.SGD([action_sequence], lr=self.optimization_lr)
            # print("action_sequence",action_sequence)
            for h in range(horizon):  # Количество шагов оптимизации
                state = initial_state
                total_cost = 0
                for optimization_step in range(optimization_steps):
                    optimizer.zero_grad()
                    action = action_sequence[h].unsqueeze(0)
                    next_state = self.system_model(torch.cat([state, action], dim=-1))
                    cost = self.cost_function(
                        next_state, action, reference_signals, step
                    )

                    # total_cost += cost
                    # state = next_state

                    if cost < best_cost:
                        best_cost = cost
                        best_action_sequence = action_sequence.detach().clone()

                    cost.backward()
                    optimizer.step()
        # print("best",best_action_sequence[0].detach().numpy())
        return (
            best_action_sequence[0].detach().numpy(),
            best_cost,
        )  # Возвращаем первое действие из наилучшей последовательности

    def test_model(
        self, num_episodes: int = 100, rollout: int = 10, horizon: int = 1
    ) -> List[float]:
        """
        Тестирует модель в среде, измеряя среднее вознаграждение за серию эпизодов.

        Args:
            num_episodes (int): Количество эпизодов для тестирования.
            rollout (int): Количество прогнозируемых траекторий для выбора действий.
            horizon (int): Горизонт планирования для выбора действий.

        Returns:
            List[float]: Список суммарных вознаграждений за каждый эпизод.
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
        """
        Тестирует точность предсказаний модели на заданном наборе данных.

        Args:
            states (numpy.ndarray): Массив текущих состояний.
            actions (numpy.ndarray): Массив действий.
            next_states (numpy.ndarray): Массив следующих состояний.

        Returns:
            None
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
        """Получаем параметры параметров среды. Возвращает словарь с параметрами среды."""
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
        """
        Сохраняет модель PyTorch в указанной директории. Если путь не указан,
        создает директорию с текущей датой и временем.

        Args:
            path (str, optional): Путь, где будет сохранена модель. Если None,
                создается директория с текущей датой и временем.

        Returns:
            None
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
        """
        Загружает модель из файла по указанному пути.

        Args:
            path (str): Путь к файлу с моделью.

        Returns:
            None

        """
        path = Path(path)
        path = path / "model.pth"
        self.system_model = torch.load(path, weights_only=False)
        self.system_model.eval()
