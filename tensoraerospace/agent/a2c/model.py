"""Advantage Actor-Critic (A2C) algorithm implementation module.

This module contains the A2C algorithm implementation for reinforcement learning,
including actor and critic neural networks, memory processing functions and the main
A2C agent class for aerospace system control.
"""

import datetime
import json
import time
from pathlib import Path

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

        logstds_param = nn.Parameter(torch.full((n_actions,), -1.0))
        self.register_parameter("logstds", logstds_param)

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
        """
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []

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

        self.writer = SummaryWriter()

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
        """Собирает опыт взаимодействия со средой на фиксированное количество шагов.

        Метод всегда собирает ровно max_steps шагов, автоматически начиная
        новые эпизоды если предыдущие завершились. Это обеспечивает постоянный
        размер батча для стабильного обучения A2C.

        Args:
            max_steps (int): Количество шагов для сбора опыта.

        Returns:
            list: Список кортежей (action, reward, state, next_state, done)
                  представляющих опыт взаимодействия со средой.
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

            next_state, reward, terminated, truncated, _ = self.env.step(
                actions_clipped
            )
            self.done = terminated or truncated

            memory.append((actions_clipped, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)

                # Логируем награду за эпизод
                self.writer.add_scalar(
                    "Performance/Episode_Reward",
                    self.episode_reward,
                    global_step=self.steps,
                )

                # Логируем скользящее среднее за последние 10 эпизодов
                if len(self.episode_rewards) >= 10:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    self.writer.add_scalar(
                        "Performance/Episode_Reward_Avg_10",
                        avg_reward,
                        global_step=self.steps,
                    )

                # Логируем скользящее среднее за последние 100 эпизодов
                if len(self.episode_rewards) >= 100:
                    avg_reward_100 = np.mean(self.episode_rewards[-100:])
                    self.writer.add_scalar(
                        "Performance/Episode_Reward_Avg_100",
                        avg_reward_100,
                        global_step=self.steps,
                    )

                self.episode_reward = 0

        return memory

    def learn(self, memory, steps, discount_rewards=True):
        """Обучает агента на основе собранного опыта.

        Выполняет один шаг обучения актора и критика, используя
        алгоритм Advantage Actor-Critic.

        Args:
            memory (list): Список опыта взаимодействия со средой.
            steps (int): Текущий номер шага для логирования.
            discount_rewards (bool): Применять ли дисконтирование наград.
                                   По умолчанию True.
        """
        actions, rewards, states, next_states, dones = process_memory(
            memory, self.gamma, discount_rewards, device=self.device
        )

        # Calculate TD target (always detached!)
        if discount_rewards:
            # Monte Carlo return - must detach!
            td_target = rewards.detach()
        else:
            # TD(0) bootstrap - detach to prevent gradient flow
            with torch.no_grad():
                next_value = self.critic(next_states)
            td_target = rewards + self.gamma * next_value * (1 - dones)

        # Critic learning FIRST
        value = self.critic(states)
        critic_loss = F.mse_loss(value, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()

        # Recalculate value with updated critic (no grad for advantage)
        with torch.no_grad():
            value_updated = self.critic(states)
            advantage = td_target - value_updated

            # Normalize advantage for stable learning (critical for A2C!)
            advantage_normalized = (advantage - advantage.mean()) / (
                advantage.std() + 1e-8
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

        # Reporting
        self.writer.add_scalar("Loss/Log_probs", -logs_probs.mean(), global_step=steps)
        self.writer.add_scalar("Loss/Entropy", entropy, global_step=steps)
        self.writer.add_scalar(
            "Loss/Entropy_beta", self.entropy_beta, global_step=steps
        )
        self.writer.add_scalar("Loss/Actor", actor_loss, global_step=steps)
        self.writer.add_scalar("Loss/Critic", critic_loss, global_step=steps)

        # Advantage metrics (для диагностики)
        self.writer.add_scalar(
            "Advantage/Raw_Mean", advantage.mean(), global_step=steps
        )
        self.writer.add_scalar("Advantage/Raw_Std", advantage.std(), global_step=steps)
        self.writer.add_scalar(
            "Advantage/Normalized_Mean", advantage_normalized.mean(), global_step=steps
        )

        # Value and TD target metrics
        self.writer.add_scalar("Value/Mean", value_updated.mean(), global_step=steps)
        self.writer.add_scalar(
            "Value/TD_Target_Mean", td_target.mean(), global_step=steps
        )
        self.writer.add_scalar(
            "Value/Value_Before_Update", value.mean().item(), global_step=steps
        )

        # Policy statistics
        self.writer.add_scalar(
            "Policy/Action_Std", norm_dists.stddev.mean(), global_step=steps
        )

    def train(
        self,
        steps_on_memory=128,
        episodes=2000,
        episode_length=300,
        discount_rewards=True,
        log_freq=10,
        save_freq=None,
        save_path=None,
    ):
        """Train the agent.

        Args:
            steps_on_memory (int): Number of steps to collect before learning.
                Defaults to 128.
            episodes (int): Total number of training episodes. Defaults to 2000.
            episode_length (int): Maximum episode length. Defaults to 300.
            discount_rewards (bool): Whether to use Monte Carlo returns (True)
                or TD(0) (False). Defaults to True (recommended for stability).
            log_freq (int): Frequency of console logging (in iterations).
                Defaults to 10.
            save_freq (int, optional): Frequency of saving checkpoints (in iterations).
                If None, does not save during training.
            save_path (str, optional): Base path for saving checkpoints.
                If None, uses current directory / 'checkpoints'.

        Returns:
            dict: Training statistics including episode rewards.
        """
        total_steps = (episodes * episode_length) // steps_on_memory
        best_reward = -np.inf

        for i in tqdm(range(total_steps), desc="Training"):
            memory = self.run_episode(steps_on_memory)
            self.learn(memory, self.steps, discount_rewards=discount_rewards)

            # Console logging
            if i % log_freq == 0 and len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
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

        return {
            "episode_rewards": self.episode_rewards,
            "total_steps": self.steps,
            "best_reward": best_reward,
        }

    def close(self):
        """Close TensorBoard writer and cleanup resources."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            self.writer = None

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
        """Загружает предобученную модель из локального пути или Hugging Face Hub.

        Args:
            repo_name (str): Имя репозитория или локальный путь к модели.
            access_token (str, optional): Токен доступа для Hugging Face Hub.
            version (str, optional): Версия модели для загрузки.

        Returns:
            A2C: Загруженный экземпляр модели A2C.
        """
        path = Path(repo_name)
        if path.exists():
            new_agent = cls.__load(path)
            return new_agent
        else:
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent


class A2CWithNARXCritic(A2C):
    def __init__(self, *args, history_length: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_length = history_length

    def _build_narx_batch(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # states: (T, state_dim), actions: (T, action_dim)
        return build_narx_features(states, actions, self.history_length)

    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(
            memory, self.gamma, discount_rewards, device=self.device
        )

        # TD target
        if discount_rewards:
            td_target = rewards.detach()
        else:
            with torch.no_grad():
                # for TD(0) with NARX critic we need next-state features; we approximate using same feature builder
                next_features = self._build_narx_batch(next_states, actions)
                next_value = self.critic(next_features)
            td_target = rewards + self.gamma * next_value * (1 - dones)

        # Critic update (with NARX features)
        features = self._build_narx_batch(states, actions)
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
            advantage_normalized = (advantage - advantage.mean()) / (
                advantage.std() + 1e-8
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

        # Logging
        self.writer.add_scalar("Loss/Actor", actor_loss, global_step=steps)
        self.writer.add_scalar("Loss/Critic", critic_loss, global_step=steps)
        self.writer.add_scalar("Advantage/Mean", advantage.mean(), global_step=steps)
        self.writer.add_scalar(
            "Policy/Action_Std", norm_dists.stddev.mean(), global_step=steps
        )
