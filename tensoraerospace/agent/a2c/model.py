"""
Модуль реализации алгоритма Advantage Actor-Critic (A2C).

Этот модуль содержит реализацию алгоритма A2C для обучения с подкреплением,
включая нейронные сети актора и критика, функции обработки памяти и основной
класс агента A2C для управления аэрокосмическими системами.
"""

import datetime
import json
import shutil
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


def mish(input):
    """Функция активации Mish.

    Mish - это гладкая, непрерывная функция активации, определяемая как:
    f(x) = x * tanh(softplus(x))

    Args:
        input (torch.Tensor): Входной тензор.

    Returns:
        torch.Tensor: Результат применения функции активации Mish.
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """Модуль PyTorch для функции активации Mish.

    Этот класс оборачивает функцию активации Mish в модуль PyTorch,
    что позволяет использовать её в нейронных сетях.
    """

    def __init__(self):
        """Инициализирует модуль Mish."""
        super().__init__()

    def forward(self, input):
        """Прямой проход через функцию активации Mish.

        Args:
            input (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Результат применения функции активации Mish.
        """
        return mish(input)


# Helper function to convert numpy arrays to tensors
def t(x):
    """Преобразует numpy массив в PyTorch тензор.

    Args:
        x: Входные данные (numpy массив или другой тип).

    Returns:
        torch.Tensor: Тензор PyTorch с типом float.
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class Actor(nn.Module):
    """Нейронная сеть актора для алгоритма A2C.

    Актор генерирует политику - распределение вероятностей действий
    для каждого состояния. Использует нормальное распределение для
    непрерывных действий.

    Args:
        state_dim (int): Размерность пространства состояний.
        n_actions (int): Количество действий.
        activation: Функция активации для скрытых слоев. По умолчанию nn.Tanh.

    Attributes:
        n_actions (int): Количество действий.
        model (nn.Sequential): Основная нейронная сеть.
        logstds (nn.Parameter): Логарифмы стандартных отклонений для действий.
    """

    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        """Инициализирует актора.

        Args:
            state_dim (int): Размерность пространства состояний.
            n_actions (int): Количество действий.
            activation: Функция активации для скрытых слоев.
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

        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X):
        """Прямой проход через сеть актора.

        Args:
            X (torch.Tensor): Входные состояния.

        Returns:
            torch.distributions.Normal: Нормальное распределение действий.
        """
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        return torch.distributions.Normal(means, stds)


class Critic(nn.Module):
    """Нейронная сеть критика для алгоритма A2C.

    Критик оценивает ценность состояний, предсказывая ожидаемую
    суммарную награду из данного состояния.

    Args:
        state_dim (int): Размерность пространства состояний.
        activation: Функция активации для скрытых слоев. По умолчанию nn.Tanh.

    Attributes:
        model (nn.Sequential): Основная нейронная сеть.
    """

    def __init__(self, state_dim, activation=nn.Tanh):
        """Инициализирует критика.

        Args:
            state_dim (int): Размерность пространства состояний.
            activation: Функция активации для скрытых слоев.
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
        """Прямой проход через сеть критика.

        Args:
            X (torch.Tensor): Входные состояния.

        Returns:
            torch.Tensor: Оценки ценности состояний.
        """
        return self.model(X)


def discounted_rewards(rewards, dones, gamma):
    """Вычисляет дисконтированные награды для эпизода.

    Args:
        rewards (list): Список наград за каждый шаг.
        dones (list): Список флагов завершения эпизода.
        gamma (float): Коэффициент дисконтирования.

    Returns:
        list: Список дисконтированных наград.
    """
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, gamma=0.99, discount_rewards=True):
    """Обрабатывает память опыта для обучения.

    Args:
        memory (list): Список кортежей (action, reward, state, next_state, done).
        gamma (float): Коэффициент дисконтирования. По умолчанию 0.99.
        discount_rewards (bool): Применять ли дисконтирование наград. По умолчанию True.

    Returns:
        tuple: Кортеж тензоров (actions, rewards, states, next_states, dones).
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

    actions = t(actions).view(-1, 1)
    states = t(states)
    next_states = t(next_states)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)

    return actions, rewards, states, next_states, dones


def clip_grad_norm_(module, max_grad_norm):
    """Обрезает градиенты по норме для стабилизации обучения.

    Args:
        module: Оптимизатор PyTorch.
        max_grad_norm (float): Максимальная норма градиентов.
    """
    nn.utils.clip_grad_norm_(
        [p for g in module.param_groups for p in g["params"]], max_grad_norm
    )


class A2C(BaseRLModel):
    """Реализация алгоритма Advantage Actor-Critic (A2C).

    A2C - это алгоритм обучения с подкреплением, который использует
    актора для выбора действий и критика для оценки состояний.
    Алгоритм минимизирует потери актора и критика одновременно.

    Args:
        env: Среда для обучения.
        actor: Нейронная сеть актора.
        critic: Нейронная сеть критика.
        gamma (float): Коэффициент дисконтирования. По умолчанию 0.9.
        entropy_beta (float): Коэффициент энтропийного бонуса. По умолчанию 0.01.
        actor_lr (float): Скорость обучения актора. По умолчанию 4e-4.
        critic_lr (float): Скорость обучения критика. По умолчанию 4e-3.
        max_grad_norm (float): Максимальная норма градиентов. По умолчанию 0.5.
        seed (int, optional): Семя для воспроизводимости результатов.

    Attributes:
        env: Среда для обучения.
        state: Текущее состояние среды.
        done (bool): Флаг завершения эпизода.
        steps (int): Общее количество шагов.
        episode_reward (float): Награда за текущий эпизод.
        episode_rewards (list): История наград по эпизодам.
        actor: Нейронная сеть актора.
        critic: Нейронная сеть критика.
        gamma (float): Коэффициент дисконтирования.
        entropy_beta (float): Коэффициент энтропийного бонуса.
        actor_optim: Оптимизатор для актора.
        critic_optim: Оптимизатор для критика.
        writer: TensorBoard writer для логирования.
    """

    def __init__(
        self,
        env,
        actor,
        critic,
        gamma=0.9,
        entropy_beta=0.01,
        actor_lr=4e-4,
        critic_lr=4e-3,
        max_grad_norm=0.5,
        seed=None,
    ):
        """Инициализирует агента A2C.

        Args:
            env: Среда для обучения.
            actor: Нейронная сеть актора.
            critic: Нейронная сеть критика.
            gamma (float): Коэффициент дисконтирования.
            entropy_beta (float): Коэффициент энтропийного бонуса.
            actor_lr (float): Скорость обучения актора.
            critic_lr (float): Скорость обучения критика.
            max_grad_norm (float): Максимальная норма градиентов.
            seed (int, optional): Семя для воспроизводимости результатов.
        """
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.seed = seed
        if seed:
            torch.manual_seed(seed)
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.actor = actor
        self.critic = critic

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

    def reset(self):
        """Сбрасывает состояние агента и среды для нового эпизода."""
        self.episode_reward = 0
        self.done = False
        self.state, info = self.env.reset()

    def run_episode(self, max_steps):
        """Выполняет один эпизод взаимодействия со средой.

        Args:
            max_steps (int): Максимальное количество шагов в эпизоде.

        Returns:
            list: Список кортежей (action, reward, state, next_state, done)
                  представляющих опыт взаимодействия со средой.
        """
        memory = []

        for i in range(max_steps):
            if self.done:
                self.reset()

            dists = self.actor(t(self.state))
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(
                actions,
                self.env.action_space.low.min(),
                self.env.action_space.high.max(),
            )

            next_state, reward, terminated, truncated, info = self.env.step(
                actions_clipped
            )
            self.done = terminated or truncated

            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)
                self.writer.add_scalar(
                    "episode_reward", self.episode_reward, global_step=self.steps
                )
                break

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
            memory, self.gamma, discount_rewards
        )

        td_target = (
            rewards
            if discount_rewards
            else rewards + self.gamma * self.critic(next_states) * (1 - dones)
        )
        value = self.critic(states)
        advantage = td_target - value

        # Actor learning
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (
            -logs_probs * advantage.detach()
        ).mean() - entropy * self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()

        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.actor_optim.step()

        # Critic learning
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()

        # Reporting
        self.writer.add_scalar("Loss/Log_probs", -logs_probs.mean(), global_step=steps)
        self.writer.add_scalar("Loss/Entropy", entropy, global_step=steps)
        self.writer.add_scalar(
            "Loss/Entropy_beta", self.entropy_beta, global_step=steps
        )
        self.writer.add_scalar("Loss/Actor", actor_loss, global_step=steps)
        self.writer.add_scalar("Loss/Advantage", advantage.mean(), global_step=steps)
        self.writer.add_scalar(
            "Performance/Reward",
            np.mean(rewards.detach().cpu().numpy()),
            global_step=steps,
        )
        self.writer.add_scalar("Loss/Critic", critic_loss, global_step=steps)

    def train(self, steps_on_memory=32, episodes=2000, episode_length=300):
        """Запускает процесс обучения агента.

        Args:
            steps_on_memory (int): Количество шагов для накопления опыта
                                 перед обучением. По умолчанию 32.
            episodes (int): Общее количество эпизодов обучения. По умолчанию 2000.
            episode_length (int): Максимальная длина эпизода. По умолчанию 300.
        """
        total_steps = (episodes * episode_length) // steps_on_memory

        for i in tqdm(range(total_steps)):
            memory = self.run_episode(steps_on_memory)
            self.learn(memory, self.steps, discount_rewards=False)

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
        # Очистка целевой директории, чтобы избежать конфликтов при тестировании
        path.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                # Игнорируем ошибки удаления отдельных файлов/папок
                pass
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
        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)

    @classmethod
    def __load(cls, path):
        """Загружает модель A2C из указанной директории.

        Args:
            path (str or Path): Путь к директории с сохраненной моделью.

        Returns:
            A2C: Загруженный экземпляр модели A2C.

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
            env = get_class_from_string(config["env"]["name"])(
                **config["env"]["params"]
            )
        else:
            env = get_class_from_string(config["env"]["name"])()
        critic = torch.load(critic_path)
        actor = torch.load(actor_path)
        new_agent = cls(
            env=env, critic=critic, actor=actor, **config["policy"]["params"]
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
