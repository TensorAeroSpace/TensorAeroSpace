from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

try:  # Prefer gymnasium when available for typing accuracy
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older environments
    import gym

from typing import Callable, Optional, Tuple

from .shared_optim import SharedAdam
from .utils import push_and_pull, record, set_init, v_wrap


class Net(nn.Module):
    """Нейронная сеть для аппроксимации политики и значения состояний
    в задачах обучения с подкреплением.

    Args:
        s_dim (int): Размерность пространства состояний.
        a_dim (int): Размерность пространства действий.

    Attributes:
        s_dim (int): Размерность пространства состояний.
        a_dim (int): Размерность пространства действий.
        a1 (torch.nn.Linear): Первый слой политики.
        mu (torch.nn.Linear): Слой для среднего значения распределения
            политики.
        sigma (torch.nn.Linear): Слой для стандартного отклонения
            распределения политики.
        c1 (torch.nn.Linear): Первый слой функции значения.
        v (torch.nn.Linear): Выход слоя функции значения.
        distribution (torch.distributions.Distribution): Распределение для
            моделирования действий агента.
    """

    def __init__(self, s_dim: int, a_dim: int) -> None:
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 256)
        self.mu = nn.Linear(256, a_dim)
        self.sigma = nn.Linear(256, a_dim)
        self.c1 = nn.Linear(s_dim, 256)
        self.v = nn.Linear(256, 1)
        self.softplus = nn.Softplus()
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Выполняет один шаг прямого распространения.

        Args:
            x (torch.Tensor): Входные данные, состояние среды.

        Returns:
            tuple: Возвращает предсказанные значения mu, sigma и value
            для данного состояния.
        """
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = self.softplus(self.sigma(a1)) + 0.001  # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s: torch.Tensor) -> np.ndarray:
        """Выбор действия агента на основе текущего состояния.

        Args:
            s (torch.Tensor): Текущее состояние среды.

        Returns:
            numpy.ndarray: Выбранное действие.
        """
        self.eval()
        with torch.no_grad():
            mu, sigma, _ = self.forward(s)
            base = self.distribution(mu, sigma)
            dist = torch.distributions.Independent(base, 1) if self.a_dim > 1 else base
            a = dist.sample()
        return a.cpu().numpy().squeeze(0)

    def loss_func(
        self, s: torch.Tensor, a: torch.Tensor, v_t: torch.Tensor
    ) -> torch.Tensor:
        """Вычисляет функцию потерь для обучения сети.

        Args:
            s (torch.Tensor): Состояния.
            a (torch.Tensor): Действия.
            v_t (torch.Tensor): Целевые значения функции состояния.

        Returns:
            torch.Tensor: Значение функции потерь.
        """
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        base = self.distribution(mu, sigma)
        dist = torch.distributions.Independent(base, 1) if self.a_dim > 1 else base
        log_prob = dist.log_prob(a)  # shape: [batch]
        entropy = dist.entropy()  # shape: [batch]
        exp_v = log_prob * td.detach().squeeze(-1) + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss.squeeze(-1)).mean()
        return total_loss


class Worker(mp.Process):
    """Класс рабочего процесса для асинхронного обучения агента.

    Attributes:
        name (str): Уникальное имя процесса.
        g_ep (multiprocessing.Value): Глобальный счётчик эпизодов.
        g_ep_r (multiprocessing.Value): Глобальный счётчик суммарного
            вознаграждения.
        res_queue (multiprocessing.Queue): Очередь для результатов.
        gnet (torch.nn.Module): Глобальная нейронная сеть.
        opt (torch.optim.Optimizer): Оптимизатор для обновления глобальной
            сети.
        lnet (Net): Локальная нейронная сеть.
        env (gym.Env): Среда OpenAI Gym.
        gamma (float): Коэффициент дисконтирования.
        max_ep (int): Максимальное количество эпизодов.
        max_ep_step (int): Максимальное количество шагов в эпизоде.
        update_global_iter (int): Частота обновления глобальной сети.

    Args:
        env (gym.Env): Среда для обучения агента.
        gnet (torch.nn.Module): Глобальная модель для совместного обучения.
        opt (torch.optim.Optimizer): Оптимизатор для глобальной сети.
        global_ep (multiprocessing.Value): Счётчик общего количества эпизодов.
        global_ep_r (multiprocessing.Value): Счётчик суммарного
            вознаграждения по всем процессам.
        res_queue (multiprocessing.Queue): Очередь для хранения
            результатов.
        name (int): Номер процесса.
        num_actions (int): Количество возможных действий в среде.
        num_observations (int): Количество наблюдений (переменных
            состояния) в среде.
        MAX_EP (int): Максимальное количество эпизодов.
        MAX_EP_STEP (int): Максимальное количество шагов в каждом эпизоде.
        GAMMA (float): Коэффициент дисконтирования будущих вознаграждений.
        update_global_iter (int): Частота обновления глобальной модели.
    """

    def __init__(
        self,
        env: gym.Env,
        gnet: Net,
        opt: SharedAdam,
        global_ep: mp.Value,
        global_ep_r: mp.Value,
        res_queue: mp.Queue,
        name: int,
        num_actions: int,
        num_observations: int,
        MAX_EP: int,
        MAX_EP_STEP: int,
        GAMMA: float,
        update_global_iter: int,
        render: bool = False,
        writer: Optional["torch.utils.tensorboard.SummaryWriter"] = None,
        global_step: Optional[mp.Value] = None,
    ) -> None:
        super(Worker, self).__init__()
        self.name = "w%i" % name
        self.g_ep, self.g_ep_r, self.res_queue = (
            global_ep,
            global_ep_r,
            res_queue,
        )
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(num_observations, num_actions)  # local network
        self.env = env
        self.gamma = GAMMA
        self.max_ep = MAX_EP
        self.max_ep_step = MAX_EP_STEP
        self.update_global_iter = update_global_iter
        self.render = render
        self.writer = writer
        self.global_step = global_step

    def run(self) -> None:
        """Выполнение рабочего процесса, содержащего обучение агента."""
        total_step = 1
        # initial sync from global to local to avoid stale params
        self.lnet.load_state_dict(self.gnet.state_dict())
        while self.g_ep.value < self.max_ep:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                s = reset_out[0]
            else:
                s = reset_out
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.0
            for t in range(self.max_ep_step):
                if self.render and self.name == "w0" and hasattr(self.env, "render"):
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                if (
                    hasattr(self.env, "action_space")
                    and hasattr(self.env.action_space, "low")
                    and hasattr(self.env.action_space, "high")
                ):
                    low = self.env.action_space.low
                    high = self.env.action_space.high
                else:
                    low, high = -np.inf, np.inf
                a_clipped = np.clip(a, low, high)
                step_out = self.env.step(a_clipped)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    s_, r, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    s_, r, done, _ = step_out
                if t == self.max_ep_step - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                # use raw rewards
                # normalization strategy should be config-driven
                buffer_r.append(r)

                if (
                    total_step % self.update_global_iter == 0 or done
                ):  # update global and assign to local net
                    # sync and get metrics
                    metrics = push_and_pull(
                        self.opt,
                        self.lnet,
                        self.gnet,
                        done,
                        s_,
                        buffer_s,
                        buffer_a,
                        buffer_r,
                        self.gamma,
                    )

                    # Log training metrics to TensorBoard
                    if self.writer is not None and self.global_step is not None:
                        with self.global_step.get_lock():
                            step = self.global_step.value
                            self.global_step.value += 1
                        self.writer.add_scalar(
                            f"Loss/{self.name}/total",
                            metrics["loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/value",
                            metrics["value_loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/policy",
                            metrics["policy_loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/entropy",
                            metrics["entropy"],
                            step,
                        )

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and record episode
                        record(
                            self.g_ep,
                            self.g_ep_r,
                            ep_r,
                            self.res_queue,
                            self.name,
                            self.writer,
                        )
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


# Defaults for Agent wrapper
DEFAULT_MAX_EP = 10
DEFAULT_MAX_EP_STEP = 200
DEFAULT_GAMMA = 0.99
DEFAULT_UPDATE_GLOBAL_ITER = 10
DEFAULT_LR = 1e-4


def setup_global_params(
    *,
    max_episodes: int = DEFAULT_MAX_EP,
    max_ep_step: int = DEFAULT_MAX_EP_STEP,
    gamma: float = DEFAULT_GAMMA,
    update_global_iter: int = DEFAULT_UPDATE_GLOBAL_ITER,
    lr: float = DEFAULT_LR,
) -> None:
    """Update defaults used by Agent.

    This matches the previous TF API name to ease migration.
    """
    globals_dict = globals()
    globals_dict["DEFAULT_MAX_EP"] = max_episodes
    globals_dict["DEFAULT_MAX_EP_STEP"] = max_ep_step
    globals_dict["DEFAULT_GAMMA"] = gamma
    globals_dict["DEFAULT_UPDATE_GLOBAL_ITER"] = update_global_iter
    globals_dict["DEFAULT_LR"] = lr


class Agent:
    """Simple A3C Agent wrapper around multiprocessing Workers.

    Args:
        env_function: callable that returns a new env for a given worker id.
        gamma: discount factor.
        n_workers: number of worker processes.
        lr: learning rate for SharedAdam.
        max_episodes: total episodes to run per global counter.
        max_ep_step: max steps per episode.
        update_global_iter: frequency to push/pull.
        render: render from worker w0 (optional).

    Note: For unit tests or debugging, set run_in_main=True to avoid
    spawning processes. The single worker will run in the main process.
    """

    def __init__(
        self,
        env_function: Callable[[int], gym.Env],
        gamma: float = DEFAULT_GAMMA,
        n_workers: Optional[int] = None,
        lr: float = DEFAULT_LR,
        max_episodes: int = DEFAULT_MAX_EP,
        max_ep_step: int = DEFAULT_MAX_EP_STEP,
        update_global_iter: int = DEFAULT_UPDATE_GLOBAL_ITER,
        render: bool = False,
        run_in_main: bool = False,
        log_dir: str = "runs/a3c",
    ) -> None:
        self.env_function = env_function
        self.gamma = gamma
        self.n_workers = (
            mp.cpu_count() if not n_workers or n_workers <= 0 else n_workers
        )
        self.lr = lr
        self.max_episodes = max_episodes
        self.max_ep_step = max_ep_step
        self.update_global_iter = update_global_iter
        self.render = render
        self.run_in_main = run_in_main

        # infer spaces
        probe_env = self.env_function(0)
        s_dim = int(probe_env.observation_space.shape[0])
        a_dim = int(probe_env.action_space.shape[0])
        probe_env.close()

        # global net and optimizer
        self.gnet = Net(s_dim, a_dim)
        self.gnet.share_memory()
        self.opt = SharedAdam(self.gnet.parameters(), lr=self.lr)

        # shared counters
        self.global_ep = mp.Value("i", 0)
        self.global_ep_r = mp.Value("d", 0.0)
        self.global_step = mp.Value("i", 0)
        # Queue type annotation kept as string to avoid requiring typing
        # extensions for multiprocessing types.
        self.res_queue: "mp.Queue" = mp.Queue()

        # TensorBoard writer
        from typing import Optional

        self.writer: Optional["torch.utils.tensorboard.SummaryWriter"] = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    def train(self) -> None:
        workers = []
        if self.run_in_main:
            # run a single worker in current process (useful for tests)
            env = self.env_function(0)
            w = Worker(
                env=env,
                gnet=self.gnet,
                opt=self.opt,
                global_ep=self.global_ep,
                global_ep_r=self.global_ep_r,
                res_queue=self.res_queue,
                name=0,
                num_actions=self.gnet.a_dim,
                num_observations=self.gnet.s_dim,
                MAX_EP=self.max_episodes,
                MAX_EP_STEP=self.max_ep_step,
                GAMMA=self.gamma,
                update_global_iter=self.update_global_iter,
                render=self.render,
                writer=self.writer,
                global_step=self.global_step,
            )
            # directly call run without starting a new process
            w.run()
            env.close()
            return

        for i in range(self.n_workers):
            env = self.env_function(i)
            w = Worker(
                env=env,
                gnet=self.gnet,
                opt=self.opt,
                global_ep=self.global_ep,
                global_ep_r=self.global_ep_r,
                res_queue=self.res_queue,
                name=i,
                num_actions=self.gnet.a_dim,
                num_observations=self.gnet.s_dim,
                MAX_EP=self.max_episodes,
                MAX_EP_STEP=self.max_ep_step,
                GAMMA=self.gamma,
                update_global_iter=self.update_global_iter,
                render=self.render if i == 0 else False,
                writer=self.writer,
                global_step=self.global_step,
            )
            w.start()
            workers.append(w)

        # wait workers
        finished = 0
        while finished < self.n_workers:
            r = self.res_queue.get()
            if r is None:
                finished += 1

        for w in workers:
            w.join()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup resources."""
        if self.writer is not None:
            self.writer.close()
