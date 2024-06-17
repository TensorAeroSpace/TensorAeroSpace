import math

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from .utils import push_and_pull, record, set_init, v_wrap


class Net(nn.Module):
    """Нейронная сеть для аппроксимации политики и значения состояний в задачах обучения с подкреплением.

    Args:
        s_dim (int): Размерность пространства состояний.
        a_dim (int): Размерность пространства действий.

    Attributes:
        s_dim (int): Размерность пространства состояний.
        a_dim (int): Размерность пространства действий.
        a1 (torch.nn.Linear): Первый слой политики.
        mu (torch.nn.Linear): Слой для среднего значения распределения политики.
        sigma (torch.nn.Linear): Слой для стандартного отклонения распределения политики.
        c1 (torch.nn.Linear): Первый слой функции значения.
        v (torch.nn.Linear): Выход слоя функции значения.
        distribution (torch.distributions.Distribution): Распределение для моделирования действий агента.
    """
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 256)
        self.mu = nn.Linear(256, a_dim)
        self.sigma = nn.Linear(256, a_dim)
        self.c1 = nn.Linear(s_dim, 256)
        self.v = nn.Linear(256, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        """Выполняет один шаг прямого распространения.

        Args:
            x (torch.Tensor): Входные данные, состояние среды.

        Returns:
            tuple: Возвращает предсказанные значения mu, sigma и value для данного состояния.
        """
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        """Выбор действия агента на основе текущего состояния.

        Args:
            s (torch.Tensor): Текущее состояние среды.

        Returns:
            numpy.ndarray: Выбранное действие.
        """
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
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

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    """Класс рабочего процесса для асинхронного обучения агента.

    Attributes:
        name (str): Уникальное имя процесса.
        g_ep (multiprocessing.Value): Глобальный счётчик эпизодов.
        g_ep_r (multiprocessing.Value): Глобальный счётчик суммарного вознаграждения.
        res_queue (multiprocessing.Queue): Очередь для результатов.
        gnet (torch.nn.Module): Глобальная нейронная сеть.
        opt (torch.optim.Optimizer): Оптимизатор для обновления глобальной сети.
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
        global_ep_r (multiprocessing.Value): Счётчик суммарного вознаграждения по всем процессам.
        res_queue (multiprocessing.Queue): Очередь для хранения результатов.
        name (int): Номер процесса.
        num_actions (int): Количество возможных действий в среде.
        num_observations (int): Количество наблюдений (переменных состояния) в среде.
        MAX_EP (int): Максимальное количество эпизодов.
        MAX_EP_STEP (int): Максимальное количество шагов в каждом эпизоде.
        GAMMA (float): Коэффициент дисконтирования будущих вознаграждений.
        update_global_iter (int): Частота обновления глобальной модели.
    """
    def __init__(self, env, gnet, opt, global_ep, global_ep_r, res_queue, name, num_actions, num_observations, MAX_EP, MAX_EP_STEP, GAMMA, update_global_iter):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(num_observations, num_actions)           # local network
        self.env = env
        self.gamma = GAMMA
        self.max_ep = MAX_EP
        self.max_ep_step = MAX_EP_STEP
        self.update_global_iter = update_global_iter

    def run(self):
        """Выполнение рабочего процесса, содержащего обучение агента."""
        total_step = 1
        while self.g_ep.value < self.max_ep:
            s, info = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(self.max_ep_step):
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, terminated, trunkated, info = self.env.step(a.clip(-2, 2))
                done = terminated or trunkated
                if t == self.max_ep_step - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)    # normalize

                if total_step % self.update_global_iter == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)
