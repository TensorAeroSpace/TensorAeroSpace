import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def clip_grad_norm_(module, max_grad_norm):
    """
    Обрезает градиенты параметров модуля для предотвращения "взрыва градиентов".

    Args:
        module (torch.nn.Module): Модуль, градиенты параметров которого необходимо обрезать.
        max_grad_norm (float): Максимальная норма градиента.

    Returns:
        None
    """
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def mish(input):
    """
    Применяет функцию активации Mish к входным данным.

    Args:
        input (Tensor): Входные данные для функции активации.

    Returns:
        Tensor: Результат применения функции активации Mish.
    """
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    """
    Модуль PyTorch, реализующий функцию активации Mish.
    """
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

def t(x):
    """
    Преобразует входные данные в тензор PyTorch типа float.

    Args:
        x (array-like или torch.Tensor): Входные данные для преобразования.

    Returns:
        torch.Tensor: Преобразованный тензор PyTorch.
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class Actor(nn.Module):
    """
    Модуль PyTorch, реализующий актора для алгоритмов актор-критик.

    Args:
        state_dim (int): Размерность пространства состояний.
        n_actions (int): Количество действий.
        activation (torch.nn.Module): Функция активации.
    """
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        """
        Инициализирует экземпляр класса Actor.
        """
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    def forward(self, X):
        """
        Выполняет прямой проход модели актора.

        Args:
            X (Tensor): Входные данные, представляющие состояние среды.

        Returns:
            torch.distributions.Normal: Нормальное распределение, представляющее политику действий.
        """
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(),  1e-3, 50)
        
        return torch.distributions.Normal(means, stds)
    
class Critic(nn.Module):
    """
    Модуль PyTorch, реализующий критика для алгоритмов актор-критик.

    Args:
        state_dim (int): Размерность пространства состояний.
        activation (torch.nn.Module): Функция активации.
    """
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim+state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )
    def forward(self, X):
        """
        Выполняет прямой проход модели критика.

        Args:
            X (Tensor): Входные данные, представляющие состояние среды.

        Returns:
            Tensor: Оценка значения состояния.
        """
        return self.model(X)
    
def discounted_rewards(rewards, dones, gamma):
    """
    Расчет дисконтированных вознаграждений для последовательности вознаграждений с учетом фактора завершения эпизода.
    
    Args:
        rewards (list[float]): Список полученных вознаграждений.
        dones (list[bool]): Список булевых значений, указывающих, является ли соответствующее вознаграждение последним в эпизоде.
        gamma (float): Коэффициент дисконтирования.

    Returns:
        list[float]: Список дисконтированных вознаграждений.
    """
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]

def process_memory_narx(memory, gamma=0.99, discount_rewards=True):
    """
    Обработка памяти для агента с использованием модели NARX (Nonlinear AutoRegressive with eXogenous inputs).
    Преобразует сохраненные в памяти взаимодействия в формат, подходящий для обучения модели.

    Args:
        memory (list[tuple]): Список кортежей вида (действие, вознаграждение, состояние, следующее состояние, завершено).
        gamma (float, optional): Коэффициент дисконтирования для расчета дисконтированных вознаграждений. По умолчанию 0.99.
        discount_rewards (bool, optional): Флаг, указывающий на необходимость дисконтирования вознаграждений. По умолчанию True.

    Returns:
        tuple: Кортеж, содержащий обработанные действия, вознаграждения, состояния, следующие состояния, флаги завершения и критические состояния.
    """
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    critic_states = []  # Инициализация для хранения состояний и предыдущих действий

    # Используем None или 0 как заполнитель для предыдущего действия первого состояния
    prev_state = np.zeros(memory[0][2].shape)
    prev_next_state = np.zeros(memory[0][2].shape)
    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(np.concatenate((next_state.flatten(), prev_next_state)))
        dones.append(done)
        # Добавляем текущее состояние и предыдущее действие в hist_values
        critic_states.append(np.concatenate((state.flatten(), prev_state)))
        prev_state = state.flatten()  # Обновляем предыдущее действие для следующей итерации
        prev_next_state =  next_state.flatten()  #
    if discount_rewards:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).view(-1, 1)
    states = t(states)
    next_states = t(next_states)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)
    critic_states = t(critic_states)  # Преобразование списка в тензор

    return actions, rewards, states, next_states, dones, critic_states




class A2CLearner():
    """
    Класс, реализующий процесс обучения агента с использованием алгоритма Actor-Critic с функцией преимущества (A2C).
    
    Args:
        actor (torch.nn.Module): модель актера, определяющая политику действий агента.
        critic (torch.nn.Module): модель критика, оценивающая стоимость состояний.
        gamma (float, optional): коэффициент дисконтирования. По умолчанию равен 0.9.
        entropy_beta (float, optional): коэффициент для регулирования энтропии в функции потерь актера. По умолчанию равен 0.01.
        actor_lr (float, optional): скорость обучения для оптимизатора актера. По умолчанию равна 4e-4.
        critic_lr (float, optional): скорость обучения для оптимизатора критика. По умолчанию равна 4e-3.
        max_grad_norm (float, optional): максимальная норма градиента для обрезки. По умолчанию равна 0.5.
    """
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0.01,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.writer  = SummaryWriter()

    
    def learn(self, memory, steps, discount_rewards=True):
        """
        Функция обучения, использующая собранные в памяти взаимодействия для обновления моделей актера и критика.
        
        Args:
            memory (list): список взаимодействий среды, содержащих состояния, действия, вознаграждения и т.д.
            steps (int): текущий шаг обучения, используется для логирования.
            discount_rewards (bool, optional): флаг для использования дисконтированных вознаграждений. По умолчанию True.
        """
        actions, rewards, states, next_states, dones, critic_states = process_memory_narx(memory, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma*self.critic(next_states)*(1-dones)
        value = self.critic(critic_states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.writer.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.writer.add_histogram("parameters/actor",
                             torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.writer.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.writer.add_histogram("parameters/critic",
                             torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()
        
        # reports
        self.writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
        self.writer.add_scalar("losses/entropy", entropy, global_step=steps) 
        self.writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps) 
        self.writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        self.writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        self.writer.add_scalar("losses/critic", critic_loss, global_step=steps)


class Runner():
    """
    Класс для выполнения взаимодействия агента с средой и сбора данных обучения.
    
    Args:
        env (gym.Env): среда, с которой взаимодействует агент.
        actor (torch.nn.Module): модель актера, используемая для выбора действий.
        writer (SummaryWriter): объект для логирования в TensorBoard.
    """
    def __init__(self, env, actor, writer):
        self.env = env
        self.actor = actor
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        # Initialize previous action as zeros; adjust the size based on your action space
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.writer = writer

    def reset(self):
        """
        Сброс среды и внутренних переменных перед началом нового эпизода.
        """
        self.episode_reward = 0
        self.done = False
        self.state, info = self.env.reset()
        # Reset previous action at the start of each episode
        self.prev_action = np.zeros(self.env.action_space.shape)
    
    def run(self, max_steps, memory=None):
        """
        Выполнение заданного числа шагов в среде для сбора данных обучения.
        
        Args:
            max_steps (int): максимальное количество шагов в среде за один вызов функции.
            memory (list, optional): список для сохранения взаимодействий. Если None, будет создан новый список.
            
        Returns:
            list: собранные взаимодействия среды.
        """
        if not memory: 
            memory = []
        
        for i in range(max_steps):
            if self.done: 
                self.reset()
            
            dists = self.actor(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0))
            actions = dists.sample().detach().numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            next_state, reward, terminated, truncated, info = self.env.step(actions_clipped[0])
            self.done = terminated or truncated
            
            # Here, instead of just the state, we store the state concatenated with the previous action
            memory.append((actions, reward, self.state, next_state, self.done))

            self.prev_action = actions_clipped[0]  # Update the previous action
            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                # Assuming writer is defined and configured globally
                self.writer.add_scalar("episode_reward", self.episode_reward, global_step=self.steps)
                    
        return memory
