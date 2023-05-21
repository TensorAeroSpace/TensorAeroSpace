import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """Нейронная сеть для аппроксимации функции значения.

    Args:
        num_inputs (int): Количество входных признаков.
        hidden_dim (int): Размерность скрытых слоев.

    Attributes:
        linear1 (nn.Linear): Первый линейный слой.
        linear2 (nn.Linear): Второй линейный слой.
        linear3 (nn.Linear): Третий линейный слой.

    """
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        """Прямой проход нейронной сети.

        Args:
            state (torch.Tensor): Тензор входного состояния.

        Returns:
            torch.Tensor: Тензор выходного значения.

        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    """Нейронная сеть для оценки функции Q.

    Args:
        num_inputs (int): Количество входных признаков.
        num_actions (int): Количество действий.
        hidden_dim (int): Размерность скрытых слоев.

    Attributes:
        linear1 (nn.Linear): Первый линейный слой для Q1.
        linear2 (nn.Linear): Второй линейный слой для Q1.
        linear3 (nn.Linear): Третий линейный слой для Q1.
        linear4 (nn.Linear): Первый линейный слой для Q2.
        linear5 (nn.Linear): Второй линейный слой для Q2.
        linear6 (nn.Linear): Третий линейный слой для Q2.

    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 арха
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 арха
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """Прямой проход нейронной сети для оценки функции Q.

        Args:
            state (torch.Tensor): Тензор входного состояния.
            action (torch.Tensor): Тензор входного действия.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Тензоры Q1 и Q2.

        """
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    """Гауссова политика для алгоритмов обучения с подкреплением.

    Args:
        num_inputs (int): Количество входных признаков.
        num_actions (int): Количество действий.
        hidden_dim (int): Размерность скрытых слоев.
        action_space (Optional[gym.Space]): Пространство действий. По умолчанию None.

    Attributes:
        linear1 (nn.Linear): Первый линейный слой.
        linear2 (nn.Linear): Второй линейный слой.
        mean_linear (nn.Linear): Линейный слой для среднего значения.
        log_std_linear (nn.Linear): Линейный слой для логарифма стандартного отклонения.
        action_scale (torch.Tensor): Масштаб действий.
        action_bias (torch.Tensor): Смещение действий.

    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """Прямой проход нейронной сети для генерации среднего значения и логарифма стандартного отклонения.

        Args:
            state (torch.Tensor): Тензор входного состояния.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Тензоры среднего значения и логарифма стандартного отклонения.

        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """Сэмплирование действия из гауссовой политики.

        Args:
            state (torch.Tensor): Тензор входного состояния.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Тензор действия, логарифм вероятности и среднее значение.

        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # для трюка репараметризации (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        #  Применение ограничения на действия
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        """Перемещение модели на указанное устройство.

        Args:
            device (Union[str, torch.device]): Устройство для перемещения модели.

        Returns:
            GaussianPolicy: Перемещенная модель.

        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """Детерминированная политика для алгоритмов обучения с подкреплением.

    Args:
        num_inputs (int): Количество входных признаков.
        num_actions (int): Количество действий.
        hidden_dim (int): Размерность скрытых слоев.
        action_space (Optional[gym.Space]): Пространство действий. По умолчанию None.

    Attributes:
        linear1 (nn.Linear): Первый линейный слой.
        linear2 (nn.Linear): Второй линейный слой.
        mean (nn.Linear): Линейный слой для среднего значения.
        noise (torch.Tensor): Тензор для добавления шума.
        action_scale (torch.Tensor): Масштаб действий.
        action_bias (torch.Tensor): Смещение действий.

    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # Масштабирование действий
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """Прямой проход нейронной сети для генерации среднего значения.

        Args:
            state (torch.Tensor): Тензор входного состояния.

        Returns:
            torch.Tensor: Тензор среднего значения.

        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        """Сэмплирование действия из детерминированной политики.

        Args:
            state (torch.Tensor): Тензор входного состояния.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Тензор действия, фиктивное значение логарифма вероятности и среднее значение.

        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        """Перемещение модели на указанное устройство.

        Args:
            device (Union[str, torch.device]): Устройство для перемещения модели.

        Returns:
            DeterministicPolicy: Перемещенная модель.

        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
