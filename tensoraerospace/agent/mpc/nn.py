import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Uniform
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import uniform
from tqdm import tqdm


class Net(nn.Module):
    """Создает нейронную сеть для моделирования динамики системы.

    Сеть состоит из трех линейных слоев и функций активации ReLU между ними.
    Входной слой принимает вектор из 3 элементов, представляющих состояния системы.
    Второй и третий слои - это скрытые слои с 128 нейронами.
    Выходной слой генерирует вектор из 2 элементов, представляющих предсказание следующего состояния системы.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # 3 состояния + 1 действие = 4
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Предсказание следующего состояния

    def forward(self, x):
        """Выполняет прямое распространение входных данных через сеть.

        Args:
            x (torch.Tensor): Входные данные, представляющие состояния системы.

        Returns:
            torch.Tensor: Предсказание следующего состояния системы.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)




class MPCAgent(object):
    """
    Агент, использующий метод Модельно-Прогностического Управления (MPC) для оптимизации действий в среде.

    Attributes:
        gamma (float): Коэффициент дисконтирования.
        action_dim (int): Размерность пространства действий.
        observation_dim (int): Размерность пространства наблюдений.
        model (torch.nn.Module): Модель для аппроксимации динамики среды.
        cost_function (callable): Функция стоимости, используемая для оценки действий.
        lr (float): Скорость обучения для оптимизатора модели.
        criterion (torch.nn.modules.loss): Критерий потерь для обучения модели.

    Methods:
        train_model(states, actions, next_states, epochs=100, batch_size=64):
            Обучает модель динамики среды.
        collect_data(env, num_episodes=1000):
            Собирает данные о состояниях, действиях и следующих состояниях, исполняя политику в среде.
        choose_action(state, rollout, horizon):
            Выбирает оптимальное действие, используя прогнозируемое моделирование.
        choose_action_ref(state, rollout, horizon, reference_signals, step):
            Выбирает оптимальное действие с учетом эталонных сигналов.
        test_model(env, num_episodes=100, rollout=10, horizon=1):
            Тестирует модель, измеряя среднее вознаграждение в среде.
        test_network(states, actions, next_states):
            Тестирует точность предсказаний модели на заданном наборе данных.
    """
    def __init__(self, gamma, action_dim, observation_dim, model, cost_function, lr=1e-3, criterion=torch.nn.MSELoss()):
        self.gamma = gamma
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.system_model = model
        self.system_model_optimizer = optim.Adam(self.system_model.parameters(), lr=lr)
        self.cost_function = cost_function
        self.writer = SummaryWriter()
        self.criterion = criterion
    
    def train_model(self, states, actions, next_states, epochs=100, batch_size=64):
        """
        Обучает модель динамики среды, используя данные о состояниях, действиях и следующих состояниях.

        Args:
            states (numpy.ndarray): Массив текущих состояний.
            actions (numpy.ndarray): Массив действий, совершенных в этих состояниях.
            next_states (numpy.ndarray): Массив следующих состояний после совершения действий.
            epochs (int): Количество эпох обучения.
            batch_size (int): Размер батча для обучения.

        Returns:
            None
        """
        for epoch in  (pbar := tqdm(range(epochs))):
            permutation = np.random.permutation(states.shape[0])
            for i in range(0, states.shape[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_states, batch_actions, batch_next_states = states[indices], actions[indices], next_states[indices]
                inputs = np.hstack((batch_states, batch_actions.reshape(-1, 1)))
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(batch_next_states, dtype=torch.float32)
                self.system_model_optimizer.zero_grad()
                outputs = self.system_model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.system_model_optimizer.step()
            
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            pbar.set_description(f"Loss {loss.item()}")

    def collect_data(self, env, num_episodes=1000):
        """
        Собирает данные о состояниях, действиях и следующих состояниях, исполняя случайную политику в среде.

        Args:
            env (gym.Env): Среда, в которой собираются данные.
            num_episodes (int): Количество эпизодов для сбора данных.

        Returns:
            tuple: Возвращает кортеж из трех массивов (states, actions, next_states).
        """
        states, actions, next_states = [], [], []
        for _ in tqdm(range(num_episodes)):
            state, info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                state = next_state
        return np.array(states), np.array(actions), np.array(next_states)

    def choose_action(self, state, rollout, horizon):
        """
        Выбирает оптимальное действие, используя модель для прогнозирования и оценки последствий действий.

        Args:
            state (numpy.ndarray): Текущее состояние среды.
            rollout (int): Количество прогнозируемых траекторий для оценки.
            horizon (int): Горизонт планирования (количество шагов вперед для оценки).

        Returns:
            numpy.ndarray: Возвращает массив, содержащий выбранное действие.
        """
        initial_state = torch.tensor([state], dtype=torch.float32)
        best_action = None
        max_trajectory_value = -float('inf')
        action_distribution = uniform(loc=-2, scale=4)  # Assuming a continuous action space for simplicity

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

    def choose_action_ref(self, state, rollout, horizon, reference_signals, step):
        """
        Выбирает оптимальное действие с учетом эталонных сигналов.

        Args:
            state (numpy.ndarray): Текущее состояние среды.
            rollout (int): Количество прогнозируемых траекторий для оценки.
            horizon (int): Горизонт планирования.
            reference_signals (numpy.ndarray): Эталонные сигналы для оценки действий.
            step (int): Текущий временной шаг в среде.

        Returns:
            numpy.ndarray: Возвращает массив, содержащий выбранное действие.
        """
        initial_state = torch.tensor([state], dtype=torch.float32)
        best_action = None
        max_trajectory_value = -float('inf')
        action_distribution = Uniform(-60, 60)
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
            if trajectory_value > max_trajectory_value:
                max_trajectory_value = trajectory_value
                best_action = first_action
        return best_action.numpy()
    
    def test_model(self, env, num_episodes=100, rollout=10, horizon=1):
        """
        Тестирует модель в среде, измеряя среднее вознаграждение за серию эпизодов.

        Args:
            env (gym.Env): Среда для тестирования.
            num_episodes (int): Количество эпизодов для тестирования.
            rollout (int): Количество прогнозируемых траекторий для выбора действий.
            horizon (int): Горизонт планирования для выбора действий.

        Returns:
            list: Список суммарных вознаграждений за каждый эпизод.
        """
        total_rewards = []  # Список для хранения суммарных вознаграждений за каждый эпизод
        for episode in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state, rollout, horizon)
                state, reward, terminated, truncated, info= env.step(action[0])
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
            print(f'Episode {episode+1}: Total Reward = {total_reward}')
            total_rewards.append(total_reward)

        average_reward = sum(total_rewards) / num_episodes
        self.writer.add_scalar('Test/AverageReward', average_reward, num_episodes)
        return total_rewards
    
    def test_network(self, states, actions, next_states):
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
            mse_loss = torch.nn.functional.mse_loss(predicted_next_states, true_next_states)
            print(f'Test MSE Loss: {mse_loss.item()}')
            
            # Логирование потерь в TensorBoard
            self.writer.add_scalar('Test/MSE_Loss', mse_loss.item(), 0)
        
        self.system_model.train()  # Вернуть модель в режим обучения
