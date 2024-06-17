import datetime
import json
import os
from pathlib import Path

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
    """Создает нейронную сеть для моделирования динамики системы.

    Сеть состоит из трех линейных слоев и функций активации ReLU между ними.
    Входной слой принимает вектор из 3 элементов, представляющих состояния системы.
    Второй и третий слои - это скрытые слои с 128 нейронами.
    Выходной слой генерирует вектор из 2 элементов, представляющих предсказание следующего состояния системы.
    """
    def __init__(self, num_action, num_states):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_action+num_states, 128)  # 3 состояния + 1 действие = 4
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_states)  # Предсказание следующего состояния

    def forward(self, x):
        """Выполняет прямое распространение входных данных через сеть.

        Args:
            x (torch.Tensor): Входные данные, представляющие состояния системы.

        Returns:
            torch.Tensor: Предсказание следующего состояния системы.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)




class MPCAgent(BaseRLModel):
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
    """
    def __init__(self, gamma, action_dim, observation_dim, model, cost_function, env, min_max_action_value=(-0.5,0.5), lr=1e-3, criterion=torch.nn.MSELoss()):
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
        folder_path = super().from_pretrained(repo_name, access_token, version)
        self.system_model = torch.load(os.path.join(folder_path,'model.pth'))
        config_path = Path(folder_path)
        config_path = config_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        if config['env']['name'] != self.env.unwrapped.__class__.__name__:
            raise ValueError("Environment name in config.json does not match the environment passed to the model.")
                                
         
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

    def collect_data(self, num_episodes=1000, control_exploration_signal=None):
        """
        Собирает данные о состояниях, действиях и следующих состояниях, исполняя случайную политику в среде.

        Args:
            num_episodes (int): Количество эпизодов для сбора данных.

        Returns:
            tuple: Возвращает кортеж из трех массивов (states, actions, next_states).
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
                    next_state, reward, terminated, truncated, info = self.env.step([action])
                    done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    state = next_state
                    index_exp_signal+=1
            return np.array(states), np.array(actions), np.array(next_states)
        else:
            states, actions, next_states = [], [], []
            for _ in tqdm(range(num_episodes)):
                state, info = self.env.reset()
                done = False
                while not done:
                    action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, info = self.env.step(action)
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
        # state = torch.from_numpy(state, dtype=torch.float32)
        initial_state = torch.as_tensor(np.array([state]), dtype=torch.float32)
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
        max_trajectory_value = float('inf')
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
    
    def test_model(self, num_episodes=100, rollout=10, horizon=1):
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
            state, info = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state, rollout, horizon)
                state, reward, terminated, truncated, info= self.env.step(action[0])
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


    def get_param_env(self):
        """Получаем параметры параметров среды. Возвращает словарь с параметрами среды.
        """
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
            "model": self.system_model.__class__.__name__
            
        }
        return {
            "env":{
                "name":env_name,
                "params":env_params
                } ,
            "policy":{
                "name":agent_name,
                "params":policy_params
                
            }
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
        # Текущая дата и время в формате 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        date_str =date_str+"_"+self.__class__.__name__
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


    def load(self, path):
        """ 
        Загружает модель из файла по указанному пути.

        Args: 
            path (str): Путь к файлу с моделью.

        Returns: 
            None 
        
        """
        path = Path(path)
        path = path / "model.pth"
        self.system_model = torch.load(path)
        self.system_model.eval()


