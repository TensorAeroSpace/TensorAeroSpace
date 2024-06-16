import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from .model import DeterministicPolicy, GaussianPolicy, QNetwork
from .replay_memory import ReplayMemory
from .utils import hard_update, soft_update


class SAC(BaseRLModel):
    """Soft Actor-Critic (SAC) алгоритм для обучения с подкреплением.

    Args:
        num_inputs (int): Размерность входного пространства состояний.
        action_space: Пространство действий агента.
        args: Параметры и настройки алгоритма.

        gamma (float): Коэффициент дисконтирования.
        tau (float): Коэффициент для мягкого обновления весов целевой сети.
        alpha (float): Коэффициент для регуляризации политики.
        policy_type (str): Тип политики ("Gaussian" или "Deterministic").
        target_update_interval (int): Интервал обновления весов целевой сети.
        automatic_entropy_tuning (bool): Флаг автоматической настройки энтропии.
        cuda: Использовать cuda или нет.
    
    Attributes:

        critic: Сеть критика.
        critic_optim: Оптимизатор для обновления весов критика.
        critic_target: Целевая сеть критика.

        policy: Политика агента.
        policy_optim: Оптимизатор для обновления весов политики.

    """
    def __init__(self, env, updates_per_step=1, batch_size=32, memory_capacity=10000000, lr=0.0003, gamma=0.99, tau=0.005, alpha=0.2, policy_type="Gaussian", target_update_interval=1, automatic_entropy_tuning=False, hidden_size=256, device='cpu', verbose_histogram=False, seed=42):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.verbose_histogram = verbose_histogram
        self.memory = ReplayMemory(memory_capacity, seed=seed)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy_type = policy_type
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.env = env
        action_space = self.env.action_space
        num_inputs = self.env.observation_space.shape[0]
        self.device = torch.device(device)
        self.writer = SummaryWriter()
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        """Выбор действия на основе текущего состояния.

        Args:
            state: Текущее состояние агента.
            evaluate (bool): Флаг режима оценки.

        Returns:
            action: Выбранное действие.

        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        """Обновление параметров сетей на основе мини-пакета из памяти.

        Args:
            memory: Память для хранения переходов.
            batch_size (int): Размер мини-пакета.
            updates (int): Количество обновлений.

        Returns:
            qf1_loss (float): Значение функции потерь для первой Q-сети.
            qf2_loss (float): Значение функции потерь для второй Q-сети.
            policy_loss (float): Значение функции потерь для политики.
            alpha_loss (float): Значение функции потерь для коэффициента alpha.
            alpha_tlogs (float): Значение коэффициента alpha.

        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            
        self.writer.add_scalar("Loss/QF1", qf1_loss.item(), updates)
        self.writer.add_scalar("Loss/QF2", qf2_loss.item(), updates)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), updates)
        self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), updates)
        self.writer.add_scalar("Loss/Alpha", alpha_tlogs.item(), updates)
        
        if self.verbose_histogram:
            for name, param in self.critic.named_parameters():
                self.writer.add_histogram(f"Critic/{name}", param, updates)

            for name, param in self.policy.named_parameters():
                self.writer.add_histogram(f"Policy/{name}", param, updates)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def train(self, num_episodes):
        # Training Loop
        total_numsteps = 0
        updates = 0
        for i_episode in tqdm(range(num_episodes)):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, info = self.env.reset()
            reward_per_step = []
            done = False
            while not done:
                action = self.select_action(state)
                if len(self.memory) > self.batch_size:
                    for i in range(self.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.update_parameters(self.memory, self.batch_size, updates)
                        updates += 1
            

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                reward_per_step.append(reward)
                mask = 1 if done else float(not done)
                self.memory.push(state, action, reward, next_state, mask) # Append transition to memory
                state = next_state
            self.writer.add_scalar('Performance/Reward', episode_reward, i_episode)
            
    def get_param_env(self):
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"
        env_params = {}
        
        # Получение информации о сигнале справки, если она доступна
        try:
            ref_signal = self.env.ref_signal.__class__
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
            "tau": self.tau,
            "alpha": self.alpha,
            "verbose_histogram": self.verbose_histogram,
            "memory_capacity": self.memory.capacity,  # Assuming the ReplayMemory class has a capacity attribute.
            "policy_type": self.policy_type,
            "updates_per_step": self.updates_per_step,
            "target_update_interval": self.target_update_interval,
            "batch_size": self.batch_size,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "device": self.device.type,
            "lr": self.critic_optim.defaults['lr'],  # Or another way to get learning rate.
        }
        print(policy_params)
        
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
        policy_path = path / date_str / "policy.pth"
        critic_path = path / date_str / "critic.pth"
        critic_target_path = path / date_str / "critic_target.pth"
        
        # Создание директории, если она не существует
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile: 
            json.dump(config, outfile)
        torch.save(self.policy, policy_path)
        torch.save(self.critic, critic_path)
        torch.save(self.critic_target, critic_target_path)
    
    @classmethod
    def __load(cls, path):
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        policy_path = path / "policy.pth"
        critic_target_path = path / "critic_target.pth"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"
        
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            env = get_class_from_string(config["env"]["name"])(**config["env"]["param"])
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])
        new_agent.critic = torch.load(critic_path)
        new_agent.policy = torch.load(policy_path)
        new_agent.critic_target = torch.load(critic_target_path)
        return new_agent
        
    @classmethod
    def from_pretrained(cls, repo_name, access_token=None, version=None):
        path = Path(repo_name)
        if path.exists():
            new_agent = cls.__load(path)
            return new_agent
        else:
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent