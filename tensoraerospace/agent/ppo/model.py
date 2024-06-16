import datetime
import json
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


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """
    Инициализирует веса и смещения (биасы) слоя с помощью равномерного распределения.

    Args:
        layer (nn.Linear): Слой нейронной сети, который будет инициализирован.
        init_w (float, optional): Половина интервала для равномерного распределения. По умолчанию 3e-3.

    Returns:
        nn.Linear: Слой с инициализированными весами и смещениями.

    Примеры:
        >>> layer = nn.Linear(10, 5)
        >>> init_layer_uniform(layer)
        Linear(in_features=10, out_features=5, bias=True)
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Инициализирует модуль критика.

        Args:
            input_dim (int): Размерность входных данных.
            hidden_dim (int, optional): Размер скрытого слоя. По умолчанию равен 64.

        Осуществляет следующие операции:
        - Инициализирует первый линейный слой для преобразования входных данных в промежуточное представление.
        - Инициализирует второй линейный слой для вычисления "значения" из промежуточного представления.
        - Инициализирует второй линейный слой с использованием равномерного распределения.
        """
        super(Critic, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.v = init_layer_uniform(self.v)

    def forward(self, input_data):
        """
        Производит прямой проход сети.

        Args:
            input_data (Tensor): Тензор входных данных.

        Returns:
            Tensor: Выходной тензор, представляющий "значение" для каждого входного примера.

        Применяет последовательность операций:
        - Пропускает входные данные через первый линейный слой и применяет функцию активации ReLU.
        - Пропускает результат через второй линейный слой для вычисления конечного "значения".
        """
        x = F.relu(self.d1(input_data))
        v = self.v(x)
        return v

class Actor(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=32):
        """
        Инициализирует класс Actor, который является подклассом nn.Module.

        Args:
            input_dim (int): Размер входного слоя.
            out_dim (int): Размер выходного слоя.
            hidden_dim (int, optional): Размер скрытого слоя. По умолчанию равен 32.

        Инициализирует линейные слои для расчета промежуточных представлений и параметров действий.
        Использует пользовательские функции init_layer_uniform для инициализации слоев `mu` и `delta`.
        """
        super(Actor, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.a = nn.Linear(hidden_dim, out_dim)
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.mu = init_layer_uniform(self.mu)
        self.delta = nn.Linear(hidden_dim, out_dim)
        self.delta = init_layer_uniform(self.delta)
        self.log_std_min = -20
        self.log_std_max = 0
        self.r = nn.Linear(hidden_dim, 1)

    def forward(self, input_data, return_reward=False, continous_actions=False):
        """
        Производит прямой проход через модель, вычисляя действия агента на основе входных данных.

        Args:
            input_data (Tensor): Входные данные для модели.
            return_reward (bool, optional): Флаг, указывающий, следует ли возвращать вознаграждение. По умолчанию False.
            continous_actions (bool, optional): Флаг, указывающий, должны ли действия быть непрерывными. По умолчанию False.

        Returns:
            tuple или Tensor: В зависимости от флагов возвращает действие, распределение (и вознаграждение, если запрошено).
            Если continous_actions True, возвращает либо пару (action, dist), либо тройку (action, dist, r).
            В противном случае возвращает либо действие, либо пару (action, r).
        """
        x = F.relu(self.d1(input_data))

        if continous_actions:
            mu = torch.tanh(self.mu(x))
            log_std = torch.tanh(self.delta(x))
            log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            if return_reward:
                r = torch.flatten(F.relu(self.r(x)))
                return action, dist, r
            return action, dist
        a = F.softmax(self.a(x), dim=-1)
        if return_reward:
            r = torch.flatten(F.relu(self.r(x)))
            return a, r
        return a

def ppo_iter(epoch: int,mini_batch_size: int,states: torch.Tensor,actions: torch.Tensor,
            log_probs: torch.Tensor,returns: torch.Tensor,advantages: torch.Tensor,rewards: torch.Tensor):
    """Инициализирует итератор для PPO.

    Args:
        epoch (int): Количество эпох для итераций.
        mini_batch_size (int): Размер мини-батча для каждой итерации.
        states (torch.Tensor): Тензор состояний.
        actions (torch.Tensor): Тензор действий.
        log_probs (torch.Tensor): Тензор логарифмов вероятностей действий.
        returns (torch.Tensor): Тензор ожидаемых доходов.
        advantages (torch.Tensor): Тензор преимуществ.
        rewards (torch.Tensor): Тензор наград.
    """
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], \
            log_probs[rand_ids], returns[rand_ids], advantages[rand_ids], \
            rewards[rand_ids]
            

class PPO(BaseRLModel):
    """ Класс, реализующий агента PPO с использованием PyTorch.

    Args:
        env: объект окружения.
        gamma (float): коэффициент дисконтирования.
    """
    def __init__(self, env, gamma=0.99, max_episodes = 30, rollout_len = 2048, clip_pram = 0.2, num_epochs=64, batch_size=64, entropy_coef=0.005, actor_lr = 0.001, critic_lr = 0.005, seed= 336699):
        """Инициализация агента с заданным окружением и коэффициентом дисконтирования.
        
        Args:
            env: объект окружения, с которым будет взаимодействовать агент.
            gamma (float, optional): коэффициент дисконтирования, используемый в расчетах. По умолчанию 0.99.
        """
        self.gamma = gamma
        self.env = env
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
        self.critic = Critic(env.observation_space.shape[0])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.clip_pram = clip_pram
        torch.manual_seed(seed)
        self.rollout_len = rollout_len
        self.max_episodes = max_episodes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.ep_reward = []
        self.total_avgr = []
        self.target = False
        self.best_reward = 0
        self.avg_rewards_list = []
        self.writer = SummaryWriter()
        
        
    def act(self, state):
        """Выбирает действие для данного состояния.

        Args:
            state: текущее состояние среды.

        Returns:
            tuple: кортеж, содержащий действие, среднее действие и логарифм вероятности действия.
        """
        state = torch.FloatTensor(np.array([state]))
        action, dist = self.actor(state, continous_actions=True)
        return action.detach(), dist.mean.detach().numpy(), dist.log_prob(action)#, prob.detach().numpy()

    def actor_loss(self, probs, entropy, actions, adv, old_probs):
        """Вычисляет потери актора.

        Args:
            probs: вероятности действий новой политики.
            entropy: энтропия действий.
            actions: предпринятые действия.
            adv: преимущества (advantages).
            old_probs: вероятности действий старой политики.

        Returns:
            Tensor: значение функции потерь для актора.
        """
        ratios = torch.exp(probs - old_probs)
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * adv
        loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * entropy
        return loss

    def auxillary_task(self, r, rewards):
        """Вычисляет потери вспомогательной задачи (прогнозирование наград).

        Args:
            r: предсказанные награды.
            rewards: реальные награды.

        Returns:
            Tensor: значение функции потерь MSE между предсказанными и реальными наградами.
        """
        return F.mse_loss(r, rewards)

    def learn(self, states, actions, adv, old_probs, discnt_rewards, rewards):
        """Процедура обучения агента.

        Args:
            states: состояния, испытанные агентом.
            actions: действия, предпринятые агентом.
            adv: преимущества (advantages).
            old_probs: логарифмические вероятности предыдущих действий.
            discnt_rewards: дисконтированные награды.
            rewards: фактические полученные награды.

        Returns:
            tuple: кортеж, содержащий значения функций потерь актора и критика.
        """
        self.a_opt.zero_grad()
        self.c_opt.zero_grad()
        new_actions, new_distr, r = self.actor(states, return_reward=True, continous_actions=True)
        new_probs = new_distr.log_prob(actions)
        v = self.critic(states)
        td = discnt_rewards.squeeze() - v.squeeze()
        c_loss = td.pow(2).mean()
        a_loss = self.actor_loss(new_probs, -new_distr.entropy().mean(), actions, adv.detach(), old_probs)
        a_loss.backward()
        c_loss.backward(retain_graph=True)
        self.a_opt.step()
        self.c_opt.step()
        return a_loss.item(), c_loss.item()

    def test_reward(self):
        """Тестирование модели путем выполнения одного эпизода.

        Returns:
            float: суммарная награда за эпизод.
        """
        total_reward = 0
        reset_return = self.env.reset()
        if type(reset_return) is tuple:
            state, info = reset_return
        else:
            state = reset_return
        done = False
        while not done:
            action, mean_action, delta = self.act(state)
            step_return = self.env.step(mean_action[0])
            if len(step_return) > 4:
                next_state, reward, terminated, trunkated, info = step_return
                done = terminated or trunkated
            else:
                next_state, reward, terminated, info = step_return
                done = terminated
            total_reward += reward
        return total_reward


    def preprocess1(self, states, actions, rewards, dones, values, probs, gamma):
        """Предобработка переходов для буфера.

        Args:
            states: список состояний.
            actions: список действий.
            rewards: список наград.
            dones: список булевых значений, указывающих окончание эпизода.
            values: значения состояний.
            probs: логарифмические вероятности действий.
            gamma: коэффициент дисконтирования.

        Returns:
            tuple: кортеж, содержащий обработанные состояния, действия, награды, преимущества и вероятности.
        """

        states2 = torch.cat(states).view(-1, 3)
        actions2 = torch.cat(actions).detach()
        rewards2 = torch.cat(rewards)
        dones2 = torch.cat(dones)
        values2 = torch.cat(values).flatten()
        probs2 = torch.cat(probs).detach()

        returns2 = []
        g2 = 0
        lmbda2 = 0.8
        for i in reversed(range(len(rewards))):
            delta2 = rewards2[i] + gamma * values2[i + 1] * (1 - dones2[i]) - values2[i]
            g2 = delta2 + gamma * lmbda2 * (1 - dones2[i]) * g2
            returns2.insert(0, g2 + values2[i].view(-1, 1))

        # returns = torch.tensor(returns).detach()
        adv2 = torch.tensor(returns2).detach() - values2[:-1]
        # adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return states2, actions2, returns2, adv2, rewards2, probs2

    def train(self):
        """Функция обучения агента.

        В процессе обучения агент проходит через заданное количество эпизодов, собирает данные,
        обрабатывает их и обновляет параметры модели.
        """
        for episode in tqdm(range(self.max_episodes)):
            # print("Episode", episode)
            if self.target:
                break

            reset_return = self.env.reset()
            if type(reset_return) is tuple:
                state, info = reset_return
            else:
                state = reset_return
            done = False
            all_aloss = []
            all_entropies = []
            episode_lengths = []
            all_closs = []
            rewards = []
            states = []
            actions = []
            probs = []
            # mus = []
            # deltas = []
            dones = []
            values = []
            scores = []
            score = 0

            for step in range(self.rollout_len):
                action, mu, prob = self.act(state)
                # prob = self.actor(torch.from_numpy(np.array([state], dtype=np.float32)))
                value = self.critic(torch.FloatTensor(np.array([state])))
                # print("action_l", action)
                #print(action)
                step_return = self.env.step(action.detach().numpy()[0])
                if len(step_return) > 4:
                    next_state, reward, terminated, trunkated, info = step_return
                    done = terminated or trunkated
                else:
                    next_state, reward, terminated, info = step_return
                    done = terminated
                score += reward
                dones.append(torch.FloatTensor(np.reshape(done, (1, -1)).astype(np.float64)))
                rewards.append(torch.FloatTensor(np.reshape(reward, (1, -1)).astype(np.float64)))
                states.append(torch.FloatTensor(state))
                actions.append(action[0])
                probs.append(prob)
                values.append(value)

                state = next_state
                if done:
                    scores.append(score)
                    episode_lengths.append(step)
                    score = 0
                    reset_return = self.env.reset()
                    if type(reset_return) is tuple:
                        state, info = reset_return
                    else:
                        state = reset_return

            # Calculate next state value for the terminal state
            next_value = self.critic(torch.FloatTensor(np.array([next_state])))
            values.append(next_value)

            _, _, returns, _, _, _ = self.preprocess1(
                states, actions, rewards, dones, values, probs, self.gamma
            )
            states = torch.cat(states).view(-1, self.env.observation_space.shape[0])
            actions = torch.cat(actions).view(-1, 1)
            rewards = torch.cat(rewards)
            returns = torch.cat(returns).detach()
            values = torch.cat(values).detach()
            probs = torch.cat(probs).detach()
            advantages = returns - values[:-1]
            # Train for a number of epochs
            for state, action, old_log_prob, return_, adv, reward in ppo_iter(
                epoch=self.num_epochs,
                mini_batch_size=self.batch_size,
                states=states,
                actions=actions,
                log_probs=probs,
                returns=returns,
                advantages=advantages,
                rewards=rewards):
                a_loss, c_loss = self.learn(
                    state, action, adv, old_log_prob, return_, reward
                )
                all_aloss.append(a_loss)
                all_closs.append(c_loss)
                all_entropies.append(-adv.mean().item())  
                
            avg_reward = np.mean(scores)
            avg_aloss = np.mean(all_aloss)
            avg_closs = np.mean(all_closs)
            avg_entropy = np.mean(all_entropies)
            avg_episode_length = np.mean(episode_lengths)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/Actor', avg_aloss, episode)
            self.writer.add_scalar('Loss/Critic', avg_closs, episode)
            self.writer.add_scalar('Performance/Reward', avg_reward, episode)
            self.writer.add_scalar('Performance/Entropy', avg_entropy, episode)
            self.writer.add_scalar('Performance/Episode Length', avg_episode_length, episode)

        # print("Training completed. Average rewards list:", self.avg_rewards_list)
        
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
            "max_episodes": self.max_episodes,
            "rollout_len": self.rollout_len,
            "clip_pram": self.clip_pram,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "seed": self.seed,

            
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
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        actor_path = path / "actor.pth"
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
        new_agent.actor = torch.load(actor_path)
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