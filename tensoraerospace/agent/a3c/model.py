import datetime
from multiprocessing import cpu_count
from threading import Thread

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda

tf.keras.backend.set_floatx('float64')

GLOBAL_EP = 0


class Actor(tf.keras.Model):
    """Модель актора в A3C

    Args:
        state_size (_type_): Размер состояния подающегося на вход
        action_size (_type_): Размер вектора действий
        action_bound (_type_): Границы действий
        std_bound (_type_): Границы стандартного отклонения
    """
    def __init__(self, state_size, action_size, action_bound, std_bound):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        """
        Функция создающая модель актора
        """
        state_input = Input((self.state_size,))
        dense_1 = Dense(hidden_size, activation='relu')(state_input)
        dense_2 = Dense(hidden_size, activation='relu')(dense_1)
        out_mu = Dense(self.action_size, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_size, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])
    
    def compute_loss(self, actions, mu, std, advantages):
        """ Функция которая вычисляет ошибку актора

        Args:
            actions (_type_): батч действий
            mu (float): коэффициент
            std (float): стандартное отклонение
            advantages (_type_): батч значений функции advantage

        Returns:
            policy_loss (float): ошибка актора
        """
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (actions - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        log_policy_pdf = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
        policy_loss = log_policy_pdf * advantages
        policy_loss = tf.reduce_sum(-policy_loss)
        return policy_loss

    def train(self, states, actions, advantages):
        """Функция для одного шага обновления сети актора

        Args:
            states (_type_): батч состояний
            actions (_type_): батч действий
            advantages (_type_): батч значений функции advantage

        Returns:
            loss (float): ошибка актора
        """
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(actions, mu, std, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('actor loss', loss.numpy(), step=GLOBAL_EP)
        return loss


class Critic(tf.keras.Model):
    """
    Модель критика в A3C

    Args:
        state_size (_type_): Размер состояния подающегося на вход
    """
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        """
        Функция создающая модель критика
        """
        return tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(hidden_size, activation='relu'),
            Dense(hidden_size, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        """ Функция которая вычисляет ошибку критика

        Args:
            v_pred (_type_): батч предсказанных Q функций
            td_targets (_type_): батч целевых Q функций

        Returns:
            loss (float): ошибка критика
        """
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        """Функция для одного шага обновления сети критика

        Args:
            states (_type_): батч состояний
            td_targets (_type_): батч целевых Q функций

        Returns:
            loss (float): ошибка критика
        """
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('critic loss', loss.numpy(), step=GLOBAL_EP)
        return loss


class Worker(Thread):
    """
    Класс для одного из асинхронных агентов

    Args:
        env (_type_): среда для взаимодействия
        gamma (float): коэффициент гамма
        global_actor (_type_): ссылка на глобальную сеть актора
        global_critic (_type_): ссылка на глобальную сеть критика
    """
    def __init__(self, env, gamma, global_actor, global_critic):
        Thread.__init__(self)
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        self.gamma = gamma
        self.global_actor = global_actor
        self.global_critic = global_critic
        
        self.actor = Actor(self.state_size, self.action_size, self.action_bound, self.std_bound)
        self.critic = Critic(self.state_size)

        self.sync_with_global()
    
    def get_action(self, state):
        """Функция которая предсказывает действие для данного состояния

        Args:
            state (_type_): состояние среды

        Returns:
            action (_type_): предсказанное действие
        """
        state = np.reshape(state, [1, self.state_size])
        mu, std = self.actor.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_size)
    
    def n_step_td_target(self, rewards, next_Qs, done):
        """Функция для подсчёта отложенной награды

        Args:
            rewards (float): полученные награды
            next_Q (float): предсказанные Q функции
            done (bool): флаг завершения эпизода

        Returns:
            td_targets (float): целевые Q функции
        """
        td_targets = np.zeros_like(rewards)
        #R_to_go = 0

        #Rs = self.critic.model.predict(states)
        
        if done:
            td_targets[-1] = rewards[-1]
        
        for k in reversed(range(0, len(rewards) - 1)):
            td_targets[k] = next_Qs[k] + rewards[k]
        return td_targets

    def list_to_batch(self, list):
        """Функция для преобразования списка переходов в батч для обновления

        Args:
            list (_type_): список переходров

        Returns:
            batch (_type_): батч переходров
        """
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def sync_with_global(self):
        """
        Функция для копирования глобальных весов в асинхронного агента
        """
        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())
    
    def run(self):
        """
        Функция тренировки асинхронных агентов. В данной функции происходит цикл взаимодействия со средой и
        вызовы обновления весов глобального агента
        """
        global GLOBAL_EP
        while max_episodes > GLOBAL_EP:
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            states = []
            actions = []
            rewards = []
            next_states = []
            i1 = 0
            while not done and i1 < 10:
                i1 += 1
                action = self.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])
                action = np.reshape(action, [1, self.action_size])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state[0]
                episode_reward += reward[0][0]
                
                if len(states) >= update_interval or done:
                    states = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    next_states = self.list_to_batch(next_states)
                    
                    #curr_Qs = self.critic.model.predict(states)
                    #next_Qs = self.critic.model.predict(next_states)
                    
                    #td_targets = self.n_step_td_target((rewards+8)/8, next_Qs, done)
                    #advantages = td_targets - curr_Qs
                    
                    #actor_loss = self.global_actor.train(states, actions, advantages)
                    #critic_loss = self.global_critic.train(states, td_targets)

                    self.sync_with_global()
                    states = []
                    actions = []
                    rewards = []
                    next_states = []

            with summary_writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=GLOBAL_EP)
            GLOBAL_EP += 1

actor_lr = 0.0005
critic_lr = 0.001
gamma = 0.99
hidden_size = 128
update_interval = 50

max_episodes = 50

def setup_global_params(actor_lr_f, critic_lr_f, gamma_f, hidden_size_f, update_interval_f, max_episodes_f):
    """
    Функция для установки глобальных параметров для алгоритма
    Args:
        actor_lr_f (float): learning rate актора
        critic_lr_f (float): learning rate критика
        gamma_f (float): коэффициент гамма
        hidden_size_f (float): размер скрытого слоя
        update_interval_f (float): интервал обновления сетей
        max_episodes_f (float): максимальное количество эпизодов обучения
    """
    global actor_lr
    global critic_lr
    global gamma
    global hidden_size
    global update_interval
    global max_episodes
    actor_lr = actor_lr_f
    critic_lr = critic_lr_f
    gamma = gamma_f
    hidden_size = hidden_size_f
    update_interval = update_interval_f
    max_episodes = max_episodes_f

summary_writer = 0

class Agent:
    """
    Общий класс для A3C агента

    Args:
        env_function (_type_): функция для создания среды
        gamma (float): коэффициент гамма
    """
    def __init__(self, env_function, gamma):
        self.env_function = env_function
        env = self.env_function(0)
        self.gamma = gamma
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(self.state_size, self.action_size, self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_size)

        self.num_workers = cpu_count()
        env.close()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer("/tf/logs/run" + current_time)
        global summary_writer
        summary_writer = self.summary_writer

    def train(self):
        """
        Функция для запуска алгоритма обучения
        """
        self.workers = []

        for i in range(self.num_workers):
            env = self.env_function(i)
            self.workers.append(Worker(
                env, self.gamma, self.global_actor, self.global_critic))
        
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()
            worker.env.close()
