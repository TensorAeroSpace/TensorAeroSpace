import tensorflow as tf

import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

np.random.seed(1)
tf.random.set_seed(1)

class Model(tf.keras.Model):

    "Класс для DQN модели"

    def __init__(self, num_actions):

        "Инициализация"

        super().__init__(name='basic_prddqn')
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    def call(self, inputs):

        "Функция forward. Возвращает q функции для действий"

        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    def action_value(self, obs):

        "Функция стратегии. Принимает на вход состояние, возвращает действие."

        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]


def test_model():

    "функция для проверки работоспособности модели"

    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = Model(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0 eager mode: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


class SumTree:

    "Класс бинарного дерева для реплей буфера"

    def __init__(self, capacity):

        "Инициализация"

        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):

        "Количество записей в буфере"

        return self.tree[0]

    def add(self, priority, transition):

        "Функция для добавления объекта в буфер"

        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx, priority):

        "Обновление приоритетов в буфере"

        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)

    def _propagate(self, idx, change):

        "Обратное распространение приоритетов"

        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):

        "Получение значения листа дерева"

        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):

        "Поиск элемента в дереве"

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


class PERAgent:

    "Класс для DQN агента"

    def __init__(self, model, target_model, env, learning_rate=.0012, epsilon=.1, epsilon_dacay=0.995, min_epsilon=.01,
                 gamma=.9, batch_size=8, target_update_iter=400, train_nums=5000, buffer_size=200, replay_period=20,
                 alpha=0.4, beta=0.4, beta_increment_per_sample=0.001):
        self.model = model
        self.target_model = target_model
        # gradient clip
        opt = ko.Adam(learning_rate=learning_rate)  # , clipvalue=10.0
        self.model.compile(optimizer=opt, loss=self._per_loss)  # loss=self._per_loss

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.epsilon = epsilon                      # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # minibatch k
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps

        # replay buffer params [(s, a, r, ns, done), ...]
        self.b_obs = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_dones = np.empty(self.batch_size, dtype=np.bool)

        self.replay_buffer = SumTree(buffer_size)   # sum-tree data structure
        self.buffer_size = buffer_size              # replay buffer size N
        self.replay_period = replay_period          # replay period K
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.beta = beta                            # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.margin = 0.01                          # pi = |td_error| + margin
        self.p1 = 1                                 # initialize priority for the first transition
        # self.is_weight = np.empty((None, 1))
        self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
        self.abs_error_upper = 1

    def _per_loss(self, y_target, y_pred):

        "Получение функции ошибки"

        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))

    def train(self):

        "Функция для обучения"

        # initialize the initial observation of the agent
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, info = self.env.step(action)    # take the action in the env to return s', r, done
            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity:])
            self.store_transition(p, obs, action, reward, next_obs, done)  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.buffer_size:
                # if t % self.replay_period == 0:  # transition sampling and update
                losses = self.train_step()
                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                obs = self.env.reset()   # one episode end
            else:
                obs = next_obs

    def train_step(self):

        "Шаг обучения"

        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)
        # Double Q-Learning
        best_action_idxes, _ = self.model.action_value(self.b_next_states)  # get actions through the current network
        target_q = self.get_target_value(self.b_next_states)    # get target q-value through the target network
        # get td_targets of batch states
        td_target = self.b_rewards + \
            self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - self.b_dones)
        predict_q = self.model.predict(self.b_obs)
        td_predict = predict_q[np.arange(predict_q.shape[0]), self.b_actions]
        abs_td_error = np.abs(td_target - td_predict) + self.margin
        clipped_error = np.where(abs_td_error < self.abs_error_upper, abs_td_error, self.abs_error_upper)
        ps = np.power(clipped_error, self.alpha)
        # priorities update
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, p)

        for i, val in enumerate(self.b_actions):
            predict_q[i][val] = td_target[i]

        target_q = predict_q  # just to change a more explicit name
        losses = self.model.train_on_batch(self.b_obs, target_q)

        return losses

    # proportional prioritization sampling
    def sum_tree_sample(self, k):

        "Получение элемента из буфера"

        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.replay_buffer.tree[-self.replay_buffer.capacity:]) / self.replay_buffer.total_p
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t
            # P(j)
            sampling_probabilities = p / self.replay_buffer.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.buffer_size * sampling_probabilities, -self.beta) / max_weight
        return idxes, is_weights

    def evaluation(self, env, render=False):

        "Валидация обученной сети"

        obs, done, ep_reward = env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:  # visually show
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    # store transitions into replay butter, now sum tree.
    def store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)

    # rank-based prioritization sampling
    def rand_based_sample(self, k):
        pass

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay