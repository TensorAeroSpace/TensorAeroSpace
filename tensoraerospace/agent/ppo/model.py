import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v
    

class actor(tf.keras.Model):
    def __init__(self): 
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.a = tf.keras.layers.Dense(3 ** 7,activation='softmax')
        self.r = tf.keras.layers.Dense(1, activation='relu')

    def call(self, input_data, return_reward=False):
        x = self.d1(input_data)
        a = self.a(x)
        if return_reward:
            r = self.r(x)
            return a, r
        return a

class Agent():
    """ Класс агента ppo

    Args:
        env (_type_): среда
        gamma (float): коэффициент дисконтирования
    """
    def __init__(self, env, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2
        self.env = env
        tf.random.set_seed(336699)
        self.max_steps = 10
        self.max_episodes = 10
        self.ep_reward = []
        self.total_avgr = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []

          
    def act(self,state):
        
        """ Функция которая возвращает действие агента

        Args:
            state (_type_): вектор состояния

        Returns:
            action (int): дискретное действие
        """
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        if int(action.numpy()[0]) >= 3 ** 7:
            return int(action.numpy()[0]) - 1
        return int(action.numpy()[0])
  


    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        """ Функция которая вычисляет ошибку актора

        Args:
            probs (_type_): батч вероятностей действий
            actions (_type_): батч действий
            adv (_type_): батч значений advantage функции
            old_probs (_type_): батч старых вероятностей действий
            closs (_type_): ошибка критика


        Returns:
            loss (float): ошибка актора
        """
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        sur1 = []
        sur2 = []
        
        for pb, t, op,a  in zip(probability, adv, old_probs, actions):
            t =  tf.constant(t)
            ratio = tf.math.divide(pb[a],op[a])
            s1 = tf.math.multiply(ratio,t)
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss
    
    def auxillary_task(self, r, rewards):

        """ Функция которая вычисляет ошибку для auxillary task (предсказание награды)

        Args:
            r (_type_): батч предсказанных наград
            rewards (_type_): батч реальных наград


        Returns:
            loss (float): ошибка предскзателя наград
        """

        loss = tf.reduce_mean(tf.math.square(r - rewards))
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards, rewards):

        """ Функция которая вычисляет общую ошибку

        Args:
            states (_type_): батч состояний
            actions (_type_): батч действий
            adv (_type_): батч значений advantage функции
            old_probs (_type_): батч старых вероятностей действий
            discnt_rewards (_type_): батч отложенных наград
            rewards (_type_): батч реальных наград


        Returns:
            a_loss (float): ошибка актора
            с_loss (float): ошибка критика
        """

        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),3 ** 7))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p, r = self.actor(states, return_reward=True, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, tf.cast(v, np.float32))
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss) + self.auxillary_task(r, rewards)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
    
    def test_reward(self):

        """ Функция которая тестирует алгоритм на одном эпизоде
        """

        total_reward = 0
        state = self.env.reset()
        done = False
        for i in range(self.max_steps):
            action = np.argmax(self.actor(np.array([state])).numpy())
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += reward

        return total_reward
    
    
    def preprocess1(self, states, actions, rewards, done, values, gamma):

        """ Функция которая обрабатывает переходы для сохранения в буффер

        Args:
            states (_type_): батч состояний
            actions (_type_): батч действий
            rewards (_type_): батч реальных наград
            done (_type_): батч флагов окончания эпизода
            values (_type_): батч значений выходов критика
            gamma (float): коэффициент дисконтирования


        Returns:
            states (_type_): батч состояний
            actions (_type_): батч действий
            returns (_type_): батч отложенных наград
            adv (_type_): батч значений advantage функции
            rewards (_type_): батч реальных наград
        """

        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * self.dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        return states, actions, returns, adv, rewards    
    
    def train(self):

        """ Функция обучения агента
        """

        for s in range(self.max_episodes):
            if self.target == True:
                break
        
            done = False
            state = self.env.reset()
            self.all_aloss = []
            self.all_closs = []
            self.rewards = []
            self.states = []
            self.actions = []
            self.probs = []
            self.dones = []
            self.values = []

            for e in range(self.max_steps):
        
                action = self.act(state)
                value = self.critic(np.array([state])).numpy()
                next_state, reward, done, _ = self.env.step(action)
                self.dones.append(1-done)
                self.rewards.append(reward)
                self.states.append(state)
                self.actions.append(action)
                prob = self.actor(np.array([state]))
                self.probs.append(prob[0])
                self.values.append(value[0][0])
                state = next_state
                if done:
                    self.env.reset()
        
            value = self.critic(np.array([state])).numpy()
            self.values.append(value[0][0])
            np.reshape(self.probs, (len(self.probs),3 ** 7))
            self.probs = np.stack(self.probs, axis=0)

            self.states, self.actions, self.returns, self.adv, self.rewards  = self.preprocess1(self.states, self.actions, self.rewards, self.dones, self.values, 1)

            for epocs in range(1):
                al,cl = self.learn(self.states, self.actions, self.adv, self.probs, self.returns, self.rewards)

            avg_reward = np.mean([self.test_reward() for _ in range(5)])
            self.avg_rewards_list.append(avg_reward)
            self.env.reset()
