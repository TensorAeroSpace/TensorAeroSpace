import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.layers import Input, Dense, Lambda

from threading import Thread
from multiprocessing import cpu_count
tf.keras.backend.set_floatx('float64')

GLOBAL_EP = 0

class Actor(tf.keras.Model):
    
    def __init__(self, state_size, action_size, action_bound, std_bound):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        state_input = Input((self.state_size,))
        dense_1 = Dense(hidden_size, activation='relu')(state_input)
        dense_2 = Dense(hidden_size, activation='relu')(dense_1)
        out_mu = Dense(self.action_size, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_size, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])
    
    def compute_loss(self, actions, mu, std, advantages):
        # log_policy_pdf = self.log_pdf(mu, std, actions)
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (actions - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        log_policy_pdf = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
        policy_loss = log_policy_pdf * advantages
        policy_loss = tf.reduce_sum(-policy_loss)
        return policy_loss

    def train(self, states, actions, advantages):
        
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(actions, mu, std, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic(tf.keras.Model):

    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(hidden_size, activation='relu'),
            Dense(hidden_size, activation='relu'),
            # Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Worker(Thread):
    def __init__(self, id, env, gamma, global_actor, global_critic):
        Thread.__init__(self)
        self.name = "w%i" % id
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        self.gamma = gamma
        self.global_actor = global_actor
        self.global_critic = global_critic
        
        self.actor = Actor(self.state_size, self.action_size,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_size)
        
        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        mu, std = self.actor.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_size)
    
    def n_step_td_target(self, rewards, next_Q, done):
        td_targets = np.zeros_like(rewards)
        R_to_go = 0
        
        if not done:
            R_to_go = next_Q
        
        for k in reversed(range(0, len(rewards))):
            R_to_go = rewards[k] + self.gamma * R_to_go 
            td_targets[k] = R_to_go
        return td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def sync_with_global(self):
        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())
    
    def run(self):
        global GLOBAL_EP
        while max_episodes > GLOBAL_EP:
        # for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            states      = []
            actions     = []
            rewards     = []
            
            while not done:
                action = self.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                
                next_state, reward, done, _ = self.env.step(action)
                
                state      = np.reshape(state, [1, self.state_size])
                action     = np.reshape(action, [1, self.action_size])
                reward     = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state[0]
                episode_reward += reward[0][0]
                
                if len(states) >= update_interval or done:
                    states  = self.list_to_batch(states)
                    actions = self.list_to_batch(actions)
                    rewards = self.list_to_batch(rewards)
                    
                    curr_Qs = self.critic.model.predict(states)
                    next_Q = self.critic.model.predict(next_state)
                    
                    td_targets = self.n_step_td_target(
                        (rewards+8)/8, next_Q, done)
                    # advantages   = td_targets - self.critic.model.predict(states)
                    advantages   = td_targets - curr_Qs
                    
                    actor_loss = self.global_actor.train(states, actions, advantages)
                    critic_loss = self.global_critic.train(states, td_targets)

                    self.sync_with_global()
                    states     = []
                    actions    = []
                    rewards    = []

            print(self.name + ' | EP{} EpisodeReward={}'.format(GLOBAL_EP+1, episode_reward))
            GLOBAL_EP += 1

class Agent:
    
    def __init__(self, env_name, gamma):
        env = get_plane_env("//tf//repos//linux_build//build.x86_64", server=True)
        self.env_name = env_name
        self.gamma = gamma
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(self.state_size, self.action_size,
                                 self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_size)
        
        #self.num_workers = 4
        self.num_workers = cpu_count()
        env.close()
        
    def train(self):
        print("Training on {} cores".format(self.num_workers))
        input("Enter to start")
        self.workers = []

        for i in range(self.num_workers):
            env = get_plane_env("//tf//repos//linux_build//build.x86_64", server=True, worker=i)
            self.workers.append(Worker(
                i, env, self.gamma, self.global_actor, self.global_critic))
        
        # [worker.start() for worker in self.workers]
        # [worker.join() for worker in self.workers]
        
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()
            worker.env.close()
    
    # def save_model(self):
    #     self.global_critic.save("a3c_value_model.h5")
    #     self.global_actor.save("a3c_policy_model.h5")

#if __name__ == "__main__":
    
#    env_name = "Pendulum-v0"
    # set environment
#    actor_lr = 0.0005
#    critic_lr = 0.001
#    gamma = 0.99
#    hidden_size = 128
#    update_interval = 50
    
#    max_episodes = 50  # Set total number of episodes to train agent on.
#    agent = A3CAgent(env_name, gamma)
#    agent.train()
    # agent.save_model()
