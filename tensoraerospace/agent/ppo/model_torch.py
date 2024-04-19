import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.d1 = nn.Linear(128, activation='relu')
        self.v = nn.Linear(1, activation=None)

    def forward(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v

class Actor(nn.Module):
    def __init__(self, num_actions=7):
        super(Actor, self).__init__()
        self.d1 = nn.Linear(128, activation='relu')
        self.a = nn.Linear(3 ** num_actions, activation='softmax')
        self.mu = nn.Linear(num_actions, activation='relu')
        self.delta = nn.Linear(num_actions, activation='relu')
        self.r = nn.Linear(1, activation='relu')

    def forward(self, input_data, return_reward=False, continuos=False):
        x = self.d1(input_data)
        a = self.a(x)
        mu = self.mu(x)
        delta = self.delta(x)
        if return_reward:
            r = self.r(x)
            return a, r
        if continuos:
            return mu, delta
        return a



class Agent():
    """ Класс агента ppo using PyTorch

    Args:
        env: Environment object
        gamma (float): Discount factor
    """
    def __init__(self, env, actor=Actor, critic=Critic, gamma=0.99):
        self.gamma = gamma
        self.actor = actor
        self.critic = critic      
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=7e-3)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=7e-3)
        self.clip_pram = 0.2
        self.env = env
        torch.manual_seed(336699)
        self.max_steps = 10
        self.max_episodes = 10
        self.ep_reward = []
        self.total_avgr = []
        self.target = False
        self.best_reward = 0
        self.avg_rewards_list = []

    def act(self, state):
        """ Return the action for a given state """
        print("state",state)
        state = torch.from_numpy(np.array([state])).float()
        prob = self.actor(state)
        dist = torch.distributions.Categorical(probs=prob)
        action = dist.sample()
        return action.detach().numpy()

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        """ Calculate actor loss """
        entropy = -(probs * probs.log()).mean()
        ratios = torch.exp(torch.log(probs) - torch.log(old_probs))
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * adv
        loss = -torch.min(surr1, surr2).mean() + 0.001 * entropy + closs
        return loss

    def auxillary_task(self, r, rewards):
        """ Calculate auxiliary task loss (reward prediction) """
        return F.mse_loss(r, rewards)

    def learn(self, states, actions, adv, old_probs, discnt_rewards, rewards):
        """ Learning step for the agent """
        actions = torch.tensor(actions)
        adv = torch.tensor(adv)
        old_probs = torch.tensor(old_probs)
        discnt_rewards = torch.tensor(discnt_rewards)
        rewards = torch.tensor(rewards)

        self.a_opt.zero_grad()
        self.c_opt.zero_grad()

        p, r = self.actor(states, return_reward=True)
        v = self.critic(states)

        td = discnt_rewards - v.squeeze()
        c_loss = 0.5 * td.pow(2).mean()
        a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss) + self.auxillary_task(r, rewards)

        a_loss.backward()
        c_loss.backward()

        self.a_opt.step()
        self.c_opt.step()

        return a_loss.item(), c_loss.item()

    def test_reward(self):
        """ Test the model by running one episode """
        total_reward = 0
        state, info = self.env.reset()
        done = False
        while not done:
            action = self.act(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
        return total_reward

    # Include the other methods (`preprocess1`, `train`, etc.) with appropriate PyTorch modifications.

    def preprocess1(self, states, actions, rewards, dones, values, gamma):
        """ Preprocess transitions for the buffer """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        returns = []
        g = 0
        lmbda = 0.95
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            g = delta + gamma * lmbda * (1 - dones[i]) * g
            returns.insert(0, g + values[i])

        returns = torch.tensor(returns, dtype=torch.float32)
        adv = returns - values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return states, actions, returns, adv, rewards

    def train(self):
        """ Training function for the agent """
        for episode in range(self.max_episodes):
            if self.target:
                break

            state, info = self.env.reset()
            done = False
            all_aloss = []
            all_closs = []
            rewards = []
            states = []
            actions = []
            probs = []
            dones = []
            values = []

            for step in range(self.max_steps):
                action = self.act(state)
                prob = self.actor(torch.from_numpy(np.array([state], dtype=np.float32)))
                value = self.critic(torch.from_numpy(np.array([state], dtype=np.float32)))
                print("action_l", action)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                dones.append(float(not done))
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                probs.append(prob.detach().numpy()[0])
                values.append(value.item())

                state = next_state
                if done:
                    state = self.env.reset()

            # Calculate next state value for the terminal state
            next_value = self.critic(torch.from_numpy(np.array([state], dtype=np.float32))).item()
            values.append(next_value)

            states, actions, returns, advantages, rewards = self.preprocess1(
                states, actions, rewards, dones, values, self.gamma
            )

            # Train for a number of epochs
            for _ in range(1):  # Could be more than one epoch if needed
                a_loss, c_loss = self.learn(
                    states, actions, advantages, torch.tensor(probs), returns, rewards
                )
                all_aloss.append(a_loss)
                all_closs.append(c_loss)

            avg_reward = np.mean([self.test_reward() for _ in range(5)])
            self.avg_rewards_list.append(avg_reward)

        print("Training completed. Average rewards list:", self.avg_rewards_list)
