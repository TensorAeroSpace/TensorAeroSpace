import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.v = init_layer_uniform(self.v)

    def forward(self, input_data):
        x = F.relu(self.d1(input_data))
        v = self.v(x)
        return v

class Actor(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=128):
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
        x = F.relu(self.d1(input_data))

        if continous_actions:
            mu = torch.tanh(self.mu(x))
            log_std = torch.tanh(self.delta(x))
            log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            std = torch.exp(log_std)

            # print(input_data)
            # print(mu)
            # print(std)


            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            # if(torch.any(dist.log_prob(action) > 0)):
            #     print(dist.log_prob(action))
            if return_reward:
                r = torch.flatten(F.relu(self.r(x)))
                return action, dist, r
            return action, dist
        a = F.softmax(self.a(x), dim=-1)
        if return_reward:
            r = torch.flatten(F.relu(self.r(x)))
            return a, r
        return a

def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], \
            log_probs[rand_ids], returns[rand_ids], advantages[rand_ids], \
            rewards[rand_ids]

class Agent():
    """ Класс агента ppo using PyTorch

    Args:
        env: Environment object
        gamma (float): Discount factor
    """
    def __init__(self, env, gamma=0.99):
        self.gamma = gamma
        self.env = env
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
        self.critic = Critic(env.observation_space.shape[0])      
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=0.005)
        self.clip_pram = 0.2
        torch.manual_seed(336699)
        self.rollout_len = 2048
        self.max_episodes = 100000
        self.num_epochs = 64
        self.batch_size = 64
        self.entropy_coef = 0.005
        self.ep_reward = []
        self.total_avgr = []
        self.target = False
        self.best_reward = 0
        self.avg_rewards_list = []

    def act(self, state):
        """ Return the action for a given state """
        # print("state",state)
        state = torch.from_numpy(np.array([state])).float()
        action, dist = self.actor(state, continous_actions=True)
        # dist = torch.distributions.Categorical(probs=prob)
        # dist = torch.distributions.Normal(mu, delta)
        # action = dist.sample()
        # prob = dist.log_prob(action)
        return action.detach().numpy()[0], dist.mean.detach().numpy(), dist.log_prob(action).detach().numpy()#, prob.detach().numpy()

    def actor_loss(self, probs, entropy, actions, adv, old_probs):
        """ Calculate actor loss """
        # entropy = -probs.entropy().mean()
        # print(probs.log_prob(actions).size())
        # print(old_probs.log_prob(actions).size())
        # print(probs)
        # print(old_probs)
        ratios = torch.exp(probs - old_probs)
        # print(probs)
        #$ print(probs[probs > 1])
        # print(probs[probs < 0])
        # print(old_probs)
        # print(old_probs[old_probs > 1])
        # print(old_probs[old_probs < 0])
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1.0 - self.clip_pram, 1.0 + self.clip_pram) * adv
        loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * entropy
        return loss

    def auxillary_task(self, r, rewards):
        """ Calculate auxiliary task loss (reward prediction) """
        return F.mse_loss(r, rewards)

    def learn(self, states, actions, adv, old_probs, discnt_rewards, rewards):
        """ Learning step for the agent """
        actions = torch.tensor(actions)
        adv = torch.tensor(adv)
        #old_probs = torch.tensor(old_probs)
        discnt_rewards = torch.tensor(discnt_rewards)
        rewards = torch.tensor(rewards)

        self.a_opt.zero_grad()
        self.c_opt.zero_grad()

        # print(torch.Tensor(mus).size())
        # print(torch.Tensor(deltas).size())

        new_actions, new_distr, r = self.actor(states, return_reward=True, continous_actions=True)
        # print(torch.Tensor(new_mus).size())
        # print(torch.Tensor(new_deltas).size())
        new_probs = new_distr.log_prob(actions)
        v = self.critic(states)

        td = discnt_rewards - v.squeeze()
        c_loss = 0.5 * td.pow(2).mean()
        # print(c_loss)
        a_loss = self.actor_loss(new_probs, -new_distr.entropy().mean(), actions, adv.detach(), old_probs)# + self.auxillary_task(r, rewards)
        # print(a_loss)

        a_loss.backward()
        c_loss.backward()

        self.a_opt.step()
        self.c_opt.step()


        return a_loss.item(), c_loss.item()

    def test_reward(self):
        """ Test the model by running one episode """
        total_reward = 0
        reset_return = self.env.reset()
        if type(reset_return) is tuple:
            state, info = reset_return
        else:
            state = reset_return
        done = False
        while not done:
            action, mean_action, delta = self.act(state)
            step_return = self.env.step(mean_action)
            if len(step_return) > 4:
                next_state, reward, terminated, trunkated, info = step_return
                done = terminated or trunkated
            else:
                next_state, reward, terminated, info = step_return
                done = terminated
            total_reward += reward
        return total_reward

    # Include the other methods (`preprocess1`, `train`, etc.) with appropriate PyTorch modifications.

    def preprocess1(self, states, actions, rewards, dones, values, probs, gamma):
        """ Preprocess transitions for the buffer """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)

        returns = []
        g = 0
        lmbda = 0.8
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            g = delta + gamma * lmbda * (1 - dones[i]) * g
            returns.insert(0, g + values[i])

        returns = torch.tensor(returns, dtype=torch.float32)
        adv = returns - values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return states, actions, returns, adv, rewards, probs

    def train(self):
        """ Training function for the agent """
        for episode in range(self.max_episodes):
            if self.target:
                break

            reset_return = self.env.reset()
            if type(reset_return) is tuple:
                state, info = reset_return
            else:
                state = reset_return
            done = False
            all_aloss = []
            all_closs = []
            rewards = []
            states = []
            actions = []
            probs = []
            # mus = []
            # deltas = []
            dones = []
            values = []

            for step in range(self.rollout_len):
                action, mu, prob = self.act(state)
                # prob = self.actor(torch.from_numpy(np.array([state], dtype=np.float32)))
                value = self.critic(torch.from_numpy(np.array([state], dtype=np.float32)))
                # print("action_l", action)
                #print(action)
                step_return = self.env.step(action)
                if len(step_return) > 4:
                    next_state, reward, terminated, trunkated, info = step_return
                    done = terminated or trunkated
                else:
                    next_state, reward, terminated, info = step_return
                    done = terminated
                dones.append(float(not done))
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                probs.append(prob[0])
                # mus.append(mu)
                # deltas.append(delta)
                values.append(value.item())

                state = next_state
                if done:
                    reset_return = self.env.reset()
                    if type(reset_return) is tuple:
                        state, info = reset_return
                    else:
                        state = reset_return

            # Calculate next state value for the terminal state
            next_value = self.critic(torch.from_numpy(np.array([state], dtype=np.float32))).item()
            values.append(next_value)

            states, actions, returns, advantages, rewards, probs = self.preprocess1(
                states, actions, rewards, dones, values, probs, self.gamma
            )

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
            # for _ in range(1):  # Could be more than one epoch if needed
            #     a_loss, c_loss = self.learn(
            #         states, actions, advantages, probs, returns, rewards
            #    )
            #    all_aloss.append(a_loss)
            #    all_closs.append(c_loss)
            #print(a_loss, c_loss)

            avg_reward = np.mean([self.test_reward() for _ in range(5)])
            print(avg_reward)
            self.avg_rewards_list.append(avg_reward)

        print("Training completed. Average rewards list:", self.avg_rewards_list)
