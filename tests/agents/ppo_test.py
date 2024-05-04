import pytest
import numpy as np
import torch
from tensoraerospace.agent import PPO
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)


        return action
    
@pytest.fixture
def setup_ppo_agent():
    env = gym.make('Pendulum-v1')
    env = ActionNormalizer(env)
    env.reset()
    agent = PPO(env, gamma=0.9, max_episodes = 100)
    return agent, env


def test_collect_data(setup_ppo_agent):
    agent, env = setup_ppo_agent
    state, info = env.reset()
    action, mean_action, log_prob = agent.act(state)
    assert action.shape == torch.Size([1, 1])
    assert mean_action.shape == torch.Size([1, 1])
    assert log_prob.shape == torch.Size([1, 1])

# Тестирование функции learn
def test_learn(setup_ppo_agent):
    agent, env = setup_ppo_agent
    states = torch.randn(10, env.observation_space.shape[0])
    actions = torch.randint(env.action_space.shape[0], (10, 1))
    adv = torch.randn(10, 1)
    old_probs = torch.randn(10, 1)
    discnt_rewards = torch.randn(10, 1)
    rewards = torch.randn(10, 1)
    agent.learn(states, actions, adv, old_probs, discnt_rewards, rewards)

# Тестирование функции train
def test_train(setup_ppo_agent):
    agent, env = setup_ppo_agent
    agent.max_episodes = 1
    agent.train()
