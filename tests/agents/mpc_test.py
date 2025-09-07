import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.mpc.stochastic import MPCAgent, Net


# Фикстура для создания среды и агента
@pytest.fixture
def setup_mpc_agent():
    def example_cost_function(state, action):
        theta = state[0, 0].item()
        theta_dot = state[0, 1].item()
        return theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)

    env = gym.make("Pendulum-v1")
    model = Net(env.action_space.shape[0], env.observation_space.shape[0])
    agent = MPCAgent(
        gamma=0.99,
        action_dim=env.action_space.shape[0],
        observation_dim=env.observation_space.shape[0],
        model=model,
        cost_function=example_cost_function,
        env=env,
    )
    return agent


def test_collect_data(setup_mpc_agent):
    agent = setup_mpc_agent
    states, actions, next_states = agent.collect_data(num_episodes=1)
    assert len(states) == len(actions) == len(next_states) == 200
    assert states.shape == (200, 3)
    assert actions.shape == (200, 1)
    assert next_states.shape == (200, 3)


def test_train_model(setup_mpc_agent):
    agent = setup_mpc_agent
    states, actions, next_states = agent.collect_data(num_episodes=1)
    agent.train_model(states, actions, next_states, epochs=1)
    assert agent.system_model_optimizer.param_groups[0]["lr"] == 1e-3


def test_choose_action(setup_mpc_agent):
    agent = setup_mpc_agent
    state = np.array([0.5, 0.3, -0.1])
    action = agent.choose_action(state, rollout=5, horizon=3)
    assert action.shape == (1, 1)


def test_test_model(setup_mpc_agent):
    agent = setup_mpc_agent
    rewards = agent.test_model(num_episodes=5, rollout=3, horizon=2)
    assert len(rewards) == 5


def test_test_network(setup_mpc_agent):
    agent = setup_mpc_agent
    states = np.random.random((100, 3))
    actions = np.random.random((100, 1))
    next_states = np.random.random((100, 3))

    agent.test_network(states, actions, next_states)
