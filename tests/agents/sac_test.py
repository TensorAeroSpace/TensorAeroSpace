import pytest
import gymnasium as gym
import numpy as np

from tensoraerospace.agent import SAC  # Make sure to update the path to your SAC class.
from tensoraerospace.agent.sac import ReplayMemory

@pytest.fixture
def sac_instance():
    """Fixture for creating an instance of SAC algorithm."""
    env = gym.make("Pendulum-v1")
    return SAC(env=env)

@pytest.fixture
def memory():
    """Fixture for creating a replay memory."""
    return ReplayMemory(10000, seed=42)

def test_action_selection(sac_instance):
    """Test action selection of the SAC model."""
    env = sac_instance.env
    state, info = env.reset()
    action = sac_instance.select_action(state)
    assert env.action_space.contains(action), "Selected action is not valid."

def test_training(sac_instance):
    """Test the training process of the SAC model."""
    num_episodes = 5

    # Mock training
    sac_instance.train(num_episodes)

    assert sac_instance.writer, "No training writer found."

