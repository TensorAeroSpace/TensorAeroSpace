# tests.py

import datetime
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest
import torch

from tensoraerospace.agent.a2c.model import (  # Assume importing A2C and necessary exceptions
    A2C,
    Actor,
    Critic,
    Mish,
)


@pytest.fixture
def mock_environment():
    # Create a mock environment with necessary attributes for A2C
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions, activation=Mish)
    critic = Critic(state_dim, activation=Mish)

    model = A2C(
        env=env,
        actor=actor,
        critic=critic,
        gamma=0.9,
        entropy_beta=0.01,
        actor_lr=4e-4,
        critic_lr=4e-3,
    )
    return model


def test_get_param_env(mock_environment):
    param_env = mock_environment.get_param_env()

    assert (
        param_env["env"]["name"]
        == "gymnasium.envs.classic_control.pendulum.PendulumEnv"
    )
    assert (
        param_env["policy"]["name"]
        == f"{mock_environment.__module__}.{mock_environment.__class__.__name__}"
    )


def test_save(mock_environment):
    path = "/tmp/mock_model"

    # Call save
    mock_environment.save(path)

    # Check directory existence
    saved_dirs = os.listdir(path)
    model_dir = Path(path) / saved_dirs[0]  # Access the first item directly
    assert model_dir.exists()

    # Check config file
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    assert (
        config["policy"]["name"]
        == f"{mock_environment.__module__}.{mock_environment.__class__.__name__}"
    )


def test_from_pretrained(mock_environment, monkeypatch):
    monkeypatch.chdir("/tmp/mock_model")
    mock_environment.save("/tmp/mock_model")
    date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    date_str = date_str + "_" + A2C.__name__
    loaded_model = A2C.from_pretrained(f"/tmp/mock_model/{date_str}")
    assert loaded_model.gamma == mock_environment.gamma
    assert loaded_model.entropy_beta == mock_environment.entropy_beta
    assert loaded_model.actor_lr == mock_environment.actor_lr
    assert loaded_model.critic_lr == mock_environment.critic_lr
    assert loaded_model.max_grad_norm == mock_environment.max_grad_norm
    assert loaded_model.seed == mock_environment.seed
    # TODO Добавить проверку на то, что модель загрузилась Critic и Actor
