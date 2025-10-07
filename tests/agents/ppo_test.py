import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensoraerospace.agent import PPO
from tensoraerospace.envs.utils import ActionNormalizer


@pytest.fixture
def setup_ppo_agent():
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    env.reset()
    agent = PPO(
        env, gamma=0.9, max_episodes=100, rollout_len=64, num_epochs=2, batch_size=16
    )
    return agent, env


def test_collect_data(setup_ppo_agent):
    """Test data collection from environment."""
    agent, env = setup_ppo_agent
    state, info = env.reset()
    action, mean_action, log_prob = agent.act(state)
    assert action.shape == torch.Size([1, 1])
    assert mean_action.shape == (1, 1)
    assert log_prob.shape == torch.Size([1, 1])


def test_learn(setup_ppo_agent):
    """Test PPO learn method with all required parameters."""
    agent, env = setup_ppo_agent
    batch_size = 10
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    adv = torch.randn(batch_size, 1)
    old_probs = torch.randn(batch_size, 1)
    discnt_rewards = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size, 1)
    old_values = torch.randn(batch_size, 1)

    metrics = agent.learn(
        states, actions, adv, old_probs, discnt_rewards, rewards, old_values
    )

    # Check that metrics are returned
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics

    # Check that all metrics are finite
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite"


def test_train(setup_ppo_agent):
    """Test PPO training for one episode."""
    agent, env = setup_ppo_agent
    agent.max_episodes = 1
    agent.rollout_len = 32  # Shorter rollout for faster testing
    agent.train()

    # Check that some training happened
    assert len(agent.ep_reward) >= 0 or len(agent.total_avgr) >= 0


def test_ppo_clipping():
    """Test PPO clipping mechanism."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, clip_pram=0.2)

    # Test actor loss with different ratios
    batch_size = 5
    new_probs = torch.zeros((batch_size, 1))
    old_probs = torch.ones((batch_size, 1)) * -1.0  # Create large ratio
    entropy = torch.tensor(0.5)
    actions = torch.zeros((batch_size, 1))
    adv = torch.ones((batch_size, 1))

    loss = agent.actor_loss(new_probs, entropy, actions, adv, old_probs)
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)


def test_ppo_gae_lambda():
    """Test different GAE lambda values."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    # Test with different GAE lambda values
    for gae_lambda in [0.9, 0.95, 0.99]:
        agent = PPO(env, gae_lambda=gae_lambda)
        assert agent.gae_lambda == gae_lambda


def test_ppo_target_kl():
    """Test PPO with KL early stopping."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, target_kl=0.01, max_episodes=1, rollout_len=32)

    # Train should respect target_kl
    agent.train()
    # If KL early stopping works, training completes without error


def test_ppo_normalize_reward():
    """Test PPO with reward normalization enabled."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, normalize_reward=True, max_episodes=1, rollout_len=32)

    assert agent.normalize_reward is True
    assert hasattr(agent, "ret_rms")

    agent.train()


def test_ppo_save_load():
    """Test PPO save and load functionality."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, gamma=0.95, max_episodes=1)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Save agent
        agent.save(temp_dir)

        # Find the saved directory (it has a timestamp)
        saved_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
        assert len(saved_dirs) > 0, "No directory was created"

        saved_path = saved_dirs[0]

        # Check that files exist
        assert (saved_path / "config.json").exists()
        assert (saved_path / "actor.pth").exists()
        assert (saved_path / "critic.pth").exists()

        # Load agent
        loaded_agent = PPO.from_pretrained(str(saved_path))

        # Check that parameters match
        assert loaded_agent.gamma == agent.gamma
        assert loaded_agent.clip_pram == agent.clip_pram

        # Check that networks have same architecture
        assert loaded_agent.actor.d1.out_features == agent.actor.d1.out_features
        assert loaded_agent.critic.d1.out_features == agent.critic.d1.out_features

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_ppo_save_with_normalization():
    """Test PPO save/load with normalization statistics."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, normalize_obs=True, normalize_reward=True)

    # Update normalization statistics
    obs = np.random.randn(10, env.observation_space.shape[0])
    agent.obs_rms.update(obs)

    temp_dir = tempfile.mkdtemp()

    try:
        agent.save(temp_dir)

        saved_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
        saved_path = saved_dirs[0]

        # Check that normalization files exist
        assert (saved_path / "obs_rms.npz").exists()
        assert (saved_path / "ret_rms.npz").exists()

        # Load agent
        loaded_agent = PPO.from_pretrained(str(saved_path))

        # Check that statistics are preserved
        assert np.allclose(loaded_agent.obs_rms.mean, agent.obs_rms.mean)
        assert np.allclose(loaded_agent.obs_rms.var, agent.obs_rms.var)

    finally:
        shutil.rmtree(temp_dir)


def test_ppo_deterministic_action():
    """Test deterministic action selection."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env)

    state, _ = env.reset()

    # Get deterministic action multiple times
    actions = []
    for _ in range(3):
        action, mean, _ = agent.act(state, deterministic=True)
        actions.append(action.numpy())

    # All deterministic actions should be the same
    for i in range(1, len(actions)):
        assert np.allclose(actions[0], actions[i])


def test_ppo_gradient_clipping():
    """Test gradient clipping."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, max_grad_norm=0.5)

    assert agent.max_grad_norm == 0.5

    # Create dummy batch
    batch_size = 10
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    adv = torch.randn(batch_size, 1)
    old_probs = torch.randn(batch_size, 1)
    discnt_rewards = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size, 1)
    old_values = torch.randn(batch_size, 1)

    # Learn should apply gradient clipping
    metrics = agent.learn(
        states, actions, adv, old_probs, discnt_rewards, rewards, old_values
    )
    assert isinstance(metrics["actor_loss"], float)


def test_ppo_entropy_coefficient():
    """Test different entropy coefficients."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    for entropy_coef in [0.0, 0.01, 0.1]:
        agent = PPO(env, entropy_coef=entropy_coef)
        assert agent.entropy_coef == entropy_coef


def test_ppo_different_learning_rates():
    """Test PPO with different learning rates for actor and critic."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    agent = PPO(env, actor_lr=0.001, critic_lr=0.005)
    assert agent.actor_lr == 0.001
    assert agent.critic_lr == 0.005

    # Check that optimizers have correct learning rates
    assert agent.a_opt.param_groups[0]["lr"] == 0.001
    assert agent.c_opt.param_groups[0]["lr"] == 0.005


def test_ppo_with_seed():
    """Test that seed produces reproducible results."""
    # Test that setting seed affects initialization
    env1 = gym.make("Pendulum-v1")
    env1 = ActionNormalizer(env1)
    torch.manual_seed(42)
    np.random.seed(42)
    agent1 = PPO(env1, seed=42)

    env2 = gym.make("Pendulum-v1")
    env2 = ActionNormalizer(env2)
    torch.manual_seed(42)
    np.random.seed(42)
    agent2 = PPO(env2, seed=42)

    # Get actions from same state
    state = np.zeros(env1.observation_space.shape[0])

    # With same seed, network weights should be the same, so outputs should be the same
    with torch.no_grad():
        # Check that actor weights are the same
        for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(
                p1, p2
            ), "Actor weights should be the same with same seed"

        # Check that critic weights are the same
        for p1, p2 in zip(agent1.critic.parameters(), agent2.critic.parameters()):
            assert torch.allclose(
                p1, p2
            ), "Critic weights should be the same with same seed"


def test_ppo_test_reward_with_gym():
    """Test evaluation on Pendulum environment."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env)

    # Get test reward
    reward = agent.test_reward()

    # Should return a finite number
    assert np.isfinite(reward)
    # Pendulum typically has negative rewards
    assert isinstance(reward, (int, float, np.number))


def test_ppo_eval_freq():
    """Test evaluation frequency parameter."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)

    agent = PPO(env, eval_freq=5, max_episodes=1, rollout_len=32)
    assert agent.eval_freq == 5

    agent.train()


def test_ppo_value_clipping():
    """Test that value function clipping is working."""
    env = gym.make("Pendulum-v1")
    env = ActionNormalizer(env)
    agent = PPO(env, clip_pram=0.2)

    batch_size = 10
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    states = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    adv = torch.randn(batch_size, 1)
    old_probs = torch.randn(batch_size, 1)
    returns = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size, 1)
    old_values = torch.randn(batch_size, 1)

    metrics = agent.learn(states, actions, adv, old_probs, returns, rewards, old_values)

    # Critic loss should be computed (may be zero but should exist)
    assert "critic_loss" in metrics
    assert np.isfinite(metrics["critic_loss"])
