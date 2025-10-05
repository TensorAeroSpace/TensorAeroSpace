import numpy as np
import pytest
import torch

from tensoraerospace.agent.ppo.model import (
    PPO,
    Actor,
    Critic,
    RunningMeanStd,
    init_layer_uniform,
    ppo_iter,
)


class _DummyEnv:
    def __init__(self):
        self.observation_space = type("S", (), {"shape": (3,)})
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )
        self.unwrapped = self  # Add unwrapped attribute for compatibility

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def test_actor_forward():
    """Test Actor forward pass returns correct shapes and types."""
    actor = Actor(input_dim=3, out_dim=2, hidden_dim=64)
    x = torch.zeros((4, 3))

    action, dist = actor(x)
    assert action.shape == (4, 2), "Action shape should be (batch_size, action_dim)"
    assert isinstance(dist, torch.distributions.Normal), "Distribution should be Normal"

    # Test that actions are bounded by tanh
    assert torch.all(action >= -1.0) and torch.all(
        action <= 1.0
    ), "Actions should be in [-1, 1]"


def test_critic_forward():
    """Test Critic forward pass returns correct shapes."""
    critic = Critic(input_dim=3, hidden_dim=64)
    x = torch.zeros((4, 3))

    v = critic(x)
    assert v.shape == (4, 1), "Value shape should be (batch_size, 1)"


def test_actor_distribution():
    """Test that Actor produces valid probability distributions."""
    actor = Actor(input_dim=3, out_dim=2)
    x = torch.randn((4, 3))

    action, dist = actor(x)

    # Test that we can sample from distribution
    sample = dist.sample()
    assert sample.shape == (4, 2)

    # Test that we can compute log probabilities
    log_prob = dist.log_prob(action)
    assert log_prob.shape == (4, 2)

    # Test that entropy is a valid tensor (can be negative for low variance distributions)
    entropy = dist.entropy()
    assert entropy.shape == (4, 2), "Entropy shape should match action dimensions"
    assert not torch.isnan(entropy).any(), "Entropy should not contain NaN values"


def test_ppo_iter_shapes():
    """Test ppo_iter yields correct batch shapes."""
    n = 8
    states = torch.zeros((n, 3))
    actions = torch.zeros((n, 1))
    log_probs = torch.zeros((n, 1))
    returns = torch.zeros((n, 1))
    advantages = torch.zeros((n, 1))
    rewards = torch.zeros((n, 1))
    values = torch.zeros((n, 1))

    it = ppo_iter(
        epoch=1,
        mini_batch_size=4,
        states=states,
        actions=actions,
        log_probs=log_probs,
        returns=returns,
        advantages=advantages,
        rewards=rewards,
        values=values,
    )
    s, a, lp, r, adv, rew, v = next(it)
    assert s.shape == (4, 3), "States batch shape incorrect"
    assert a.shape == (4, 1), "Actions batch shape incorrect"
    assert lp.shape == (4, 1), "Log probs batch shape incorrect"
    assert r.shape == (4, 1), "Returns batch shape incorrect"
    assert adv.shape == (4, 1), "Advantages batch shape incorrect"
    assert rew.shape == (4, 1), "Rewards batch shape incorrect"
    assert v.shape == (4, 1), "Values batch shape incorrect"


def test_ppo_iter_epochs():
    """Test that ppo_iter produces correct number of batches."""
    n = 16
    states = torch.zeros((n, 3))
    actions = torch.zeros((n, 1))
    log_probs = torch.zeros((n, 1))
    returns = torch.zeros((n, 1))
    advantages = torch.zeros((n, 1))
    rewards = torch.zeros((n, 1))
    values = torch.zeros((n, 1))

    # Test with 2 epochs and mini_batch_size=4
    # Should produce 2 * (16 / 4) = 8 batches
    it = ppo_iter(
        epoch=2,
        mini_batch_size=4,
        states=states,
        actions=actions,
        log_probs=log_probs,
        returns=returns,
        advantages=advantages,
        rewards=rewards,
        values=values,
    )
    batches = list(it)
    assert len(batches) == 8, "Should produce 8 batches (2 epochs * 4 mini-batches)"


def test_ppo_act():
    """Test PPO agent action selection."""
    env = _DummyEnv()
    agent = PPO(env=env, max_episodes=1, rollout_len=8, num_epochs=1, batch_size=4)
    state = np.zeros(3, dtype=np.float32)

    # Test stochastic action
    action, mean, log_prob = agent.act(state, deterministic=False)
    assert action.shape[-1] == 2, "Action dimension should be 2"
    assert mean.shape[-1] == 2, "Mean action dimension should be 2"
    assert log_prob.shape[-1] == 1, "Log prob should be scalar per sample"

    # Test deterministic action
    action_det, mean_det, log_prob_det = agent.act(state, deterministic=True)
    # For deterministic, action should equal mean
    assert torch.allclose(
        action_det, torch.from_numpy(mean_det)
    ), "Deterministic action should equal mean"


def test_ppo_actor_loss():
    """Test PPO actor loss computation."""
    env = _DummyEnv()
    agent = PPO(env=env, max_episodes=1, rollout_len=8, num_epochs=1, batch_size=4)

    # Create dummy tensors
    probs = torch.zeros((4, 1))
    entropy = torch.tensor(0.5)
    actions = torch.zeros((4, 1))
    adv = torch.ones((4, 1))
    old_probs = torch.zeros((4, 1))

    loss = agent.actor_loss(probs, entropy, actions, adv, old_probs)
    assert isinstance(loss.item(), float), "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_init_layer_uniform():
    """Test uniform layer initialization."""
    layer = torch.nn.Linear(10, 5)
    init_layer_uniform(layer, init_w=0.1)

    # Check that weights are within bounds
    assert torch.all(layer.weight.data >= -0.1) and torch.all(layer.weight.data <= 0.1)
    assert torch.all(layer.bias.data >= -0.1) and torch.all(layer.bias.data <= 0.1)


def test_running_mean_std():
    """Test RunningMeanStd statistics tracking."""
    rms = RunningMeanStd(shape=(3,))

    # Initial state
    assert np.allclose(rms.mean, np.zeros(3))
    assert np.allclose(rms.var, np.ones(3))

    # Update with batch
    batch = np.random.randn(10, 3)
    rms.update(batch)

    # Check that stats are updated
    assert not np.allclose(rms.mean, np.zeros(3))
    assert rms.count > 1e-4


def test_running_mean_std_update_from_moments():
    """Test RunningMeanStd update from moments."""
    rms = RunningMeanStd(shape=(2,))

    batch_mean = np.array([1.0, 2.0])
    batch_var = np.array([0.5, 0.5])
    batch_count = 10

    rms.update_from_moments(batch_mean, batch_var, batch_count)

    # After update, mean should be close to batch_mean
    assert rms.count > 1e-4
    assert not np.allclose(rms.mean, np.zeros(2))


def test_ppo_normalize_obs():
    """Test observation normalization."""
    env = _DummyEnv()
    agent = PPO(env=env, normalize_obs=True)

    # Update statistics
    obs = np.array([1.0, 2.0, 3.0])
    agent.obs_rms.update(np.array([obs]))

    # Normalize
    normalized = agent._normalize_obs(obs)
    assert normalized.shape == obs.shape
    assert np.all(
        np.abs(normalized) <= 10.0
    ), "Normalized obs should be clipped to [-10, 10]"


def test_ppo_preprocess():
    """Test PPO preprocess1 method."""
    env = _DummyEnv()
    agent = PPO(env=env, gamma=0.99, gae_lambda=0.95)

    # Create dummy rollout
    n_steps = 4
    states = [torch.randn(3) for _ in range(n_steps)]
    actions = [torch.randn(2) for _ in range(n_steps)]
    rewards = [torch.tensor([1.0]) for _ in range(n_steps)]
    dones = [torch.tensor([0.0]) for _ in range(n_steps)]
    values = [torch.tensor([0.5]) for _ in range(n_steps + 1)]  # n+1 values
    probs = [torch.randn(1) for _ in range(n_steps)]

    states2, actions2, returns2, adv2, rewards2, probs2 = agent.preprocess1(
        states, actions, rewards, dones, values, probs, agent.gamma
    )

    assert states2.shape == (n_steps, 3)
    # actions2 is flattened, so check total elements
    assert (
        actions2.numel() == n_steps * 2
    ), f"Expected {n_steps * 2} elements, got {actions2.numel()}"
    assert len(returns2) == n_steps
    assert adv2.shape[0] == n_steps


def test_ppo_test_reward():
    """Test PPO test_reward evaluation method."""
    env = _DummyEnv()
    agent = PPO(env=env, max_episodes=1, rollout_len=8)

    # Mock environment to return done after 3 steps
    step_count = [0]
    original_step = env.step

    def mock_step(action):
        step_count[0] += 1
        if step_count[0] >= 3:
            return np.zeros(3), 1.0, True, False, {}
        return np.zeros(3), 1.0, False, False, {}

    env.step = mock_step
    reward = agent.test_reward()
    assert reward == 3.0, "Should accumulate reward over 3 steps"
    env.step = original_step


def test_ppo_get_param_env():
    """Test PPO get_param_env method."""
    env = _DummyEnv()
    agent = PPO(
        env=env,
        gamma=0.95,
        max_episodes=10,
        rollout_len=512,
        actor_hidden_dim=128,
        critic_hidden_dim=128,
    )

    params = agent.get_param_env()

    assert "env" in params
    assert "policy" in params
    assert params["policy"]["params"]["gamma"] == 0.95
    assert params["policy"]["params"]["max_episodes"] == 10
    assert params["policy"]["params"]["rollout_len"] == 512
    assert params["policy"]["params"]["actor_hidden_dim"] == 128
    assert params["policy"]["params"]["critic_hidden_dim"] == 128


def test_ppo_with_different_hidden_dims():
    """Test PPO initialization with different hidden dimensions."""
    env = _DummyEnv()
    agent = PPO(env=env, actor_hidden_dim=128, critic_hidden_dim=256)

    assert agent.actor.d1.out_features == 128
    assert agent.critic.d1.out_features == 256


def test_ppo_without_normalization():
    """Test PPO without observation and reward normalization."""
    env = _DummyEnv()
    agent = PPO(env=env, normalize_obs=False, normalize_reward=False)

    obs = np.array([1.0, 2.0, 3.0])
    normalized = agent._normalize_obs(obs)

    # Should return original observation
    assert np.array_equal(normalized, obs)
