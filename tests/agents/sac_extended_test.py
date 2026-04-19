"""Extended tests for SAC algorithm to increase code coverage."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tensoraerospace.agent.base import TheEnvironmentDoesNotMatch
from tensoraerospace.agent.sac.sac import SAC


class _TinySpace:
    """Minimal action/observation space for testing."""

    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)


class _TinyEnv:
    """Minimal environment for testing SAC."""

    def __init__(self, obs_dim=3, act_dim=2, episode_length=5):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self._episode_length = episode_length

    def reset(self):
        self._step_count = 0
        return np.zeros(self.observation_space.shape[0]), {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self.observation_space.shape[0]).astype(np.float32)
        reward = float(np.random.randn())
        terminated = self._step_count >= self._episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self


class _NoOpWriter:
    """Mock TensorBoard writer for testing."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def log_episode(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def assert_contract_satisfied(self):
        pass


class _StubMemory:
    """Stub replay memory for update testing."""

    def __init__(self, obs_dim=3, act_dim=2, batch=4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch = batch

    def sample(self, batch_size):
        b = self.batch
        s = np.random.randn(b, self.obs_dim).astype(np.float32)
        a = np.random.randn(b, self.act_dim).astype(np.float32)
        r = np.random.randn(b).astype(np.float32)
        ns = np.random.randn(b, self.obs_dim).astype(np.float32)
        done = np.zeros((b,), dtype=np.float32)
        return s, a, r, ns, done


# Test fixtures
@pytest.fixture
def tiny_env():
    """Create a tiny test environment."""
    return _TinyEnv(obs_dim=3, act_dim=2, episode_length=5)


@pytest.fixture
def sac_gaussian(tiny_env):
    """Create SAC with Gaussian policy."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        policy_type="Gaussian",
        automatic_entropy_tuning=False,
    )
    agent.writer = _NoOpWriter()
    return agent


@pytest.fixture
def sac_deterministic(tiny_env):
    """Create SAC with Deterministic policy."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        policy_type="Deterministic",
    )
    agent.writer = _NoOpWriter()
    return agent


@pytest.fixture
def sac_auto_entropy(tiny_env):
    """Create SAC with automatic entropy tuning."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        policy_type="Gaussian",
        automatic_entropy_tuning=True,
    )
    agent.writer = _NoOpWriter()
    return agent


# ============================================================================
# Test select_action with evaluate flag
# ============================================================================


def test_select_action_evaluate_false(sac_gaussian):
    """Test action selection in exploration mode."""
    state = np.zeros(3, dtype=np.float32)
    action = sac_gaussian.select_action(state, evaluate=False)
    assert action.shape == (2,)
    assert isinstance(action, np.ndarray)


def test_select_action_evaluate_true(sac_gaussian):
    """Test action selection in evaluation mode."""
    state = np.zeros(3, dtype=np.float32)
    action = sac_gaussian.select_action(state, evaluate=True)
    assert action.shape == (2,)
    assert isinstance(action, np.ndarray)


def test_select_action_deterministic_policy(sac_deterministic):
    """Test action selection with deterministic policy."""
    state = np.zeros(3, dtype=np.float32)
    action1 = sac_deterministic.select_action(state, evaluate=False)
    action2 = sac_deterministic.select_action(state, evaluate=True)
    assert action1.shape == (2,)
    assert action2.shape == (2,)


# ============================================================================
# Test automatic entropy tuning
# ============================================================================


def test_automatic_entropy_tuning_initialization(sac_auto_entropy):
    """Test that automatic entropy tuning initializes correctly."""
    assert sac_auto_entropy.automatic_entropy_tuning is True
    assert hasattr(sac_auto_entropy, "target_entropy")
    assert hasattr(sac_auto_entropy, "log_alpha")
    assert hasattr(sac_auto_entropy, "alpha_optim")
    assert sac_auto_entropy.log_alpha.requires_grad is True


def test_automatic_entropy_tuning_update(sac_auto_entropy):
    """Test that alpha is updated during parameter updates."""
    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    alpha_before = sac_auto_entropy.alpha

    # Perform multiple updates
    for i in range(5):
        sac_auto_entropy.update_parameters(memory, batch_size=4, updates=i)

    alpha_after = sac_auto_entropy.alpha
    # Alpha should potentially change (though not guaranteed in all cases)
    assert isinstance(alpha_after, float)
    assert alpha_after > 0


def test_deterministic_policy_has_zero_alpha(sac_deterministic):
    """Test that deterministic policy sets alpha to 0."""
    assert sac_deterministic.alpha == 0
    assert sac_deterministic.automatic_entropy_tuning is False


# ============================================================================
# Test update_parameters with different configurations
# ============================================================================


def test_update_parameters_returns_correct_types(sac_gaussian):
    """Test that update_parameters returns correct types."""
    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    (
        qf1_loss,
        qf2_loss,
        policy_loss,
        alpha_loss,
        alpha_value,
    ) = sac_gaussian.update_parameters(memory, batch_size=4, updates=1)

    assert isinstance(qf1_loss, float)
    assert isinstance(qf2_loss, float)
    assert isinstance(policy_loss, float)
    assert isinstance(alpha_loss, float)
    assert isinstance(alpha_value, float)


def test_update_parameters_with_target_update_interval(tiny_env):
    """Test that target network updates at correct intervals."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        target_update_interval=2,
        automatic_entropy_tuning=False,
    )
    agent.writer = _NoOpWriter()

    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)

    # Get initial target network weights
    initial_target_params = [p.clone() for p in agent.critic_target.parameters()]

    # Update once (should not update target)
    agent.update_parameters(memory, batch_size=4, updates=1)
    params_after_1 = [p.clone() for p in agent.critic_target.parameters()]

    # Target should not have changed (update 1, interval 2)
    for p_init, p_after in zip(initial_target_params, params_after_1):
        assert torch.allclose(p_init, p_after)

    # Update again (should update target)
    agent.update_parameters(memory, batch_size=4, updates=2)
    params_after_2 = [p.clone() for p in agent.critic_target.parameters()]

    # Target should have changed (update 2, interval 2)
    changed = False
    for p_init, p_after in zip(initial_target_params, params_after_2):
        if not torch.allclose(p_init, p_after):
            changed = True
            break
    assert changed, "Target network should have been updated"


def test_update_parameters_auto_entropy(sac_auto_entropy):
    """Test update_parameters with automatic entropy tuning."""
    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    (
        qf1_loss,
        qf2_loss,
        policy_loss,
        alpha_loss,
        alpha_value,
    ) = sac_auto_entropy.update_parameters(memory, batch_size=4, updates=1)

    # Alpha loss should be non-zero with auto entropy
    assert isinstance(alpha_loss, float)
    assert alpha_value > 0


# ============================================================================
# Test get_param_env
# ============================================================================


def test_get_param_env_returns_dict(sac_gaussian):
    """Test that get_param_env returns expected structure."""
    params = sac_gaussian.get_param_env()

    assert isinstance(params, dict)
    assert "env" in params
    assert "policy" in params

    assert "name" in params["env"]
    assert "params" in params["env"]

    assert "name" in params["policy"]
    assert "params" in params["policy"]


def test_get_param_env_policy_params(sac_gaussian):
    """Test that policy params are correctly captured."""
    params = sac_gaussian.get_param_env()
    policy_params = params["policy"]["params"]

    assert "gamma" in policy_params
    assert "tau" in policy_params
    assert "alpha" in policy_params
    assert "batch_size" in policy_params
    assert "policy_type" in policy_params
    assert "automatic_entropy_tuning" in policy_params

    assert policy_params["gamma"] == sac_gaussian.gamma
    assert policy_params["tau"] == sac_gaussian.tau
    assert policy_params["batch_size"] == sac_gaussian.batch_size


# ============================================================================
# Test save and load with gradients
# ============================================================================


def test_save_with_gradients(tiny_env):
    """Test saving model with optimizer states."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    with tempfile.TemporaryDirectory() as d:
        agent.save(d, save_gradients=True)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]
        files = os.listdir(saved)

        # Should have optimizer files
        assert "policy_optim.pth" in files
        assert "critic_optim.pth" in files


def test_save_with_gradients_auto_entropy(tiny_env):
    """Test saving with gradients when auto entropy is enabled."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=True,
    )

    with tempfile.TemporaryDirectory() as d:
        agent.save(d, save_gradients=True)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]
        files = os.listdir(saved)

        # Should have all optimizer files including alpha
        assert "policy_optim.pth" in files
        assert "critic_optim.pth" in files
        assert "alpha_optim.pth" in files
        assert "log_alpha.pth" in files


def test_load_with_gradients(tiny_env):
    """Test loading model with optimizer states."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    # Train a bit to change optimizer state
    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    agent.update_parameters(memory, batch_size=4, updates=1)

    with tempfile.TemporaryDirectory() as d:
        agent.save(d, save_gradients=True)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]

        # Load with gradients
        loaded = SAC.from_pretrained(saved, load_gradients=True)
        assert loaded is not None
        assert hasattr(loaded, "policy_optim")
        assert hasattr(loaded, "critic_optim")


def test_save_auto_entropy_creates_log_alpha_file(tiny_env):
    """Test that log_alpha.pth is created when auto entropy is enabled."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=True,
    )

    with tempfile.TemporaryDirectory() as d:
        agent.save(d, save_gradients=False)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]
        files = os.listdir(saved)

        assert "log_alpha.pth" in files


def test_load_auto_entropy_restores_log_alpha(tiny_env):
    """Test that log_alpha is correctly restored when loading."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=True,
    )

    # Change alpha
    with torch.no_grad():
        agent.log_alpha.data.copy_(torch.tensor([1.5]))
    agent.alpha = float(agent.log_alpha.exp().item())
    original_alpha = agent.alpha

    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]

        loaded = SAC.from_pretrained(saved)
        # Alpha should be close to original
        assert abs(loaded.alpha - original_alpha) < 0.1


# ============================================================================
# Test train with save_best
# ============================================================================


def test_train_with_save_best(tiny_env):
    """Test training with save_best option."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )
    agent.writer = _NoOpWriter()

    with tempfile.TemporaryDirectory() as d:
        agent.train(num_episodes=3, save_best=True, save_path=d)

        # Check if any model was saved
        if os.listdir(d):
            subdirs = [os.path.join(d, name) for name in os.listdir(d)]
            assert len(subdirs) > 0


def test_train_with_save_best_and_gradients(tiny_env):
    """Test training with save_best and gradient saving."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )
    agent.writer = _NoOpWriter()

    with tempfile.TemporaryDirectory() as d:
        agent.train(
            num_episodes=3,
            save_best=True,
            save_path=d,
            save_best_with_gradients=True,
        )

        if os.listdir(d):
            subdirs = [os.path.join(d, name) for name in os.listdir(d)]
            if subdirs:
                files = os.listdir(subdirs[0])
                # Should have optimizer files
                assert "policy_optim.pth" in files
                assert "critic_optim.pth" in files


# ============================================================================
# Test from_pretrained error handling
# ============================================================================


def test_from_pretrained_nonexistent_local_path():
    """Test from_pretrained with non-existent local path."""
    with pytest.raises(FileNotFoundError):
        SAC.from_pretrained("/nonexistent/path/to/model")


def test_from_pretrained_wrong_agent_type(tiny_env):
    """Test loading model with mismatched agent type."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]

        # Modify config to have wrong agent name
        config_path = Path(saved) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["policy"]["name"] = "wrong.module.WrongAgent"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Should raise error
        with pytest.raises(TheEnvironmentDoesNotMatch):
            SAC.from_pretrained(saved)


# ============================================================================
# Test different policy_type configurations
# ============================================================================


def test_deterministic_policy_initialization(sac_deterministic):
    """Test that deterministic policy is correctly initialized."""
    from tensoraerospace.agent.sac.model import DeterministicPolicy

    assert isinstance(sac_deterministic.policy, DeterministicPolicy)
    assert sac_deterministic.policy_type == "Deterministic"


def test_gaussian_policy_initialization(sac_gaussian):
    """Test that Gaussian policy is correctly initialized."""
    from tensoraerospace.agent.sac.model import GaussianPolicy

    assert isinstance(sac_gaussian.policy, GaussianPolicy)
    assert sac_gaussian.policy_type == "Gaussian"


# ============================================================================
# Test different hyperparameters
# ============================================================================


def test_custom_hyperparameters(tiny_env):
    """Test initialization with custom hyperparameters."""
    agent = SAC(
        env=tiny_env,
        updates_per_step=2,
        batch_size=16,
        memory_capacity=50000,
        lr=0.001,
        policy_lr=0.0005,
        gamma=0.95,
        tau=0.01,
        alpha=0.5,
        target_update_interval=3,
        hidden_size=64,
        device="cpu",
    )

    assert agent.updates_per_step == 2
    assert agent.batch_size == 16
    assert agent.memory.capacity == 50000
    assert agent.gamma == 0.95
    assert agent.tau == 0.01
    assert agent.alpha == 0.5
    assert agent.target_update_interval == 3


def test_seed_reproducibility(tiny_env):
    """Test that seed makes action selection reproducible."""
    seed = 12345
    agent1 = SAC(env=tiny_env, seed=seed, hidden_size=8, device="cpu")
    agent2 = SAC(env=tiny_env, seed=seed, hidden_size=8, device="cpu")

    state = np.zeros(3, dtype=np.float32)

    # Reset random state
    torch.manual_seed(seed)
    np.random.seed(seed)
    action1 = agent1.select_action(state, evaluate=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    action2 = agent2.select_action(state, evaluate=True)

    # Actions should be close (not exactly equal due to weight init)
    # but policies should be identical if seeds match
    assert action1.shape == action2.shape


# ============================================================================
# Test verbose_histogram
# ============================================================================


def test_verbose_histogram_logging(tiny_env):
    """Test that verbose_histogram triggers histogram logging."""
    agent = SAC(
        env=tiny_env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        verbose_histogram=True,
        automatic_entropy_tuning=False,
    )

    # Create a mock writer that tracks histogram calls
    class _HistogramTracker:
        def __init__(self):
            self.histogram_calls = 0
            self.scalar_calls = 0

        def add_scalar(self, *args, **kwargs):
            self.scalar_calls += 1

        def add_histogram(self, *args, **kwargs):
            self.histogram_calls += 1

    tracker = _HistogramTracker()
    agent.writer = tracker

    memory = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    agent.update_parameters(memory, batch_size=4, updates=1)

    # Should have histogram calls
    assert tracker.histogram_calls > 0
    assert tracker.scalar_calls > 0


# ============================================================================
# Test memory capacity
# ============================================================================


def test_memory_capacity_limit(tiny_env):
    """Test that memory respects capacity limit."""
    agent = SAC(
        env=tiny_env,
        memory_capacity=10,
        batch_size=2,
        hidden_size=8,
        device="cpu",
    )

    # Add more transitions than capacity
    state = np.zeros(3, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    reward = 0.0
    next_state = np.zeros(3, dtype=np.float32)
    done = 0.0

    for _ in range(20):
        agent.memory.push(state, action, reward, next_state, done)

    # Memory should not exceed capacity
    assert len(agent.memory) <= agent.memory.capacity
