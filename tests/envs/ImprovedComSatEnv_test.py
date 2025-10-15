"""Tests for ImprovedComSatEnv."""

import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs import ImprovedComSatEnv
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period


@pytest.fixture
def env_setup():
    """Create test environment with reference signal."""
    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)

    # Reference signal
    reference_signals = unit_step(
        degree=0.001, tp=tp, time_step=10, output_rad=True
    ).reshape(1, -1)

    # Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
    initial_state = np.array([6371.0, 0.0, 0.001])

    return ImprovedComSatEnv(
        initial_state=initial_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        dt=dt,
    )


def test_initialization(env_setup):
    """Test environment initialization."""
    env = env_setup
    assert len(env.initial_state) == 3, "Initial state should have 3 elements"
    assert isinstance(env.action_space, spaces.Box), "Action space should be Box"
    assert isinstance(
        env.observation_space, spaces.Box
    ), "Observation space should be Box"
    assert env.action_space.shape == (1,), "Action space should be (1,)"
    assert env.observation_space.shape == (4,), "Observation space should be (4,)"
    assert env.current_step == 0, "Initial step should be 0"

    # Check action space bounds
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)

    # Check observation space bounds
    assert np.allclose(env.observation_space.low, -1.0)
    assert np.allclose(env.observation_space.high, 1.0)


def test_reset_function(env_setup):
    """Test environment reset functionality."""
    env = env_setup

    # Take a few steps
    for _ in range(5):
        env.step(np.array([0.1], dtype=np.float32))

    assert env.current_step > 0, "Step should have advanced"

    # Reset
    obs, info = env.reset()

    assert env.current_step == 0, "Step should be reset to 0"
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (4,), "Observation should have shape (4,)"
    assert isinstance(info, dict), "Info should be a dictionary"

    # Check that observation is normalized
    assert np.all(obs >= -1.0) and np.all(
        obs <= 1.0
    ), "Observation should be in [-1, 1]"


def test_step_function(env_setup):
    """Test step function."""
    env = env_setup
    env.reset()

    # Take a step with normalized action
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Check return types
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert isinstance(reward, float), "Reward should be float"
    assert isinstance(terminated, bool), "Terminated should be bool"
    assert isinstance(truncated, bool), "Truncated should be bool"
    assert isinstance(info, dict), "Info should be dict"

    # Check shapes and bounds
    assert obs.shape == (4,), "Observation should have shape (4,)"
    assert np.all(obs >= -1.0) and np.all(
        obs <= 1.0
    ), "Observation should be in [-1, 1]"

    # Check that step advanced
    assert env.current_step == 1, "Step should advance"


def test_action_clipping(env_setup):
    """Test that actions are clipped to valid range."""
    env = env_setup
    env.reset()

    # Try action outside bounds
    large_action = np.array([5.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(large_action)

    # Should not crash and should return valid observation
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)

    # Take another step with negative large action
    large_negative_action = np.array([-5.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(large_negative_action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)


def test_termination_conditions(env_setup):
    """Test that environment terminates on constraint violations."""
    env = env_setup
    env.reset()

    # Apply large actions to potentially trigger termination
    max_steps = 200
    for i in range(max_steps):
        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            # Check that termination penalty was applied
            assert reward == -100.0, "Termination should give penalty reward"
            break

        if truncated:
            # Episode ended normally
            break

    # At least one termination condition should be met
    assert terminated or truncated, "Episode should terminate within max_steps"


def test_reward_structure(env_setup):
    """Test reward function components."""
    env = env_setup
    env.reset()

    # Take several steps and collect rewards
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    # Rewards should be finite
    assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"

    # Most rewards should be negative (cost function)
    # unless we're perfectly tracking (unlikely with random actions)
    assert np.mean(rewards) < 0, "Average reward with random policy should be negative"


def test_state_indices(env_setup):
    """Test state index properties."""
    env = env_setup

    assert env._idx_rho == 0
    assert env._idx_rho_dot == 1
    assert env._idx_theta_dot == 2


def test_initial_action_handling(env_setup):
    """Test that initial action is handled correctly."""
    dt = 0.01
    tp = generate_time_period(tn=20, dt=dt)
    number_time_steps = len(tp)
    reference_signals = unit_step(
        degree=0.001, tp=tp, time_step=10, output_rad=True
    ).reshape(1, -1)
    initial_state = np.array([6371.0, 0.0, 0.001])

    # Create environment with initial thrust and flag enabled
    env = ImprovedComSatEnv(
        initial_state=initial_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        dt=dt,
        initial_thrust=5.0,
        use_initial_action_on_first_step=True,
    )

    env.reset()

    # First step should use initial thrust
    action = np.array([0.0], dtype=np.float32)
    obs1, reward1, _, _, _ = env.step(action)

    # Subsequent steps should use agent action
    obs2, reward2, _, _, _ = env.step(action)

    # The observations should be different
    # (initial thrust applied vs zero thrust)
    assert not np.allclose(
        obs1, obs2
    ), "Observations should differ when different thrusts are applied"


def test_get_init_args(env_setup):
    """Test retrieval of initialization arguments."""
    env = env_setup
    init_args = env.get_init_args()

    assert isinstance(init_args, dict), "Should return dict"
    assert "initial_state" in init_args
    assert "reference_signal" in init_args
    assert "number_time_steps" in init_args
    assert "dt" in init_args
    assert "self" not in init_args, "Should not include 'self'"
    assert "__class__" not in init_args, "Should not include '__class__'"


def test_normalization_parameters(env_setup):
    """Test that normalization parameters are set correctly."""
    env = env_setup

    assert env.max_angular_velocity > 0
    assert env.max_radial_velocity > 0
    assert env.max_radial_position_deviation > 0
    assert env.max_thrust > 0
    assert env.nominal_rho > 0

    # Check reward weights are positive
    assert env.w_theta_dot > 0
    assert env.w_rho > 0
    assert env.w_rho_dot >= 0
    assert env.w_action >= 0
    assert env.w_smooth >= 0
    assert env.w_jerk >= 0


def test_multiple_episodes(env_setup):
    """Test running multiple episodes."""
    env = env_setup

    for episode in range(3):
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 50

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Each episode should run at least a few steps
        assert steps > 0, f"Episode {episode} should take at least 1 step"
