import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.rocket import ImprovedMissileEnv


@pytest.fixture
def missile_env_default():
    # Missile model state order: [u, w, q, theta]
    initial_state = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    return ImprovedMissileEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )


def test_improved_missile_initialization(missile_env_default):
    env = missile_env_default
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (4,)
    assert env.current_step == 0


def test_improved_missile_reset(missile_env_default):
    env = missile_env_default
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_improved_missile_step_shapes_and_types(missile_env_default):
    env = missile_env_default
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_improved_missile_action_clamping(missile_env_default):
    env = missile_env_default
    env.reset()
    env.step(np.array([2.0], dtype=np.float32))  # clamp to 1.0
    assert env.previous_action == pytest.approx(1.0, abs=1e-6)


def test_improved_missile_truncation_flag():
    initial_state = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 3), dtype=np.float32)
    env = ImprovedMissileEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=3,
        dt=0.01,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )
    env.reset()
    _, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert truncated is True


def test_improved_missile_initial_action_override_on_first_step():
    initial_state = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    env = ImprovedMissileEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=10.0,
        use_initial_action_on_first_step=True,
    )
    env.reset()
    env.step(np.array([-1.0], dtype=np.float32))
    assert env.previous_action == pytest.approx(
        10.0 / env.max_elevator_angle_deg, abs=1e-6
    )


def test_improved_missile_get_init_args(missile_env_default):
    env = missile_env_default
    d = env.get_init_args()
    assert isinstance(d, dict)
    assert "initial_state" in d
    assert "reference_signal" in d
    assert "number_time_steps" in d
    assert "dt" in d
    assert "initial_elevator_deg" in d
    assert "use_initial_action_on_first_step" in d
    assert "self" not in d
    assert "__class__" not in d


def test_improved_missile_termination_on_pitch_limit(monkeypatch, missile_env_default):
    env = missile_env_default
    env.reset()

    def _forced_run_step(action):
        return np.array([100.0, 0.0, 0.0, env.max_pitch_rad * 1.1], dtype=np.float32)

    monkeypatch.setattr(env.model, "run_step", _forced_run_step)
    _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert reward == -100.0
