import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.lapan import ImprovedLAPANEnv


@pytest.fixture
def lapan_env_default():
    # LAPAN state order: [u, w, q, theta]
    initial_state = np.array([50.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    return ImprovedLAPANEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )


def test_improved_lapan_initialization(lapan_env_default):
    env = lapan_env_default
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (4,)
    assert env.current_step == 0


def test_improved_lapan_reset(lapan_env_default):
    env = lapan_env_default
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_improved_lapan_step_shapes_and_types(lapan_env_default):
    env = lapan_env_default
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_improved_lapan_action_clamping(lapan_env_default):
    env = lapan_env_default
    env.reset()
    env.step(np.array([2.0], dtype=np.float32))  # clamp to 1.0
    assert env.previous_action == pytest.approx(1.0, abs=1e-6)


def test_improved_lapan_truncation_flag():
    initial_state = np.array([50.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 3), dtype=np.float32)
    env = ImprovedLAPANEnv(
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


def test_improved_lapan_initial_action_override_on_first_step():
    initial_state = np.array([50.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    env = ImprovedLAPANEnv(
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
        np.deg2rad(10.0) / env.max_elevator_angle_rad, abs=1e-6
    )


def test_improved_lapan_get_init_args(lapan_env_default):
    env = lapan_env_default
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


def test_improved_lapan_termination_on_pitch_limit(monkeypatch, lapan_env_default):
    env = lapan_env_default
    env.reset()

    def _forced_run_step(action):
        # Force theta beyond pitch envelope
        return np.array([50.0, 0.0, 0.0, env.max_pitch_rad * 1.1], dtype=np.float32)

    monkeypatch.setattr(env.model, "run_step", _forced_run_step)
    _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert reward == -100.0





