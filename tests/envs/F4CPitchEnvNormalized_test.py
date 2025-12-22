import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.f4c import F4CPitchEnvNormalized


@pytest.fixture
def f4c_env_default():
    # F4C state order: [u, w, q, theta] (per env docs)
    initial_state = np.array([200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    return F4CPitchEnvNormalized(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )


def test_f4c_pitch_env_normalized_initialization(f4c_env_default):
    env = f4c_env_default
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (4,)
    assert env.current_step == 0


def test_f4c_pitch_env_normalized_reset(f4c_env_default):
    env = f4c_env_default
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_f4c_pitch_env_normalized_step_shapes_and_types(f4c_env_default):
    env = f4c_env_default
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_f4c_pitch_env_normalized_action_clamping(f4c_env_default):
    env = f4c_env_default
    env.reset()
    env.step(np.array([2.0], dtype=np.float32))  # clamp to 1.0
    assert env.previous_action == pytest.approx(1.0, abs=1e-6)


def test_f4c_pitch_env_normalized_truncation_flag():
    initial_state = np.array([200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 3), dtype=np.float32)
    env = F4CPitchEnvNormalized(
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


def test_f4c_pitch_env_normalized_initial_action_override_on_first_step():
    initial_state = np.array([200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    env = F4CPitchEnvNormalized(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=10.0,
        use_initial_action_on_first_step=True,
    )
    env.reset()
    env.step(np.array([-1.0], dtype=np.float32))
    # initial action is stored in radians and normalized by max_elevator_angle_rad
    assert env.previous_action == pytest.approx(
        np.deg2rad(10.0) / env.max_elevator_angle_rad, abs=1e-6
    )


def test_f4c_pitch_env_normalized_get_init_args(f4c_env_default):
    env = f4c_env_default
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


def test_f4c_pitch_env_normalized_termination_on_pitch_limit(monkeypatch, f4c_env_default):
    env = f4c_env_default
    env.reset()

    def _forced_run_step(action):
        return np.array([200.0, 0.0, 0.0, env.max_pitch_rad * 1.1], dtype=np.float32)

    monkeypatch.setattr(env.model, "run_step", _forced_run_step)
    _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert reward == -100.0





