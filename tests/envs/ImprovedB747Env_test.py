import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

INITIAL_STATE = [[0], [0], [0], [0]]
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
REFERENCE_SIGNAL = np.reshape(
    unit_step(degree=5, tp=tp, time_step=0.1, output_rad=True), [1, -1]
)
NUMBER_TIME_STEPS = 1000


@pytest.fixture
def env_setup_default():
    return ImprovedB747Env(
        initial_state=INITIAL_STATE,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )


def test_initialization(env_setup_default):
    env = env_setup_default
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (4,)
    assert env.current_step == 0
    assert env.number_time_steps == NUMBER_TIME_STEPS


def test_reset(env_setup_default):
    env = env_setup_default
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_step_shapes_and_types(env_setup_default):
    env = env_setup_default
    obs, info = env.reset()
    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_clamping(env_setup_default):
    env = env_setup_default
    env.reset()
    high_action = np.array([2.0], dtype=np.float32)  # above 1.0 -> clamp to 1.0
    _, _, _, _, _ = env.step(high_action)
    assert env.previous_action == pytest.approx(1.0, rel=0, abs=1e-6)


def test_truncation_flag():
    small_steps_env = ImprovedB747Env(
        initial_state=INITIAL_STATE,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=2,  # very small to hit truncation quickly
        dt=dt,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )
    small_steps_env.reset()
    _, _, terminated, truncated, _ = small_steps_env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    # For number_time_steps=2, truncated becomes True at current_step>=1
    assert truncated is True


def test_initial_action_override_on_first_step():
    env = ImprovedB747Env(
        initial_state=INITIAL_STATE,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        initial_elevator_deg=10.0,  # normalized = 0.4
        use_initial_action_on_first_step=True,
    )
    env.reset()
    input_action = np.array([-1.0], dtype=np.float32)
    _, _, _, _, _ = env.step(input_action)
    # previous_action should reflect initial_elevator_deg on first step, not input
    assert env.previous_action == pytest.approx(
        10.0 / env.max_stabilizer_angle_deg, rel=0, abs=1e-6
    )


def test_get_init_args(env_setup_default):
    env = env_setup_default
    init_args = env.get_init_args()

    # Check that the method returns a dictionary
    assert isinstance(init_args, dict)

    # Check that expected keys are present
    assert "initial_state" in init_args
    assert "reference_signal" in init_args
    assert "number_time_steps" in init_args
    assert "dt" in init_args
    assert "initial_elevator_deg" in init_args
    assert "use_initial_action_on_first_step" in init_args

    # Check that internal keys are removed
    assert "self" not in init_args
    assert "__class__" not in init_args

    # Verify values
    assert init_args["number_time_steps"] == NUMBER_TIME_STEPS
    assert init_args["dt"] == dt
    assert init_args["initial_elevator_deg"] == 0.0
    assert init_args["use_initial_action_on_first_step"] is False
