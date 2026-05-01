import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.comsat import (  # Import the environment from where it is defined
    ComSatEnv,
)
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

# Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
INITIAL_STATE = [[6371.0], [0.0], [0.001]]
dt = 0.01  # Дискретизация
tp = generate_time_period(tn=20, dt=dt)  # Временной периуд
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)  # Количество временных шагов
# Reference signal for angular velocity control
REFERENCE_SIGNAL = np.reshape(
    unit_step(degree=0.1, tp=tp, time_step=0.1, output_rad=True), [1, -1]
)  # Заданный сигнал угловой скорости
NUMBER_TIME_STEPS = 1000
INITIAL_STATE_ENV = np.array([6371.0, 0.0, 0.001])


@pytest.fixture
def env_setup():
    return ComSatEnv(
        initial_state=INITIAL_STATE,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def test_initialization(env_setup):
    env = env_setup
    assert len(env.initial_state) == 3, "Initial state shape should match input."
    assert isinstance(
        env.action_space, spaces.Box
    ), "Action space should be a Box space."
    assert isinstance(
        env.observation_space, spaces.Box
    ), "Observation space should be a Box space."
    assert env.current_step == 0, "Initial step should be zero."
    assert not env.done, "Initial done should be False."


def test_step_function(env_setup):
    env = env_setup
    action = np.array([10], dtype=np.float32)  # within the valid range
    next_state, reward, done, _, info = env.step(action)
    assert isinstance(next_state, np.ndarray), "Next state should be a numpy array."
    assert isinstance(reward, float), "Reward should be a float."
    assert reward <= 0, "Reward should be non-positive (negative tracking error)."
    assert isinstance(done, bool), "Done should be a boolean."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert next_state.shape == (3,), "Next state should have shape (3,)."


def test_reset_function(env_setup):
    env = env_setup
    env.step(np.array([10], dtype=np.float32))  # change state
    assert env.current_step > 0, "Step should have advanced."
    returned = env.reset()
    assert len(returned) == 2, "Reset should return state and info."
    state, info = env.reset()
    assert env.current_step == 0, "Reset should set step back to zero."
    assert not env.done, "Reset should set done to False."
    assert state.shape == (3,), "Reset state should have shape (3,)."


def test_render_modes(capsys):
    env = ComSatEnv(
        initial_state=INITIAL_STATE,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        render_mode="human",
    )

    assert env.render() is None
    assert "ComSatEnv" in capsys.readouterr().out

    env.reset()
    env.step(np.array([10], dtype=np.float32))
    snapshot = env.render(mode="ansi")
    assert isinstance(snapshot, str)
    assert "step=1" in snapshot
    assert "action=[10]" in snapshot


def test_invalid_render_mode_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        ComSatEnv(
            initial_state=INITIAL_STATE,
            reference_signal=REFERENCE_SIGNAL,
            number_time_steps=NUMBER_TIME_STEPS,
            render_mode="rgb_array",
        )
