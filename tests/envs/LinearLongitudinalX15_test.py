import pytest
import numpy as np
from tensoraerospace.envs import LinearLongitudinalX15  # Import the environment from where it is defined
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step
from gymnasium import spaces

INITIAL_STATE = [[0],[0],[0], [0]]
dt = 0.01  # Дискретизация
tp = generate_time_period(tn=20, dt=dt) # Временной периуд
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp) # Количество временных шагов
REFERENCE_SIGNAL = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал
NUMBER_TIME_STEPS = 1000
INITIAL_STATE_ENV = np.array([0,0])

@pytest.fixture
def env_setup():
    return LinearLongitudinalX15(initial_state=INITIAL_STATE, 
                                 reference_signal=REFERENCE_SIGNAL, 
                                 number_time_steps=NUMBER_TIME_STEPS)

def test_initialization(env_setup):
    env = env_setup
    assert len(env.initial_state) == 4, "Initial state shape should match input."
    assert isinstance(env.action_space, spaces.Box), "Action space should be a Box space."
    assert isinstance(env.observation_space, spaces.Box), "Observation space should be a Box space."
    assert env.current_step == 0, "Initial step should be zero."
    assert not env.done, "Initial done should be False."

def test_step_function(env_setup):
    env = env_setup
    action = np.array([10], dtype=np.float32)  # within the valid range
    next_state, reward, done, _, info = env.step(action)
    assert isinstance(next_state, np.ndarray), "Next state should be a numpy array."
    assert isinstance(reward, np.ndarray), "Reward should be a float."
    assert isinstance(done, bool), "Done should be a boolean."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert next_state.shape == (2,), "Next state should have two dimensions by default."


def test_reset_function(env_setup):
    env = env_setup
    env.step(np.array([10], dtype=np.float32))  # change state
    assert env.current_step > 0, "Step should have advanced."
    returned = env.reset()
    assert len(returned) == 2, "Reset state should have two dimensions by default."
    state, info = env.reset()
    assert env.current_step == 0, "Reset should set step back to zero."
    assert not env.done, "Reset should set done to False."
    assert state.shape == (2,), "Reset state should have two dimensions by default."

