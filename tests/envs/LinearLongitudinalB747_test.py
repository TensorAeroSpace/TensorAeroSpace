import pytest
import numpy as np
import gymnasium as gym
from tensoraerospace.envs.b747 import LinearLongitudinalB747  # Import the environment from where it is defined
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step

# Constants for testing
INITIAL_STATE = [[0],[0],[0],[0]]
dt = 0.01  # Дискретизация
tp = generate_time_period(tn=20, dt=dt) # Временной периуд
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp) # Количество временных шагов
REFERENCE_SIGNAL = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал
NUMBER_TIME_STEPS = 1000
INITIAL_STATE_ENV = np.array([0,0])
@pytest.fixture
def env():
    """Fixture to create a fresh instance of the environment for each test."""
    return LinearLongitudinalB747(initial_state=INITIAL_STATE,
                                  reference_signal=REFERENCE_SIGNAL,
                                  number_time_steps=NUMBER_TIME_STEPS)

def test_initialization(env):
    """Test if the environment initializes correctly."""
    assert env.initial_state is not None
    assert env.number_time_steps == NUMBER_TIME_STEPS
    assert env.state_space == ['theta', 'q']
    assert env.control_space == ['stab']
    assert env.output_space == ['theta', 'q']
    assert env.reward_func is not None

def test_reset(env):
    """Test the reset functionality."""
    initial_observation, info = env.reset()
    assert np.array_equal(initial_observation, INITIAL_STATE_ENV)
    assert isinstance(info, dict)

def test_reward_function():
    """Test the reward function."""
    env = LinearLongitudinalB747(initial_state=INITIAL_STATE,
                                 reference_signal=REFERENCE_SIGNAL,
                                 number_time_steps=NUMBER_TIME_STEPS)
    state = np.array([1, 1])
    ts = 10
    reward = env.reward(state, REFERENCE_SIGNAL, ts)
    assert reward == np.abs(state[0] - REFERENCE_SIGNAL[:, ts])

def test_render(env):
    """Test that render raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        env.render()

def test_step(env):
    """Test the step functionality with valid and invalid actions."""
    env.reset()
    action = np.array([0.5])
    next_state, reward, done, _, info = env.step(action)
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, np.ndarray)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
# Run tests with pytest in the command line
if __name__ == "__main__":
    pytest.main()
