import gymnasium as gym
import pytest

from tensoraerospace.agent.pid import PID


def test_pid_init():
    env = gym.make("Pendulum-v1")
    pid = PID(env=env, kp=1, ki=1, kd=0.5, dt=0.01)
    assert pid.kp == 1
    assert pid.ki == 1
    assert pid.kd == 0.5
    assert pid.dt == 0.01
    assert pid.integral == 0
    assert pid.prev_error == 0
    assert pid.prev_measurement == 0  # New: derivative on measurement


def test_pid_select_action():
    """Test PID control signal calculation.

    Uses derivative on measurement (Simulink default):
    - derivative = -(measurement - prev_measurement) / dt

    This avoids derivative kick on setpoint changes.
    """
    # Test without env to avoid saturation clipping
    pid = PID(env=None, kp=1.0, ki=0.1, kd=0.5, dt=1.0)

    # Initialize prev_measurement to simulate steady state before step
    pid.prev_measurement = 5.0

    setpoint = 10
    measurement = 7  # Measurement increased from 5 to 7

    control_signal = pid.select_action(setpoint, measurement)

    # error = 10 - 7 = 3
    # derivative = -(7 - 5) / 1.0 = -2 (negative because measurement is rising)
    # integral = 0 + 3 * 1.0 = 3
    # output = 1*3 + 0.1*3 + 0.5*(-2) = 3 + 0.3 - 1 = 2.3
    assert control_signal == pytest.approx(2.3)


def test_pid_select_action_with_saturation():
    """Test PID with action space saturation and anti-windup."""
    env = gym.make("Pendulum-v1")  # action_space: [-2, 2]
    pid = PID(env=env, kp=10.0, ki=1.0, kd=0.0, dt=0.01)

    # Large error should saturate output
    control_signal = pid.select_action(setpoint=100, measurement=0)

    # Output should be clipped to action_space bounds
    assert control_signal == pytest.approx(2.0)  # Clipped to max


def test_pid_derivative_on_measurement():
    """Test that derivative is computed on measurement, not error.

    This prevents 'derivative kick' when setpoint changes suddenly.
    """
    pid = PID(env=None, kp=0.0, ki=0.0, kd=1.0, dt=1.0)

    # First call: prev_measurement=0, measurement=5
    # derivative = -(5 - 0) / 1.0 = -5
    output1 = pid.select_action(setpoint=10, measurement=5)
    assert output1 == pytest.approx(-5.0)

    # Second call: prev_measurement=5, measurement=5 (no change)
    # derivative = -(5 - 5) / 1.0 = 0
    output2 = pid.select_action(setpoint=20, measurement=5)  # Setpoint changed!
    assert output2 == pytest.approx(0.0)  # No derivative kick


def test_pid_integral_accumulation():
    """Test that integral term accumulates correctly."""
    pid = PID(env=None, kp=0.0, ki=1.0, kd=0.0, dt=1.0)
    pid.prev_measurement = 0  # Avoid derivative term effects

    # First call: error=2, integral=2
    pid.select_action(setpoint=2, measurement=0)
    assert pid.integral == pytest.approx(2.0)

    # Second call: error=2, integral=4
    pid.select_action(setpoint=2, measurement=0)
    assert pid.integral == pytest.approx(4.0)


def test_pid_reset():
    """Test PID reset functionality."""
    pid = PID(env=None, kp=1.0, ki=1.0, kd=1.0, dt=0.01)

    # Make some calls to accumulate state
    pid.select_action(10, 5)
    pid.select_action(10, 6)

    # State should be non-zero
    assert pid.integral != 0
    assert pid.prev_measurement != 0

    # Reset
    pid.reset()

    # State should be zero again
    assert pid.integral == 0
    assert pid.prev_error == 0
    assert pid.prev_measurement == 0


def test_pid_get_param_env():
    env = gym.make("Pendulum-v1")
    pid = PID(env=env, kp=1, ki=1, kd=0.5, dt=0.01)
    params = pid.get_param_env()
    assert "env" in params
    assert "policy" in params
    assert params["policy"]["name"] == "tensoraerospace.agent.pid.PID"
    assert params["policy"]["params"]["ki"] == 1
    assert params["policy"]["params"]["kp"] == 1
    assert params["policy"]["params"]["kd"] == 0.5
    assert params["policy"]["params"]["dt"] == 0.01


def test_pid_save_and_load():
    env = gym.make("Pendulum-v1")
    pid = PID(env=env, kp=1, ki=1, kd=0.5, dt=0.01)

    # Save and get the path to saved model
    saved_dir = pid.save("/tmp/mock_model_pid")

    # Load from the saved directory
    loaded_pid = PID.from_pretrained(str(saved_dir))
    assert pid.kp == loaded_pid.kp
    assert pid.ki == loaded_pid.ki
    assert pid.kd == loaded_pid.kd
    assert pid.dt == loaded_pid.dt
    assert pid.integral == loaded_pid.integral
    assert pid.prev_error == loaded_pid.prev_error
