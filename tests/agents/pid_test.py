import datetime

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


def test_pid_select_action():
    env = gym.make("Pendulum-v1")
    pid = PID(env=env, kp=1, ki=1, kd=0.5, dt=0.01)
    setpoint = 10
    measurement = 7
    control_signal = pid.select_action(setpoint, measurement)
    assert control_signal == pytest.approx(153.03)


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


def test_pid_save_and_load(monkeypatch):
    env = gym.make("Pendulum-v1")
    pid = PID(env=env, kp=1, ki=1, kd=0.5, dt=0.01)

    monkeypatch.chdir("/tmp/mock_model")
    pid.save("/tmp/mock_model")
    date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    date_str = date_str + "_" + PID.__name__

    loaded_pid = PID.from_pretrained(f"/tmp/mock_model/{date_str}")
    assert pid.kp == loaded_pid.kp
    assert pid.ki == loaded_pid.ki
    assert pid.kd == loaded_pid.kd
    assert pid.dt == loaded_pid.dt
    assert pid.integral == loaded_pid.integral
    assert pid.prev_error == loaded_pid.prev_error
