"""Saturation must allow integral action that brings the output back in range."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.agent.pid import PID


@pytest.mark.parametrize("ki", [1.0, -1.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_saturated_integrator_can_unwind(ki, direction):
    env = SimpleNamespace(action_space=spaces.Box(-2.0, 2.0, (1,), np.float32))
    pid = PID(env, kp=0, ki=ki, kd=0, dt=1)
    pid.integral = 5 * direction / ki
    outputs = [pid.select_action(-direction / ki, 0) for _ in range(5)]
    np.testing.assert_allclose(outputs, direction * np.array([2, 2, 2, 1, 0]))
    assert pid.integral == pytest.approx(0)


@pytest.mark.parametrize("ki", [1.0, -1.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_integration_further_into_saturation_is_blocked(ki, direction):
    env = SimpleNamespace(action_space=spaces.Box(-2.0, 2.0, (1,), np.float32))
    pid = PID(env, kp=0, ki=ki, kd=0, dt=1)
    pid.integral = 2 * direction / ki
    assert pid.select_action(direction / ki, 0) == pytest.approx(2 * direction)
    assert pid.integral == pytest.approx(2 * direction / ki)


def test_unwind_uses_saturated_bound_with_asymmetric_limits():
    env = SimpleNamespace(action_space=spaces.Box(1.0, 3.0, (1,), np.float32))
    pid = PID(env, kp=0, ki=1, kd=0, dt=1)
    pid.integral = 0
    assert pid.select_action(0.5, 0) == pytest.approx(1)
    assert pid.integral == pytest.approx(0.5)
