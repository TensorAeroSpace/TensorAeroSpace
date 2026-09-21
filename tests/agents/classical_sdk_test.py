"""Reusable PID actuator feedback and discrete optimal-feedback contracts."""

import numpy as np
import pytest

from tensoraerospace.agent.lqr import LQRAgent
from tensoraerospace.agent.pid import PID, LateralAircraftPID


def test_external_measurement_rate_prevents_derivative_kick():
    pid = PID(kp=0, ki=0, kd=2, dt=0.1)
    assert pid.select_action(50, 5, measurement_rate=0.2) == pytest.approx(-0.4)
    assert pid.select_action(-50, 5, measurement_rate=0.2) == pytest.approx(-0.4)


@pytest.mark.parametrize("sign", [-1, 1])
def test_slew_limited_pid_uses_actual_feedback_and_avoids_integral_windup(sign):
    pid = PID(kp=sign, ki=sign, kd=0, dt=0.1, output_limits=(-8, 8), rate_limit=2)
    for _ in range(30):
        assert pid.select_action(5, 0, applied_output=sign * 0.3) == pytest.approx(
            sign * 0.5
        )
    assert pid.integral == 0
    pid.reset()
    assert pid.prev_output == 0
    assert pid.select_action(0, 0, applied_output=0) == 0


def test_lateral_pid_rejects_incomplete_feedback_instead_of_dropping_an_axis():
    pid = LateralAircraftPID()
    state = np.zeros(12)
    with pytest.raises(ValueError, match="two finite"):
        pid.command(state, applied_output=[0])


def test_lqr_riccati_identity_and_closed_loop_decay():
    agent = LQRAgent([[1, 0.1], [0, 1]], [[0.005], [0.1]], np.eye(2), [[0.1]])
    A, B, K, P = agent.A, agent.B, agent.K, agent.P
    np.testing.assert_allclose(
        (A - B @ K).T @ P @ (A - B @ K) + agent.Q + K.T @ agent.R @ K, P, atol=1e-10
    )
    assert max(abs(np.linalg.eigvals(A - B @ K))) < 1
    state = np.array([1.0, 0.0])
    for _ in range(150):
        state = A @ state + B @ agent.predict(state)
    assert np.linalg.norm(state) < 1e-5


@pytest.mark.parametrize("Q,R", [([[-1]], [[1]]), ([[1]], [[0]]), ([[np.nan]], [[1]])])
def test_lqr_rejects_invalid_cost(Q, R):
    with pytest.raises(ValueError):
        LQRAgent([[1]], [[1]], Q, R)
