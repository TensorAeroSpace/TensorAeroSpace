"""Smoke + analytical tests for the nonlinear 6-DoF quadrotor model.

Validates the ODE in five regimes, each with a closed-form expectation:

1. **Hover.** ``T = m·g`` and zero torque must hold the state at the
   initial point (rigid-body equilibrium).
2. **Free fall without drag.** ``z_e(t) = 0.5·g·t²`` (textbook).
3. **Free fall with linear drag.** Terminal-velocity solution
   ``z(t) = v_∞·t + v_∞·τ·(e^{-t/τ} - 1)``.
4. **Single-step roll torque.** First-order Euler-equivalent step gives
   ``p ≈ τ_x · dt / Jx``.
5. **Gyroscopic coupling.** With ``q = r = 1, p = 0`` the ODE must
   produce ``p_dot = (Jy - Jz) · q · r / Jx`` (Newton-Euler cross term).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.quadrotor.nonlinear import NonlinearQuadrotor
from tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics import (
    quadrotor_ode_6dof,
)
from tensoraerospace.aerospacemodel.quadrotor.nonlinear.params import (
    default_parameters,
)


def _zero_state() -> np.ndarray:
    return np.zeros(12, dtype=np.float64)


def test_hover_equilibrium_is_exact():
    """T = m·g, no torque, level attitude → state stays at zero forever."""
    m = NonlinearQuadrotor(x0=_zero_state(), dt=0.01, integrator="rk4")
    u_hover = np.array([m.hover_thrust, 0.0, 0.0, 0.0])
    for _ in range(1000):  # 10 s
        m.run_step(u_hover)
    assert (
        np.max(np.abs(m.current_state)) < 1e-9
    ), "hover should be a numerically exact equilibrium"


def test_free_fall_without_drag_matches_textbook():
    """T = 0, drag disabled → z_e(t) = 0.5·g·t² exactly (RK4)."""
    m = NonlinearQuadrotor(x0=_zero_state(), dt=0.001, integrator="rk4")
    p = default_parameters()
    p.kdx = p.kdy = p.kdz = 0.0
    m.set_param(p)
    u_zero = np.zeros(4)
    for _ in range(10_000):  # 10 s
        m.run_step(u_zero)
    z = float(m.current_state[2])
    expected = 0.5 * p.g * 10.0**2
    assert abs(z - expected) / expected < 1e-3


def test_free_fall_with_linear_drag_matches_analytic():
    """T = 0 with default drag → matches terminal-velocity ODE solution."""
    m = NonlinearQuadrotor(x0=_zero_state(), dt=0.001, integrator="rk4")
    u_zero = np.zeros(4)
    for _ in range(10_000):  # 10 s
        m.run_step(u_zero)
    z = float(m.current_state[2])

    p = m.param
    v_term = p.m * p.g / p.kdz  # terminal velocity
    tau = p.m / p.kdz  # 1/e time-constant
    analytic = v_term * 10.0 + v_term * tau * (np.exp(-10.0 / tau) - 1.0)
    assert abs(z - analytic) / analytic < 1e-3


def test_single_step_roll_torque_produces_correct_angular_acceleration():
    """Small dt with τ_x = const → p[1] ≈ τ_x · dt / Jx (linear regime)."""
    m = NonlinearQuadrotor(x0=_zero_state(), dt=1e-4, integrator="rk4")
    tau_x = 0.01
    m.run_step(np.array([m.hover_thrust, tau_x, 0.0, 0.0]))
    expected_p = tau_x * m.dt / m.param.Jx
    p = float(m.current_state[9])
    assert abs(p - expected_p) < 1e-9


def test_gyro_coupling_term_in_angular_dynamics():
    """ODE produces the (Jy-Jz)·q·r / Jx coupling term verbatim."""
    p_param = default_parameters()
    state = _zero_state()
    state[10] = 1.0  # q
    state[11] = 1.0  # r
    dx = quadrotor_ode_6dof(state, np.zeros(4), 0.0, p_param)
    expected = (p_param.Jy - p_param.Jz) * 1.0 * 1.0 / p_param.Jx
    assert abs(dx[9] - expected) < 1e-12


def test_state_size_validation():
    with pytest.raises(ValueError, match="x0 must have 12"):
        NonlinearQuadrotor(x0=np.zeros(7))


def test_control_size_validation():
    m = NonlinearQuadrotor(x0=_zero_state())
    with pytest.raises(ValueError, match="Размерность"):
        m.run_step(np.zeros(3))


def test_run_step_returns_state():
    m = NonlinearQuadrotor(x0=_zero_state())
    out = m.run_step(np.array([m.hover_thrust, 0.0, 0.0, 0.0]))
    assert out.shape == (12, 1)


def test_history_grows_per_step():
    m = NonlinearQuadrotor(x0=_zero_state())
    initial_len = len(m.x_history)
    for _ in range(5):
        m.run_step(np.zeros(4))
    assert len(m.x_history) == initial_len + 5
    assert len(m.u_history) == 5


def test_euler_and_rk4_both_supported():
    for integrator in ("euler", "rk4"):
        m = NonlinearQuadrotor(x0=_zero_state(), integrator=integrator)
        u_hover = np.array([m.hover_thrust, 0.0, 0.0, 0.0])
        for _ in range(100):
            m.run_step(u_hover)
        # Hover must be preserved by both — Euler may introduce noise
        # at machine epsilon level but not drift.
        assert np.max(np.abs(m.current_state)) < 1e-9


def test_unknown_integrator_raises():
    with pytest.raises(ValueError, match="unknown integrator"):
        NonlinearQuadrotor(x0=_zero_state(), integrator="midpoint")  # type: ignore[arg-type]
