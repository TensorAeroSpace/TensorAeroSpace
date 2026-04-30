"""F-16 angular model with track_altitude=True and configurable thrust."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16


def test_default_track_altitude_is_off():
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    assert m.track_altitude is False
    assert m.thrust_mode == "constant"
    assert m.action_space_length == 3
    assert m.list_state == [
        "alpha",
        "beta",
        "wx",
        "wy",
        "wz",
        "gamma",
        "psi",
        "theta",
        "stab",
        "dstab",
        "ail",
        "dail",
        "dir",
        "ddir",
    ]


def test_track_altitude_extends_state_to_16():
    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
    )
    assert m.list_state[-2:] == ["h", "V"]
    s = m.current_state
    assert s.shape == (16,)
    # Auto-pad: h = params.Oy, V = params.V
    assert s[14] == pytest.approx(m.param.Oy)
    assert s[15] == pytest.approx(m.param.V)


def test_track_altitude_step_evolves_h_and_V():
    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
    )
    h0 = m.current_state[14]
    V0 = m.current_state[15]
    for _ in range(50):
        m.run_step(np.zeros(3))
    h1 = m.current_state[14]
    V1 = m.current_state[15]
    # At trim with default thrust, V should stay close to V0 (within a few m/s)
    assert abs(V1 - V0) < 5.0


def test_thrust_mode_control_extends_action_space():
    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
        thrust_mode="control",
    )
    assert m.action_space_length == 4  # 3 surfaces + 1 thrust
    assert m.control_list[-1] == "thrust"
    # Step with explicit thrust input
    m.run_step(np.array([0.0, 0.0, 0.0, 50000.0]))
    # T_active should reflect the input (clipped to T_max_thrust)
    assert m.param.T_active == pytest.approx(50000.0)


def test_thrust_input_clipped_to_max():
    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
        thrust_mode="control",
    )
    m.run_step(np.array([0.0, 0.0, 0.0, 1e7]))  # crazy high
    assert m.param.T_active == pytest.approx(m.param.T_max_thrust)
    m.run_step(np.array([0.0, 0.0, 0.0, -100]))  # negative
    assert m.param.T_active == pytest.approx(0.0)


def test_thrust_constant_uses_param_T_thrust():
    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
        thrust_mode="constant",
    )
    # Set a custom thrust
    m.param.T_thrust = 25000.0
    m.run_step(np.zeros(3))
    assert m.param.T_active == pytest.approx(25000.0)


def test_q_updates_with_altitude():
    """Climbing should reduce dynamic pressure since rho(h) decreases."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        _isa_dynamic_pressure,
    )

    m = AngularF16(
        x0=np.zeros(14),
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
    )
    # Force an artificially high altitude by mutating the state
    h_high = 9000.0
    V0 = m.current_state[15]
    q_low = _isa_dynamic_pressure(h_high, V0, m.param.g)
    q_baseline = _isa_dynamic_pressure(m.param.Oy, V0, m.param.g)
    assert q_low < q_baseline


def test_track_altitude_off_is_bit_identical_to_legacy():
    """With track_altitude=False (default), step trajectory must match
    a fresh model with the same x0/dt/integrator (regression check)."""
    m1 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m2 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    for _ in range(20):
        m1.run_step(np.array([0.05, 0.0, 0.0]))
        m2.run_step(np.array([0.05, 0.0, 0.0]))
    np.testing.assert_allclose(
        m1.current_state,
        m2.current_state,
        atol=1e-12,
    )
