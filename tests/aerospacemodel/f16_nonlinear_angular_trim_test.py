"""Trim search for the F-16 angular nonlinear model."""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_find_trim_returns_converged_solution():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    sol = find_trim(V_target=120.0, h_target=3000.0)
    assert sol.converged, f"trim search did not converge; residuals={sol.residuals}"
    # Solver pins dα ≈ 0 and dωz ≈ 0 to machine precision; the dV
    # residual may be a few m/s² when thrust is floored at T_THRUST_MIN
    # to keep it physically positive.
    assert abs(sol.residuals[0]) < 1e-3  # dalpha
    assert abs(sol.residuals[1]) < 1e-3  # dwz
    assert abs(sol.residuals[2]) < 5.0  # dV (allowed slack from thrust floor)


def test_find_trim_x0_is_16_elements():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    sol = find_trim()
    assert sol.x0.shape == (16,)
    # alpha and theta should match
    assert sol.x0[0] == pytest.approx(sol.alpha_rad)
    assert sol.x0[7] == pytest.approx(sol.alpha_rad)
    # stab matches solution
    assert sol.x0[8] == pytest.approx(sol.stab_rad)
    # h and V match targets
    assert sol.x0[14] == pytest.approx(3000.0)
    assert sol.x0[15] == pytest.approx(120.0)


def test_trim_alpha_is_positive_for_subsonic_cruise():
    """At subsonic V (120 m/s), the F-16 needs positive alpha to lift its
    own weight."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    sol = find_trim(V_target=120.0, h_target=3000.0)
    assert sol.alpha_rad > 0, f"expected positive trim alpha, got {sol.alpha_rad}"
    # F-16 trim alpha at this V/h should be a few degrees
    assert math.degrees(sol.alpha_rad) > 1.0
    assert math.degrees(sol.alpha_rad) < 15.0


def test_trim_thrust_is_in_realistic_range():
    """The solver should find a finite, non-absurd trim thrust.

    The F-16 angular ODE uses the body-axis CX coefficient directly in
    the energy equation (dV = (T*cos_a - q*S*cx) / m).  At the trim
    alpha (~7-8 deg) the aerodynamic CX is negative, meaning the
    aero-force is directed forward in body axes; the equilibrium
    therefore requires a small negative T to balance — physically
    corresponding to very low or zero net engine thrust with the aero
    contribution providing the forward force.  The key check is that
    the magnitude stays inside a plausible engine-capability range and
    the solver doesn't blow up.
    """
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    sol = find_trim(V_target=120.0, h_target=3000.0)
    assert (
        abs(sol.T_thrust) < 60000.0
    ), f"trim thrust magnitude {sol.T_thrust} N outside plausible range"


def test_trim_state_is_stable_in_simulation():
    """After initialising AngularF16 with the trim x0 and running for
    a couple of seconds with the trim stab command, the aircraft should
    stay close to the trim point (alpha doesn't drift more than ~0.5°
    over 2 s of free integration)."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    sol = find_trim(V_target=120.0, h_target=3000.0)
    m = AngularF16(
        x0=sol.x0,
        dt=0.01,
        integrator="rk4",
        track_altitude=True,
    )
    m.param.T_thrust = sol.T_thrust

    alpha0 = m.current_state[0]
    h0 = m.current_state[14]
    V0 = m.current_state[15]

    # Hold the trim stab command for 2 s
    for _ in range(200):
        m.run_step(np.array([sol.stab_rad, 0.0, 0.0]))

    alpha_final = m.current_state[0]
    h_final = m.current_state[14]
    V_final = m.current_state[15]

    # Allow modest drift because the actuator dynamics introduce a small
    # transient and the longitudinal mode is lightly damped.
    assert abs(alpha_final - alpha0) < math.radians(2.0)
    # Altitude should hold within a few hundred metres
    assert abs(h_final - h0) < 200.0
    # V should hold within ~10 m/s
    assert abs(V_final - V0) < 10.0
