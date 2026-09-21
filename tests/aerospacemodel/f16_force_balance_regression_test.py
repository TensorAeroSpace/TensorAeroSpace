"""Check F-16 wind-axis rates against Cartesian Newton equations in body NED."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import dynamics
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
    _isa_dynamic_pressure,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim


@pytest.mark.parametrize(
    "alpha,beta,roll,pitch",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.15, 0.0, 0.0, 0.15),
        (0.1, 0.08, 0.2, 0.3),
        (-0.05, -0.1, -0.25, 0.1),
    ],
)
@pytest.mark.parametrize("thrust", [0.0, 12000.0])
def test_velocity_derivatives_equal_newton_equations(alpha, beta, roll, pitch, thrust):
    p = F16AngularParameters(T_thrust=thrust)
    x = np.zeros(16)
    x[:8] = [alpha, beta, 0.07, -0.03, 0.02, roll, 0.0, pitch]
    x[8] = -0.06
    x[14:] = [3000.0, 120.0]
    dx = dynamics.f16_ode_6dof(x, np.array([x[8], 0.0, 0.0]), 0.0, p)
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    v = x[15] * np.array([ca * cb, sb, sa * cb])
    # Differentiate v(V, alpha, beta), independently of the model's equations.
    jacobian = np.array(
        [
            [ca * cb, -x[15] * sa * cb, -x[15] * ca * sb],
            [sb, 0.0, x[15] * cb],
            [sa * cb, x[15] * ca * cb, -x[15] * sa * sb],
        ]
    )
    actual = jacobian @ dx[[15, 0, 1]]
    qS = _isa_dynamic_pressure(x[14], x[15], p.g) * p.S
    cx = dynamics.get_cx(alpha, beta, x[8], p.lef, x[4], x[15], p.bA, p.sb)
    cy = dynamics.get_cy(alpha, beta, x[8], p.lef, x[4], x[15], p.bA, p.sb)
    cz = dynamics.get_cz(alpha, beta, 0.0, 0.0, p.lef, x[2], x[3], x[15], p.l)
    forces_ned_body = np.array([thrust - qS * cx, qS * cz, -qS * cy])
    gravity = p.g * np.array(
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    )
    omega_ned_body = np.array([x[2], x[4], -x[3]])
    expected = forces_ned_body / p.m + gravity - np.cross(omega_ned_body, v)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    "speed,height", [(100.0, 1000.0), (120.0, 3000.0), (160.0, 5000.0)]
)
def test_trim_balances_all_accelerations_with_physical_thrust(speed, height):
    solution = find_trim(V_target=speed, h_target=height)
    assert solution.converged
    assert 0.0 < solution.T_thrust < 130000.0
    np.testing.assert_allclose(solution.residuals, 0.0, atol=1e-6)


def test_trim_does_not_claim_convergence_outside_engine_capability():
    solution = find_trim(params=F16AngularParameters(T_max_thrust=1.0))
    assert not solution.converged
