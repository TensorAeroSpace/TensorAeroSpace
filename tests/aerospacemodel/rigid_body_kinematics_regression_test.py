"""Check Euler-rate inversion independently of each aircraft's aerodynamics."""

import importlib

import numpy as np
import pytest

CASES = [
    ("b747", "b747_ode_6dof", 12, 4),
    ("b737", "b737_ode_6dof", 12, 4),
    ("x15", "x15_ode_6dof", 13, 4),
    ("skywalker_x8", "x8_ode_6dof", 12, 3),
    ("aai_shadow", "shadow_ode_6dof", 12, 4),
    ("quadrotor", "quadrotor_ode_6dof", 12, 4),
]


def evaluate(case, pitch):
    package, rhs_name, n_states, n_actions = case
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{package}.nonlinear.dynamics"
    )
    params_module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{package}.nonlinear.params"
    )
    state = np.zeros(n_states)
    omega = np.array([0.1, -0.2, 0.3])
    state[6:9] = [0.4, pitch, -0.2]
    if package == "quadrotor":
        state[3] = 1.0
        state[9:12] = omega
    else:
        state[0], state[11] = 100.0, -1000.0
        state[3:6] = omega
    if n_states == 13:
        state[12] = 1000.0
    derivative = getattr(module, rhs_name)(
        state, np.zeros(n_actions), 0.0, params_module.default_parameters()
    )
    return derivative[6:9], omega


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("pitch_deg", [-120.0, -30.0, 30.0, 120.0])
def test_euler_rates_reconstruct_body_rates_on_both_sides_of_vertical(case, pitch_deg):
    theta = np.deg2rad(pitch_deg)
    rates, omega = evaluate(case, theta)
    phi = 0.4
    # Resolve the Euler rotation axes into body coordinates (inverse mapping).
    axes = np.array(
        [
            [1.0, 0.0, -np.sin(theta)],
            [0.0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0.0, -np.sin(phi), np.cos(phi) * np.cos(theta)],
        ]
    )
    np.testing.assert_allclose(axes @ rates, omega, atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("pitch", [-np.pi / 2, np.pi / 2])
def test_singularity_guard_bounds_all_euler_rates_consistently(case, pitch):
    rates, omega = evaluate(case, pitch)
    assert np.isfinite(rates).all()
    assert np.max(np.abs(rates)) <= np.linalg.norm(omega) / 1e-9 + 1.0
