"""OnboardCEModel tests — protocol contract, linear CE, F-16 adapter."""

import numpy as np
import pytest

from tensoraerospace.agent.aidi.onboard_ce import (
    F16NonlinearOnboardCE,
    LinearOnboardCE,
)


def test_linear_onboard_ce_returns_constant_matrix():
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ce = LinearOnboardCE(B)
    G = ce(np.zeros(4), np.zeros(2))
    np.testing.assert_array_equal(G, B)
    assert ce.n_state == 3 and ce.n_control == 2


def test_linear_onboard_ce_independent_of_inputs():
    B = np.array([[1.0, 2.0]])
    ce = LinearOnboardCE(B)
    np.testing.assert_array_equal(ce(np.zeros(5), np.zeros(7)), B)


def test_f16_onboard_ce_reproduces_finite_difference_over_deflection():
    """G is ∂ω̇/∂(deflection) — perturb state[8/10/12], not the control vector."""
    f16 = pytest.importorskip(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics",
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )

    params = default_parameters()

    x = np.zeros(14)
    x[0] = 0.05  # alpha
    u = np.zeros(3)

    adapter = F16NonlinearOnboardCE(params=params, perturb=1e-3)
    G = adapter(x, u)

    eps_fine = 5e-4
    G_ref = np.zeros((3, 3))
    rate_idx = [2, 4, 3]  # (p, q, r) in this codebase's wx/wy/wz layout.
    deflection_idx = [8, 10, 12]
    for j_local, j_state in enumerate(deflection_idx):
        x_plus = x.copy()
        x_plus[j_state] += eps_fine
        x_minus = x.copy()
        x_minus[j_state] -= eps_fine
        f_plus = f16.f16_ode_6dof(x_plus, u, 0.0, params)[rate_idx]
        f_minus = f16.f16_ode_6dof(x_minus, u, 0.0, params)[rate_idx]
        G_ref[:, j_local] = (f_plus - f_minus) / (2 * eps_fine)

    # G must NOT be all-zero — the deflection-to-rate gain is real.
    assert np.linalg.norm(G) > 1e-3
    np.testing.assert_allclose(G, G_ref, atol=5e-3, rtol=5e-3)


def test_f16_onboard_ce_is_deterministic():
    pytest.importorskip(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics",
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )

    adapter = F16NonlinearOnboardCE(params=default_parameters(), perturb=1e-3)
    x = np.zeros(14)
    x[0] = 0.05
    u = np.zeros(3)
    G1 = adapter(x, u)
    G2 = adapter(x, u)
    np.testing.assert_array_equal(G1, G2)
