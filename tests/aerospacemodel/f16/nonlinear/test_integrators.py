import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear._integrators import euler, rk4


def _linear_rhs(x, u, t, params):
    """dx/dt = -x + u; analytic solution x(t) = u + (x0 - u) * exp(-t)."""
    return -x + u


def test_euler_one_step_linear():
    x0 = np.array([1.0, 2.0])
    u = np.array([0.0, 0.0])
    out = euler(_linear_rhs, x0, u, t=0.0, dt=0.1, params=None)
    np.testing.assert_allclose(out, x0 + 0.1 * (-x0 + u))


def test_rk4_matches_analytic_solution_better_than_euler():
    x0 = np.array([1.0])
    u = np.array([0.0])
    dt = 0.05
    n = 100
    x_euler = x0.copy()
    x_rk4 = x0.copy()
    for k in range(n):
        x_euler = euler(_linear_rhs, x_euler, u, t=k * dt, dt=dt, params=None)
        x_rk4 = rk4(_linear_rhs, x_rk4, u, t=k * dt, dt=dt, params=None)
    analytic = np.array([np.exp(-n * dt)])
    err_euler = abs(x_euler - analytic).item()
    err_rk4 = abs(x_rk4 - analytic).item()
    assert err_rk4 < err_euler
    assert err_rk4 < 1e-6


def test_rhs_signature_called_with_all_args():
    seen = {}

    def rhs(x, u, t, params):
        seen.update({"x": x, "u": u, "t": t, "params": params})
        return np.zeros_like(x)

    x0 = np.array([1.0])
    u = np.array([0.5])
    rk4(rhs, x0, u, t=0.7, dt=0.01, params={"k": 1})
    assert seen["t"] >= 0.7
    assert seen["params"] == {"k": 1}
    np.testing.assert_array_equal(seen["u"], u)
