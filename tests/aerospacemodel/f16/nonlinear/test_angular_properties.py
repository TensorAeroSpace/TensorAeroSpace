import math

import numpy as np
from scipy.integrate import solve_ivp

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
    AngularF16,
    initial_state,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import f16_ode_6dof


def test_zero_input_zero_state_no_nans_for_one_second():
    """1 s of zero-command simulation produces finite states."""
    m = AngularF16(initial_state, dt=0.01)
    for _ in range(100):
        out = m.run_step([[0.0], [0.0], [0.0]])
        arr = np.asarray(out).reshape(-1)
        assert np.all(np.isfinite(arr))


def test_open_loop_trajectory_matches_independent_continuous_integration():
    """Check the fixed-step integrator against DOP853 across command changes."""
    model = AngularF16(initial_state, dt=0.0025, integrator="rk4")
    expected_state = np.asarray(initial_state, dtype=float).reshape(-1)
    actual = []
    expected = []
    for start, end, command in [
        (0.0, 0.3, [math.radians(0.5), 0.0, 0.0]),
        (0.3, 0.6, [0.0, math.radians(1.0), 0.0]),
        (0.6, 1.0, [0.0, 0.0, math.radians(0.5)]),
    ]:
        count = round((end - start) / model.dt)
        times = np.linspace(start, end, count + 1)[1:]
        solution = solve_ivp(
            lambda t, x: f16_ode_6dof(x, np.array(command), t, model.param),
            (start, end),
            expected_state,
            t_eval=times,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        assert solution.success
        expected.extend(solution.y.T)
        expected_state = solution.y[:, -1]
        for _ in range(count):
            actual.append(model.run_step(command).reshape(-1))
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)


def test_actuator_position_limits_enforced_3channels():
    """Sustained huge commands keep all three actuator positions clamped."""
    p = AngularF16(initial_state).get_param()
    m = AngularF16(initial_state, dt=0.005)
    huge = [[math.radians(40.0)], [math.radians(40.0)], [math.radians(40.0)]]
    for _ in range(500):
        m.run_step(huge)
    final = m.current_state
    assert abs(final[8]) <= p.maxabsstab + 1e-6  # stab at index 8
    assert abs(final[10]) <= p.maxabsail + 1e-6  # ail at index 10
    assert abs(final[12]) <= p.maxabsdir + 1e-6  # dir at index 12
