import math
import pathlib

import numpy as np
import pytest
from scipy.optimize import fsolve

from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import LongitudinalF16
from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal.params import (
    default_parameters,
)

SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "long_open_loop_1s.npz"


def test_trim_point_exists_and_balances():
    """For some (alpha, stab) the longitudinal forces and moments balance."""
    p = default_parameters()

    def residual(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        u = np.array([stab])  # commanded == current => actuator at rest
        dx = f16_ode_long(x, u, 0.0, p)
        return np.array([dx[0], dx[1]])  # dalpha, dwz

    sol, info, ier, _msg = fsolve(
        residual, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True
    )
    assert ier == 1, f"fsolve did not converge: {_msg}"
    alpha_trim, stab_trim = sol
    x_trim = np.array([alpha_trim, 0.0, stab_trim, 0.0])
    u_trim = np.array([stab_trim])
    dx = f16_ode_long(x_trim, u_trim, 0.0, p)
    np.testing.assert_allclose(dx[:2], 0.0, atol=1e-6)
    assert math.radians(-15) < alpha_trim < math.radians(25)


def test_open_loop_trajectory_matches_snapshot():
    """Regression test: a fixed open-loop episode reproduces an exact trajectory."""
    initial = np.array([[0.0], [0.0], [0.0], [0.0]])
    m = LongitudinalF16(initial, dt=0.01)
    states = []
    for k in range(100):  # 1 second
        u = math.radians(0.5) if k < 50 else math.radians(-0.5)
        out = m.run_step([[u]])
        states.append(np.asarray(out).reshape(-1))
    traj = np.stack(states, axis=0)

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SNAPSHOT, trajectory=traj)
        pytest.skip(f"snapshot created at {SNAPSHOT}; re-run to compare")

    expected = np.load(SNAPSHOT)["trajectory"]
    np.testing.assert_allclose(traj, expected, atol=1e-12)


def test_actuator_position_limit_enforced_over_long_command():
    """Sustained commanded value above maxabsstab keeps stab at the limit."""
    p = default_parameters()
    initial = np.array([[0.0], [0.0], [0.0], [0.0]])
    m = LongitudinalF16(initial, dt=0.005)
    huge = math.radians(40.0)  # well above maxabsstab=25 deg
    for _ in range(500):
        m.run_step([[huge]])
    final = np.asarray(m.x_history[-1]).reshape(-1)
    assert abs(final[2]) <= p.maxabsstab + 1e-6
