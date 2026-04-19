import math
import pathlib

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
    AngularF16,
    initial_state,
)

SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "angular_open_loop_1s.npz"


def test_zero_input_zero_state_no_nans_for_one_second():
    """1 s of zero-command simulation produces finite states."""
    m = AngularF16(initial_state, dt=0.01)
    for _ in range(100):
        out = m.run_step([[0.0], [0.0], [0.0]])
        arr = np.asarray(out).reshape(-1)
        assert np.all(np.isfinite(arr))


def test_open_loop_trajectory_snapshot():
    """Regression: a fixed sequence of stab/ail/dir commands produces an
    exact trajectory."""
    m = AngularF16(initial_state, dt=0.01)
    states = []
    for k in range(100):
        u = [
            [math.radians(0.5) if k < 30 else 0.0],
            [math.radians(1.0) if 30 <= k < 60 else 0.0],
            [math.radians(0.5) if 60 <= k else 0.0],
        ]
        out = m.run_step(u)
        states.append(np.asarray(out).reshape(-1))
    traj = np.stack(states, axis=0)

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SNAPSHOT, trajectory=traj)
        pytest.skip(f"snapshot created at {SNAPSHOT}; re-run to compare")

    expected = np.load(SNAPSHOT)["trajectory"]
    np.testing.assert_allclose(traj, expected, atol=1e-9)


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
