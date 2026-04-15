import math
import sys

import numpy as np
import pytest


def test_import_does_not_load_matlab():
    """Importing the python.* longitudinal module must not pull matlab in."""
    for mod in list(sys.modules):
        if mod.startswith("matlab"):
            sys.modules.pop(mod, None)
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (  # noqa
        LongitudinalF16,
        initial_state,
        set_initial_state,
    )

    assert "matlab" not in sys.modules
    assert "matlab.engine" not in sys.modules


def test_run_step_returns_4d_state():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m = LongitudinalF16(initial_state)
    out = m.run_step([[0.0]])
    assert isinstance(out, np.ndarray)
    arr = np.asarray(out).reshape(-1)
    assert arr.shape == (4,)


def test_run_step_accepts_numpy_array():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m = LongitudinalF16(initial_state)
    out = m.run_step(np.array([[0.0]]))
    assert out is not None


def test_run_step_rejects_wrong_action_dim():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m = LongitudinalF16(initial_state)
    with pytest.raises(ValueError):
        m.run_step([[0.0], [0.0]])


def test_selected_state_output_subset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m = LongitudinalF16(initial_state, selected_state_output=["alpha", "wz"])
    y = m.run_step([[0.0]])
    assert np.asarray(y).reshape(-1).shape == (2,)


def test_integrator_choice_runs_both_modes():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m_euler = LongitudinalF16(initial_state, integrator="euler")
    m_rk4 = LongitudinalF16(initial_state, integrator="rk4")
    for _ in range(50):
        out_e = m_euler.run_step([[math.radians(1.0)]])
        out_r = m_rk4.run_step([[math.radians(1.0)]])
        assert np.all(np.isfinite(np.asarray(out_e)))
        assert np.all(np.isfinite(np.asarray(out_r)))

    final_e = m_euler.current_state
    final_r = m_rk4.current_state
    # Euler and RK4 should diverge slightly over 50 steps with a non-trivial
    # input. If they're allclose at default tolerance, the integrator switch
    # is doing nothing.
    assert not np.allclose(
        final_e, final_r, atol=1e-9
    ), "euler and rk4 produced identical trajectories — integrator switch broken"


def test_unknown_integrator_raises():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    with pytest.raises(ValueError):
        LongitudinalF16(initial_state, integrator="midpoint")


def test_set_initial_state_returns_array_with_overrides_applied():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        set_initial_state,
    )

    out = set_initial_state({"alpha": math.radians(10.0)})
    arr = np.asarray(out, dtype=float).reshape(-1)
    assert arr[0] == pytest.approx(math.radians(10.0))


def test_set_initial_state_rejects_unknown_key():
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        set_initial_state,
    )

    with pytest.raises(Exception):
        set_initial_state({"not_a_state": 1.0})


def test_set_initial_state_does_not_accumulate_overrides_across_calls():
    """Regression: set_initial_state must be pure. Two independent calls
    should NOT see each other's overrides."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        set_initial_state,
    )

    a = set_initial_state({"alpha": math.radians(7.0)})
    b = set_initial_state({"wz": math.radians(3.0)})
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    # `a` should have only alpha=7°, not wz=3°
    assert a_arr[0] == pytest.approx(math.radians(7.0))
    assert a_arr[1] == 0.0
    # `b` should have only wz=3°, not alpha=7°
    assert b_arr[0] == 0.0
    assert b_arr[1] == pytest.approx(math.radians(3.0))


def test_list_state_survives_construction():
    """Regression: ModelBase._initialize_selected_state_index has a side
    effect of wiping self.list_state. The class must work around this so
    consumers can do `m.list_state.index('alpha')` after construction."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
        initial_state,
    )

    m = LongitudinalF16(initial_state)
    assert m.list_state == ["alpha", "wz", "stab", "dstab"]
    assert m.control_list == ["stab"]
    assert m.list_state.index("dstab") == 3


def test_top_level_python_subpackage_reexports():
    """Importing from the top-level python.* path must yield the same classes."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.python import (
        AngularF16 as AngularF16_top,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.python import (
        LongitudinalF16 as LongitudinalF16_top,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.angular import AngularF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import (
        LongitudinalF16,
    )

    assert LongitudinalF16_top is LongitudinalF16
    assert AngularF16_top is AngularF16
