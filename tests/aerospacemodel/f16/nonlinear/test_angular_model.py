import math
import sys

import numpy as np
import pytest


def test_import_does_not_load_matlab():
    for mod in list(sys.modules):
        if mod.startswith("matlab"):
            sys.modules.pop(mod, None)
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (  # noqa
        AngularF16,
        initial_state,
        set_initial_state,
    )

    assert "matlab" not in sys.modules
    assert "matlab.engine" not in sys.modules


def test_run_step_returns_14d_state():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    m = AngularF16(initial_state)
    out = m.run_step([[0.0], [0.0], [0.0]])
    assert np.asarray(out).reshape(-1).shape == (14,)


def test_run_step_accepts_flat_list():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    m = AngularF16(initial_state)
    out = m.run_step([0.0, 0.0, 0.0])
    assert out is not None


def test_run_step_rejects_wrong_action_dim():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    m = AngularF16(initial_state)
    with pytest.raises(ValueError):
        m.run_step([[0.0]])


def test_unknown_integrator_raises():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    with pytest.raises(ValueError):
        AngularF16(initial_state, integrator="midpoint")


def test_integrator_choice_runs_both_modes_and_diverges():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    m_e = AngularF16(initial_state, integrator="euler")
    m_r = AngularF16(initial_state, integrator="rk4")
    for _ in range(50):
        m_e.run_step([[math.radians(1.0)], [0.0], [0.0]])
        m_r.run_step([[math.radians(1.0)], [0.0], [0.0]])
    final_e = m_e.current_state
    final_r = m_r.current_state
    assert np.all(np.isfinite(final_e))
    assert np.all(np.isfinite(final_r))
    assert not np.allclose(
        final_e, final_r, atol=1e-9
    ), "euler and rk4 produced identical trajectories — integrator switch broken"


def test_state_vector_ordering_matches_list_state():
    """The list_state ordering must match the matlab F16State_vec2struct
    convention: stab/dstab/ail/dail/dir/ddir interleaved (not grouped)."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        AngularF16,
        initial_state,
    )

    m = AngularF16(initial_state)
    assert m.list_state.index("dstab") == 9
    assert m.list_state.index("ail") == 10
    assert m.list_state.index("dail") == 11
    assert m.list_state.index("dir") == 12
    assert m.list_state.index("ddir") == 13


def test_set_initial_state_overrides_alpha_correctly():
    """Regression: the LEGACY inital.py grouped stab/ail/dir before
    derivatives, producing a permuted vector. The new pure version must
    put alpha at index 0 and dstab at index 9."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        set_initial_state,
    )

    out = set_initial_state({"alpha": math.radians(7.0), "dstab": math.radians(2.0)})
    arr = np.asarray(out, dtype=float).reshape(-1)
    assert arr[0] == pytest.approx(math.radians(7.0))
    assert arr[9] == pytest.approx(math.radians(2.0))
    # Other indices must remain at zero
    for i in range(14):
        if i not in (0, 9):
            assert arr[i] == 0.0


def test_set_initial_state_does_not_accumulate_overrides():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        set_initial_state,
    )

    a = set_initial_state({"alpha": math.radians(5.0)})
    b = set_initial_state({"beta": math.radians(3.0)})
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    assert a_arr[0] == pytest.approx(math.radians(5.0))
    assert a_arr[1] == 0.0
    assert b_arr[0] == 0.0
    assert b_arr[1] == pytest.approx(math.radians(3.0))


def test_set_initial_state_rejects_unknown_key():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import (
        set_initial_state,
    )

    with pytest.raises(ValueError):
        set_initial_state({"not_a_state": 1.0})
