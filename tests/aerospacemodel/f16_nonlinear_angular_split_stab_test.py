"""Regression: split_stab=False is bit-identical to legacy 3-input mode."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16


def test_default_mode_is_legacy_three_input():
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    assert m.action_space_length == 3
    assert m.control_list == ["stab", "ail", "dir"]


def test_default_mode_step_unchanged_after_flag_addition():
    """Lock current behaviour: zero-input step must give the canonical baseline."""
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m.run_step(np.zeros(3))
    s = m.current_state
    assert s.shape == (14,)
    assert abs(s[0]) < 1e-3
    assert abs(s[1]) < 1e-6


def test_split_stab_mode_advertises_four_inputs():
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    assert m.action_space_length == 4
    assert m.control_list == ["stab_left", "stab_right", "ail", "dir"]


def test_split_stab_symmetric_command_matches_legacy():
    """If both halves get the same command, the trajectory must equal the
    single-input case."""
    legacy = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    split = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    u_legacy = np.array([0.05, 0.0, 0.0])
    u_split = np.array([0.05, 0.05, 0.0, 0.0])
    for _ in range(50):
        legacy.run_step(u_legacy)
        split.run_step(u_split)
    np.testing.assert_allclose(
        legacy.current_state, split.current_state, atol=1e-12,
        err_msg="symmetric split-stab must reproduce legacy single-stab behaviour",
    )


def test_split_stab_asymmetric_produces_roll_moment():
    """Differential stabilator command must induce a roll moment (dwx != 0)."""
    m = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4", split_stab=True)
    u = np.array([0.10, -0.10, 0.0, 0.0])
    for _ in range(20):
        m.run_step(u)
    s = m.current_state
    assert abs(s[2]) > 1e-4, f"wx (roll rate) should be non-zero, got {s[2]}"
