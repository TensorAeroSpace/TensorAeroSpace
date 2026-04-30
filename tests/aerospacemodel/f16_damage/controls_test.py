"""Control-surface failure mapping."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    ControlFailure,
    DamageState,
)


@pytest.fixture
def healthy():
    return DamageState.healthy(load_f16_geometry())


def test_healthy_passthrough(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )

    u = np.array([0.1, -0.05, 0.02, 0.03])
    surface_to_idx = {
        "stab_left": 0,
        "stab_right": 1,
        "aileron_left": 2,
        "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    np.testing.assert_array_equal(out, u)


def test_jam_overrides_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )

    healthy.set_control_failure(
        "stab_left",
        ControlFailure(mode="jam", jam_position_rad=0.05),
    )
    u = np.array([0.4, 0.0, 0.0, 0.0])
    surface_to_idx = {
        "stab_left": 0,
        "stab_right": 1,
        "aileron_left": 2,
        "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[0] == pytest.approx(0.05)


def test_efficiency_loss_scales_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )

    healthy.set_control_failure(
        "rudder",
        ControlFailure(mode="efficiency_loss", efficiency=0.5),
    )
    u = np.array([0.0, 0.0, 0.0, 0.4])
    surface_to_idx = {
        "stab_left": 0,
        "stab_right": 1,
        "aileron_left": 2,
        "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[3] == pytest.approx(0.2)


def test_lost_zeros_command(healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        apply_control_failures,
    )

    healthy.set_control_failure("rudder", ControlFailure(mode="lost"))
    u = np.array([0.0, 0.0, 0.0, 0.4])
    surface_to_idx = {
        "stab_left": 0,
        "stab_right": 1,
        "aileron_left": 2,
        "rudder": 3,
    }
    out = apply_control_failures(u, healthy, surface_to_idx)
    assert out[3] == 0.0
