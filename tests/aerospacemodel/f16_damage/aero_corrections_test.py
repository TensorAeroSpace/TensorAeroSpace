"""Strip-theory aero corrections from DamageState."""

from __future__ import annotations

import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


@pytest.fixture
def healthy(geo):
    return DamageState.healthy(geo)


def test_healthy_forces_are_zero(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    beta = 0.0
    assert aero_corrections.delta_cy(alpha, beta, geo, healthy) == 0.0
    assert aero_corrections.delta_cx(alpha, beta, geo, healthy) == 0.0
    assert aero_corrections.delta_cz(alpha, beta, geo, healthy) == 0.0


def test_left_tip_full_loss_reduces_cy(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    dcy = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    # Negative delta: lift contribution lost. Magnitude ~ cl_alpha × α × area/S_base.
    assert dcy < 0
    s = geo.section("left_tip")
    expected = -s.cl_alpha_contribution * alpha * (s.area / geo.total_wing_area())
    assert dcy == pytest.approx(expected, rel=0.05)


def test_partial_loss_scales_linearly(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    full = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    healthy.set_section_loss("left_tip", 0.5)
    half = aero_corrections.delta_cy(alpha, 0.0, geo, healthy)
    assert half == pytest.approx(full * 0.5, rel=0.001)


def test_partial_loss_adds_drag(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    healthy.set_section_loss("left_tip", 1.0)  # totally lost: no jagged-edge drag
    dcx_full = aero_corrections.delta_cx(0.0, 0.0, geo, healthy)
    healthy.set_section_loss("left_tip", 0.5)  # half lost: max jagged-edge drag
    dcx_half = aero_corrections.delta_cx(0.0, 0.0, geo, healthy)
    # Half-loss should give NET higher drag than full loss (full removes both
    # baseline cd0 and jagged-edge contributions; half loses less cd0 but adds
    # the maximum jagged-edge drag).
    assert dcx_half > dcx_full


def test_left_tip_loss_creates_positive_roll_moment(geo, healthy):
    """Loss of left wing tip → less lift on left → roll moment magnitude > 0
    (sign convention is matlab-port specific; here ΔMx must be nonzero)."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 1.0)
    dmx = aero_corrections.delta_mx(alpha, 0.0, geo, healthy)
    # Lost left-side lift on negative-y → roll moment magnitude > 0
    assert abs(dmx) > 1e-3


def test_symmetric_loss_no_roll(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("left_tip", 0.5)
    healthy.set_section_loss("right_tip", 0.5)
    dmx = aero_corrections.delta_mx(alpha, 0.0, geo, healthy)
    assert abs(dmx) < 1e-9


def test_yaw_moment_from_asymmetric_drag(geo, healthy):
    """Half-loss on left tip → jagged-edge drag → yaw moment."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    healthy.set_section_loss("left_tip", 0.5)
    dmz = aero_corrections.delta_mz(0.0, 0.0, geo, healthy)
    assert abs(dmz) > 1e-6


def test_pitch_moment_from_lost_stab(geo, healthy):
    """Loss of horizontal stab → pitch moment delta from lost lift on tail."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        aero_corrections,
    )
    alpha = math.radians(5.0)
    healthy.set_section_loss("stab_left", 1.0)
    dmy = aero_corrections.delta_my(alpha, 0.0, geo, healthy)
    # Tail is aft (negative aero_x_arm), losing it removes a downward
    # contribution → magnitude > 0
    assert abs(dmy) > 1e-4
