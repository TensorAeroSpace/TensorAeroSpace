"""Strip-theory aerodynamic corrections from DamageState.

All deltas are dimensionless (normalized so they add directly to the base
F-16 coefficients Cy/Cx/Cz/Mx/My/Mz). The base S used for normalisation is
the BaseGeometry's total wing area.

Per-section contributions follow the convention established in Phase 1:
  - cl_alpha_contribution is the section's CL_α (1/rad) — its share of the
    aircraft-level lift-curve slope (aggregate Σ cl_α·area/S_total ≈ 4.5)
  - cd0_contribution is the section's additive contribution to Cx0 (sums
    directly to the aircraft-level Cx0)

The deltas are expressed in the intact-aircraft coefficient frame. The ODE
therefore scales them with the intact reference geometry even when
``DamageState`` has reduced effective wing area/span. This avoids counting a
lost wing section once through smaller ``S`` and again through these deltas.
"""

from __future__ import annotations

from .geometry import AeroSection, BaseGeometry
from .state import DamageState

_LOST_PARASITE_DRAG_FRACTION = 0.25
_EXPOSED_EDGE_DRAG_COEF = 0.04


def _base_wing_area(geo: BaseGeometry) -> float:
    return geo.total_wing_area()


def _local_drag_delta(
    section: AeroSection, loss_fraction: float, S_base: float
) -> float:
    f = float(loss_fraction)
    if f <= 0.0:
        return 0.0
    delta = -_LOST_PARASITE_DRAG_FRACTION * section.cd0_contribution * f
    if section.type == "wing":
        exposed_shape = f * (2.0 - f)
        delta += _EXPOSED_EDGE_DRAG_COEF * exposed_shape * (section.area / S_base)
    return float(delta)


def delta_cy(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Lost normal-force contribution: ΔCy = -Σ cl_α_s · α · f_s · (area_s/S_base)."""
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    return float(
        -sum(
            s.cl_alpha_contribution
            * alpha
            * state.section_loss.get(s.name, 0.0)
            * (s.area / S_base)
            for s in geo.sections
            if s.type == "wing"
        )
    )


def delta_cx(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Drag delta from damaged or missing sections.

    A missing section removes some parasite drag from its wetted area, but an
    abrupt wing-loss event also creates exposed structure and separated flow.
    The exposed-edge term is intentionally nonzero for both partial and full
    losses; otherwise a fully torn-off wing tip could reduce total drag, which
    is too optimistic for the failure-control demos.
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    delta = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        delta += _local_drag_delta(s, f, S_base)
    return float(delta)


def delta_cz(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Side-force delta: dominated by vtail loss (proportional to β)."""
    delta = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        if s.type == "vtail":
            # Treat vtail's cl_alpha-equivalent for sideslip with a small
            # constant. F-16 vertical tail provides ~0.4 /rad effective
            # ∂Cz/∂β at the aircraft level.
            VTAIL_BETA_GAIN = 0.40  # 1/rad
            delta -= VTAIL_BETA_GAIN * beta * f
    return float(delta)


def _max_half_span(geo: BaseGeometry) -> float:
    """Largest absolute span_position over wing sections (used for normalisation)."""
    return max(
        (abs(s.span_position) for s in geo.sections if s.type == "wing"),
        default=1.0,
    )


def delta_mx(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Roll-moment coefficient delta from asymmetric lift loss.

    ΔMx (dimensionless) = -Σ cl_α_s · α · f_s · (area_s/S_base) · (y_arm_s/b_base)
    where b_base = 2 · max half-span (for normalisation: cmx = Mx/(q·S·l)).
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    b_base = 2.0 * _max_half_span(geo)
    return float(
        -sum(
            s.cl_alpha_contribution
            * alpha
            * state.section_loss.get(s.name, 0.0)
            * (s.area / S_base)
            * (s.span_position / b_base)
            for s in geo.sections
            if s.type == "wing"
        )
    )


def delta_mz(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Yaw-moment coefficient delta from asymmetric drag.

    Uses local ΔCx contribution on each section (lost cd0 + jagged-edge drag),
    multiplied by its y-arm.
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    b_base = 2.0 * _max_half_span(geo)
    out = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        local_dcx = _local_drag_delta(s, f, S_base)
        out += local_dcx * (s.span_position / b_base)
    return float(out)


def delta_my(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Pitch-moment coefficient delta from lost lift × x-arm.

    Normalised by S_base × bA_base (area-weighted MAC).
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    # bA_base for normalisation: area-weighted chord over wing sections
    wing_area_chord_sum = sum(
        s.chord * s.area for s in geo.sections if s.type == "wing"
    )
    bA_base = wing_area_chord_sum / S_base if S_base > 0 else 1.0
    if bA_base == 0.0:
        return 0.0
    return float(
        -sum(
            s.cl_alpha_contribution
            * alpha
            * state.section_loss.get(s.name, 0.0)
            * (s.area / S_base)
            * (s.aero_x_arm / bA_base)
            for s in geo.sections
            if s.type in ("wing", "stab")
        )
    )
