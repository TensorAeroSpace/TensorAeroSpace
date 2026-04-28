"""Strip-theory aerodynamic corrections from DamageState.

All deltas are dimensionless (normalized so they add directly to the base
F-16 coefficients Cy/Cx/Cz/Mx/My/Mz). The base S used for normalisation is
the BaseGeometry's total wing area.

Per-section contributions follow the convention established in Phase 1:
  - cl_alpha_contribution is the section's CL_α (1/rad) — its share of the
    aircraft-level lift-curve slope (aggregate Σ cl_α·area/S_total ≈ 4.5)
  - cd0_contribution is the section's additive contribution to Cx0 (sums
    directly to the aircraft-level Cx0)
"""

from __future__ import annotations

from .geometry import BaseGeometry
from .state import DamageState


_JAGGED_DRAG_COEF = 0.05  # peaks at f=0.5; calibrated heuristically


def _base_wing_area(geo: BaseGeometry) -> float:
    return geo.total_wing_area()


def delta_cy(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Lost normal-force contribution: ΔCy = -Σ cl_α_s · α · f_s · (area_s/S_base)."""
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    return float(-sum(
        s.cl_alpha_contribution * alpha * state.section_loss.get(s.name, 0.0)
        * (s.area / S_base)
        for s in geo.sections if s.type == "wing"
    ))


def delta_cx(alpha: float, beta: float, geo: BaseGeometry, state: DamageState) -> float:
    """Drag delta: lost cd0 contribution + jagged-edge drag from partial damage.

    A fully-lost wing section removes its cd0 contribution from the aircraft
    drag. A partially-damaged section also adds extra "jagged-edge" drag that
    peaks at f=0.5 (full-cut surface gives maximum exposed cross-section).
    """
    S_base = _base_wing_area(geo)
    if S_base <= 0.0:
        return 0.0
    delta = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        if f <= 0.0:
            continue
        delta -= s.cd0_contribution * f
        if s.type == "wing":
            # Jagged-edge drag: peaks at f=0.5
            delta += _JAGGED_DRAG_COEF * f * (1.0 - f) * (s.area / S_base)
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
