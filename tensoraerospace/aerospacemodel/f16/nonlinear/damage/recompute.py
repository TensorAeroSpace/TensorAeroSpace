"""Recompute mass, S, b, bA, and CG from DamageState."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import BaseGeometry
from .state import DamageState


def recompute_mass_geometry(geo: BaseGeometry, state: DamageState) -> Dict:
    """Aggregate per-section damaged contributions into bulk parameters.

    Returns a dict with keys: m, S, b, bA, cg (np.ndarray shape (3,)).
    """
    sections = geo.sections
    losses = state.section_loss

    # Effective masses
    m_eff_per_section = np.array([
        s.mass * (1.0 - losses.get(s.name, 0.0)) for s in sections
    ], dtype=np.float64)
    m_eff = float(m_eff_per_section.sum() + state.structural.extra_mass_delta_kg)
    if m_eff <= 0.0:
        raise ValueError(
            f"Effective mass {m_eff} non-positive; check damage state"
        )

    # CG: mass-weighted average of remaining masses, plus structural shift
    section_mass_total = float(m_eff_per_section.sum())
    if section_mass_total <= 0.0:
        raise ValueError(
            f"Section mass total {section_mass_total} non-positive"
        )
    cg_x = float(sum(
        m * s.cg_local[0] for m, s in zip(m_eff_per_section, sections)
    ) / section_mass_total)
    cg_y = float(sum(
        m * s.cg_local[1] for m, s in zip(m_eff_per_section, sections)
    ) / section_mass_total)
    cg_z = float(sum(
        m * s.cg_local[2] for m, s in zip(m_eff_per_section, sections)
    ) / section_mass_total)
    cg = np.array([
        cg_x + state.structural.extra_cg_shift_m[0],
        cg_y + state.structural.extra_cg_shift_m[1],
        cg_z + state.structural.extra_cg_shift_m[2],
    ])

    # Effective wing area (only type=="wing" contributes)
    s_eff = float(sum(
        s.area * (1.0 - losses.get(s.name, 0.0))
        for s in sections if s.type == "wing"
    ))

    # Effective span: outermost surviving point on each side (handles asymmetric loss)
    def half_span(side: str) -> float:
        ys = [
            abs(s.span_position) * (0.0 if losses.get(s.name, 0.0) >= 1.0 else 1.0)
            for s in sections
            if s.type == "wing" and s.side == side
        ]
        return max(ys) if ys else 0.0

    b_eff = half_span("left") + half_span("right")

    # MAC: area-weighted chord over surviving wing area
    chord_num = sum(
        s.chord * s.area * (1.0 - losses.get(s.name, 0.0))
        for s in sections if s.type == "wing"
    )
    bA_eff = float(chord_num / s_eff) if s_eff > 0.0 else 0.0

    return {"m": m_eff, "S": s_eff, "b": b_eff, "bA": bA_eff, "cg": cg}


def recompute_inertia(
    geo: BaseGeometry, state: DamageState, cg: np.ndarray
) -> Dict[str, float]:
    """Compute aircraft inertias about `cg` using Huygens–Steiner.

    Returns the diagonal moments (Jx, Jy, Jz) and the cross-product Jxy.

    Note on convention: this F-16 model uses Jxy (x-y plane product of
    inertia) as its active off-diagonal coupling — see params.py and
    dynamics.py. The fourth element of `AeroSection.inertia_local` is
    interpreted as Jxy = ∫x·y dm (positive integrand, parallel-axis
    offset is +m·rx·ry).
    """
    cg = np.asarray(cg, dtype=np.float64).reshape(3)
    Jx = Jy = Jz = Jxy = 0.0
    for s in geo.sections:
        f = state.section_loss.get(s.name, 0.0)
        m_eff = s.mass * (1.0 - f)
        if m_eff <= 0.0:
            continue
        Ixx_l, Iyy_l, Izz_l, Ixy_l = s.inertia_local
        # Scale local inertia by remaining mass fraction (uniform-density assumption)
        scale = (1.0 - f)
        Ixx_l *= scale
        Iyy_l *= scale
        Izz_l *= scale
        Ixy_l *= scale
        rx = s.cg_local[0] - cg[0]
        ry = s.cg_local[1] - cg[1]
        rz = s.cg_local[2] - cg[2]
        # Parallel-axis: I_about_cg = I_local + m * (r⊥)²
        Jx += Ixx_l + m_eff * (ry**2 + rz**2)
        Jy += Iyy_l + m_eff * (rx**2 + rz**2)
        Jz += Izz_l + m_eff * (rx**2 + ry**2)
        Jxy += Ixy_l + m_eff * rx * ry  # product of inertia: +m*rx*ry
    # Apply structural extras
    Jx += state.structural.extra_inertia_delta[0]
    Jy += state.structural.extra_inertia_delta[1]
    Jz += state.structural.extra_inertia_delta[2]
    Jxy += state.structural.extra_inertia_delta[3]
    return {"Jx": float(Jx), "Jy": float(Jy), "Jz": float(Jz), "Jxy": float(Jxy)}
