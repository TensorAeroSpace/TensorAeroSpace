"""B-747 non-dimensional stability and control derivatives at the 10
flight conditions of NASA CR-2144 §IX.

Sources within CR-2144:

* **FC1, FC2** — Tables IX-1, IX-2 (landing & power approach explicit values).
* **FC3..FC10** — Tables IX-1, IX-2 *base* values + the C_*  vs Mach
  curves from Figures IX-3..IX-12 (digitised from the graphs at the
  flight-condition markers ③..⑩). Stability axis where indicated.

All coefficients are in **per radian** unless noted. Sign conventions
follow Stevens-Lewis Appendix C / NASA CR-2144 Appendix A:

* C_L, C_D — lift / drag (stability axis), positive up / aft.
* C_m — pitching moment, positive nose up.
* C_Y, C_l, C_n — side force / rolling / yawing (body or stability
  axis as marked), positive: side force right, roll right, yaw right.

The numbers below are the curated digitisation; if you need a higher
resolution lookup, run a fresh interpolation from the CR-2144 graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LongitudinalDerivatives:
    """Trimmed longitudinal non-dimensional stability and control derivatives.

    All values are per radian (where applicable). Stability axis.
    """

    C_L0: float          # lift coefficient at trim
    C_D0: float          # drag coefficient at trim
    C_La: float          # ∂C_L/∂α
    C_Da: float          # ∂C_D/∂α
    C_ma: float          # ∂C_m/∂α (longitudinal static stability)
    C_Lq: float          # ∂C_L/∂(q·c̄/2V)
    C_mq: float          # pitch damping
    C_Ladot: float       # α̇ effect on lift  (typically 0 for jets in CR-2144)
    C_madot: float       # α̇ effect on pitching moment
    C_LM: float          # Mach effect on lift
    C_DM: float          # Mach effect on drag
    C_mM: float          # Mach effect on pitching moment
    C_Lde: float         # elevator effectiveness on lift
    C_mde: float         # elevator effectiveness on pitching moment


@dataclass(frozen=True)
class LateralDirectionalDerivatives:
    """Trimmed lateral-directional non-dimensional derivatives.

    All values are per radian. Stability axis.
    """

    C_Yb: float          # ∂C_Y/∂β
    C_lb: float          # ∂C_l/∂β  (dihedral effect)
    C_nb: float          # ∂C_n/∂β  (directional stability)
    C_Yp: float          # ∂C_Y/∂(p·b/2V)
    C_lp: float          # roll damping
    C_np: float          # ∂C_n/∂(p·b/2V)
    C_Yr: float          # ∂C_Y/∂(r·b/2V)
    C_lr: float          # ∂C_l/∂(r·b/2V)
    C_nr: float          # yaw damping
    C_Yda: float         # aileron side-force (≈ 0 conventionally)
    C_lda: float         # aileron rolling effectiveness
    C_nda: float         # aileron yaw cross-coupling
    C_Ydr: float         # rudder side-force
    C_ldr: float         # rudder rolling cross-coupling
    C_ndr: float         # rudder yawing effectiveness


# ---- Table IX-1 (Landing) and IX-2 (Power Approach) — explicit ----

_LANDING_LONG = LongitudinalDerivatives(
    C_L0=1.76, C_D0=0.263,
    C_La=5.67, C_Da=1.13, C_ma=-1.45,
    C_Lq=5.65, C_mq=-21.4,
    C_Ladot=-6.7, C_madot=-3.3,
    C_LM=-1.10, C_DM=0.0, C_mM=0.36,
    C_Lde=0.356, C_mde=-1.40,
)
_LANDING_LAT = LateralDirectionalDerivatives(
    C_Yb=-1.08, C_lb=-0.281, C_nb=0.184,
    C_Yp=0.0, C_lp=-0.502, C_np=-0.222,
    C_Yr=0.0, C_lr=0.195, C_nr=-0.36,
    C_Yda=0.0, C_lda=0.0530, C_nda=0.0083,
    C_Ydr=0.179, C_ldr=0.0, C_ndr=-0.112,
)

_PA_LONG = LongitudinalDerivatives(
    C_L0=1.11, C_D0=0.102,
    C_La=5.70, C_Da=0.66, C_ma=-1.26,
    C_Lq=5.4, C_mq=-20.8,
    C_Ladot=-6.7, C_madot=-3.2,
    C_LM=-0.81, C_DM=0.0, C_mM=0.27,
    C_Lde=0.338, C_mde=-1.34,
)
_PA_LAT = LateralDirectionalDerivatives(
    C_Yb=-0.96, C_lb=-0.221, C_nb=0.150,
    C_Yp=0.0, C_lp=-0.45, C_np=-0.121,
    C_Yr=0.0, C_lr=0.101, C_nr=-0.30,
    C_Yda=0.0, C_lda=0.0461, C_nda=0.0064,
    C_Ydr=0.175, C_ldr=0.007, C_ndr=-0.109,
)


# ---- Cruise (FC3..FC10) — digitised from CR-2144 figures IX-3..IX-12 ----
#
# Where the figure clearly shows a constant or near-constant value over
# the Mach range of interest, that single number is used. Where the
# curve has noticeable Mach dependence (e.g. C_LM, C_DM, C_lβ at high
# Mach) the value at the marked FC is read.
#
# C_L0 / C_D0 vary with α and Mach; values picked off Fig. IX-4
# (C_L vs Mach, fixed weight & altitude). C_D from Fig. IX-4 right panel
# (drag polar in lo-bar form).

# Format: derivatives[fc_id] = (LongitudinalDerivatives, LateralDirectionalDerivatives)

_CRUISE_LONG: dict[int, LongitudinalDerivatives] = {
    3: LongitudinalDerivatives(  # SL × M=0.45, α=3.10°
        C_L0=0.42, C_D0=0.027,
        C_La=4.4, C_Da=0.18, C_ma=-1.00,
        C_Lq=6.6, C_mq=-19.0,
        C_Ladot=0.0, C_madot=-3.6,
        C_LM=-0.10, C_DM=0.0, C_mM=0.05,
        C_Lde=0.32, C_mde=-1.10,
    ),
    4: LongitudinalDerivatives(  # SL × M=0.65, α=0°
        C_L0=0.21, C_D0=0.020,
        C_La=4.0, C_Da=0.06, C_ma=-0.62,
        C_Lq=6.6, C_mq=-17.5,
        C_Ladot=0.0, C_madot=-3.4,
        C_LM=-0.05, C_DM=0.0, C_mM=-0.05,
        C_Lde=0.20, C_mde=-0.85,
    ),
    5: LongitudinalDerivatives(  # 20K × M=0.50, α=6.80°
        C_L0=0.62, C_D0=0.045,
        C_La=4.5, C_Da=0.40, C_ma=-1.18,
        C_Lq=5.8, C_mq=-20.5,
        C_Ladot=0.0, C_madot=-4.0,
        C_LM=-0.08, C_DM=0.0, C_mM=0.10,
        C_Lde=0.36, C_mde=-1.50,
    ),
    6: LongitudinalDerivatives(  # 20K × M=0.65, α=2.50°
        C_L0=0.32, C_D0=0.024,
        C_La=4.4, C_Da=0.14, C_ma=-0.85,
        C_Lq=6.0, C_mq=-19.5,
        C_Ladot=0.0, C_madot=-3.7,
        C_LM=0.05, C_DM=0.0, C_mM=-0.05,
        C_Lde=0.33, C_mde=-1.30,
    ),
    7: LongitudinalDerivatives(  # 20K × M=0.80, α=0°
        C_L0=0.18, C_D0=0.020,
        C_La=4.0, C_Da=0.07, C_ma=-0.65,
        C_Lq=5.5, C_mq=-19.0,
        C_Ladot=0.0, C_madot=-4.2,
        C_LM=0.05, C_DM=0.005, C_mM=-0.10,
        C_Lde=0.27, C_mde=-1.10,
    ),
    8: LongitudinalDerivatives(  # 40K × M=0.70, α=7.30°
        C_L0=0.55, C_D0=0.035,
        C_La=4.8, C_Da=1.05, C_ma=-1.18,
        C_Lq=5.6, C_mq=-23.0,
        C_Ladot=0.0, C_madot=-4.3,
        C_LM=0.20, C_DM=0.0, C_mM=0.30,
        C_Lde=0.40, C_mde=-1.60,
    ),
    9: LongitudinalDerivatives(  # 40K × M=0.80, α=4.60°
        C_L0=0.45, C_D0=0.045,
        C_La=4.8, C_Da=0.45, C_ma=-1.05,
        C_Lq=5.6, C_mq=-22.0,
        C_Ladot=0.0, C_madot=-5.5,
        C_LM=0.10, C_DM=0.05, C_mM=0.15,
        C_Lde=0.37, C_mde=-1.45,
    ),
    10: LongitudinalDerivatives(  # 40K × M=0.90, α=2.40°
        C_L0=0.30, C_D0=0.080,
        C_La=5.5, C_Da=0.50, C_ma=-1.60,
        C_Lq=5.2, C_mq=-25.0,
        C_Ladot=0.0, C_madot=-9.0,
        C_LM=-0.55, C_DM=0.25, C_mM=-0.10,
        C_Lde=0.30, C_mde=-1.20,
    ),
}

# Lateral-directional cruise derivatives — stability axis, digitised
# from Figures IX-7..IX-12. Most coefficients are near-flat over Mach
# at fixed altitude; key Mach dependencies preserved.
_CRUISE_LAT: dict[int, LateralDirectionalDerivatives] = {
    3: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.18, C_nb=0.140,
        C_Yp=0.0, C_lp=-0.32, C_np=-0.05,
        C_Yr=0.0, C_lr=0.22, C_nr=-0.27,
        C_Yda=0.0, C_lda=0.013, C_nda=0.003,
        C_Ydr=0.150, C_ldr=0.0, C_ndr=-0.103,
    ),
    4: LateralDirectionalDerivatives(
        C_Yb=-0.84, C_lb=-0.18, C_nb=0.150,
        C_Yp=0.0, C_lp=-0.32, C_np=0.0,
        C_Yr=0.0, C_lr=0.10, C_nr=-0.255,
        C_Yda=0.0, C_lda=0.011, C_nda=0.002,
        C_Ydr=0.094, C_ldr=0.010, C_ndr=-0.075,
    ),
    5: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.18, C_nb=0.150,
        C_Yp=0.0, C_lp=-0.31, C_np=-0.07,
        C_Yr=0.0, C_lr=0.28, C_nr=-0.28,
        C_Yda=0.0, C_lda=0.013, C_nda=0.0015,
        C_Ydr=0.160, C_ldr=0.0, C_ndr=-0.108,
    ),
    6: LateralDirectionalDerivatives(
        C_Yb=-0.84, C_lb=-0.18, C_nb=0.155,
        C_Yp=0.0, C_lp=-0.31, C_np=-0.04,
        C_Yr=0.0, C_lr=0.13, C_nr=-0.275,
        C_Yda=0.0, C_lda=0.012, C_nda=0.002,
        C_Ydr=0.140, C_ldr=0.010, C_ndr=-0.100,
    ),
    7: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.16, C_nb=0.180,
        C_Yp=0.0, C_lp=-0.32, C_np=0.0,
        C_Yr=0.0, C_lr=0.06, C_nr=-0.28,
        C_Yda=0.0, C_lda=0.010, C_nda=0.0,
        C_Ydr=0.105, C_ldr=0.010, C_ndr=-0.105,
    ),
    8: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.18, C_nb=0.190,
        C_Yp=0.0, C_lp=-0.32, C_np=-0.07,
        C_Yr=0.0, C_lr=0.30, C_nr=-0.33,
        C_Yda=0.0, C_lda=0.013, C_nda=0.001,
        C_Ydr=0.155, C_ldr=0.0, C_ndr=-0.115,
    ),
    9: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.30, C_nb=0.205,
        C_Yp=0.0, C_lp=-0.33, C_np=-0.05,
        C_Yr=0.0, C_lr=0.30, C_nr=-0.33,
        C_Yda=0.0, C_lda=0.014, C_nda=0.0,
        C_Ydr=0.135, C_ldr=0.010, C_ndr=-0.135,
    ),
    10: LateralDirectionalDerivatives(
        C_Yb=-0.85, C_lb=-0.05, C_nb=0.200,
        C_Yp=0.0, C_lp=-0.34, C_np=0.025,
        C_Yr=0.0, C_lr=0.20, C_nr=-0.32,
        C_Yda=0.0, C_lda=0.011, C_nda=-0.003,
        C_Ydr=0.060, C_ldr=0.0, C_ndr=-0.110,
    ),
}


def get_longitudinal(fc_id: int) -> LongitudinalDerivatives:
    """Return the longitudinal derivative bank at flight condition ``fc_id``."""
    if fc_id == 1:
        return _LANDING_LONG
    if fc_id == 2:
        return _PA_LONG
    if fc_id in _CRUISE_LONG:
        return _CRUISE_LONG[fc_id]
    raise ValueError(f"fc_id must be in [1, 10], got {fc_id}")


def get_lateral(fc_id: int) -> LateralDirectionalDerivatives:
    """Return the lateral-directional derivative bank at flight condition ``fc_id``."""
    if fc_id == 1:
        return _LANDING_LAT
    if fc_id == 2:
        return _PA_LAT
    if fc_id in _CRUISE_LAT:
        return _CRUISE_LAT[fc_id]
    raise ValueError(f"fc_id must be in [1, 10], got {fc_id}")


# ---- Bilinear interpolation between cruise FCs on (altitude, Mach) ----

# Cruise grid layout for interpolation:
#   altitudes (ft):    [    0,  20000,  40000]
#   Mach by altitude:
#     0     ft -> [0.45, 0.65]                   FCs 3, 4
#    20 000 ft -> [0.50, 0.65, 0.80]             FCs 5, 6, 7
#    40 000 ft -> [0.70, 0.80, 0.90]             FCs 8, 9, 10
_CRUISE_GRID = {
    0.0:     [(0.45, 3), (0.65, 4)],
    20000.0: [(0.50, 5), (0.65, 6), (0.80, 7)],
    40000.0: [(0.70, 8), (0.80, 9), (0.90, 10)],
}


def _interp_along(row: list[tuple[float, int]], mach: float, attr: str) -> float:
    machs = [m for m, _ in row]
    fcs = [fc for _, fc in row]
    m_clamped = float(np.clip(mach, machs[0], machs[-1]))
    # locate adjacent pair
    for i in range(len(machs) - 1):
        if machs[i] <= m_clamped <= machs[i + 1]:
            v0 = getattr(get_longitudinal(fcs[i]), attr, None)
            if v0 is None:
                v0 = getattr(get_lateral(fcs[i]), attr)
                v1 = getattr(get_lateral(fcs[i + 1]), attr)
            else:
                v1 = getattr(get_longitudinal(fcs[i + 1]), attr)
            t = (m_clamped - machs[i]) / (machs[i + 1] - machs[i])
            return float(v0 + t * (v1 - v0))
    return getattr(get_longitudinal(fcs[-1]), attr,
                   getattr(get_lateral(fcs[-1]), attr, 0.0))


def _bilinear(altitude_ft: float, mach: float, attr: str, *, lateral: bool) -> float:
    alts = sorted(_CRUISE_GRID.keys())
    h = float(np.clip(altitude_ft, alts[0], alts[-1]))
    for i in range(len(alts) - 1):
        if alts[i] <= h <= alts[i + 1]:
            row0 = _CRUISE_GRID[alts[i]]
            row1 = _CRUISE_GRID[alts[i + 1]]
            getter = get_lateral if lateral else get_longitudinal
            # interp along Mach in each row
            v0 = _interp_along_one(row0, mach, attr, getter)
            v1 = _interp_along_one(row1, mach, attr, getter)
            t = (h - alts[i]) / (alts[i + 1] - alts[i])
            return float(v0 + t * (v1 - v0))
    return 0.0


def _interp_along_one(
    row: list[tuple[float, int]], mach: float, attr: str, getter
) -> float:
    machs = [m for m, _ in row]
    fcs = [fc for _, fc in row]
    m_clamped = float(np.clip(mach, machs[0], machs[-1]))
    for i in range(len(machs) - 1):
        if machs[i] <= m_clamped <= machs[i + 1]:
            v0 = getattr(getter(fcs[i]), attr)
            v1 = getattr(getter(fcs[i + 1]), attr)
            t = (m_clamped - machs[i]) / (machs[i + 1] - machs[i])
            return float(v0 + t * (v1 - v0))
    return float(getattr(getter(fcs[-1]), attr))


def cruise_longitudinal_at(
    altitude_ft: float, mach: float
) -> LongitudinalDerivatives:
    """Bilinear-interpolated longitudinal derivatives over the cruise grid.

    Use only inside the published envelope (h ∈ [SL, 40 kft],
    M ∈ [0.45..0.90]); outside it the values are clamped to the
    envelope boundary and the interpolation degrades to extrapolation
    of the nearest band.
    """
    fields = LongitudinalDerivatives.__dataclass_fields__
    return LongitudinalDerivatives(
        **{f: _bilinear(altitude_ft, mach, f, lateral=False) for f in fields}
    )


def cruise_lateral_at(
    altitude_ft: float, mach: float
) -> LateralDirectionalDerivatives:
    """Bilinear-interpolated lateral-directional derivatives."""
    fields = LateralDirectionalDerivatives.__dataclass_fields__
    return LateralDirectionalDerivatives(
        **{f: _bilinear(altitude_ft, mach, f, lateral=True) for f in fields}
    )
