"""Aerodynamic coefficient functions for the F-16 longitudinal model.

Direct port of GetCy.m and GetMz.m. Tables are loaded once at import.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

_AERO_DIR = Path(__file__).parent / "aero_tables"


def _clamped_lookup(interp: RegularGridInterpolator, point: np.ndarray,
                    bounds: list[tuple[float, float]]) -> float:
    clipped = np.array([np.clip(p, lo, hi) for p, (lo, hi) in zip(point, bounds)])
    return float(interp(clipped)[0])


# ---------- Cy tables ----------

_cy_data = np.load(_AERO_DIR / "getcy.npz")
_alpha1 = _cy_data["alpha1"]
_alpha2 = _cy_data["alpha2"]
_beta1 = _cy_data["beta1"]
_fi1 = _cy_data["fi1"]
_Cy1 = _cy_data["Cy1"]
_Cy_nos1 = _cy_data["Cy_nos1"]
_Cywz1 = _cy_data["Cywz1"]
_dCywz_nos1 = _cy_data["dCywz_nos1"]
_dCy_sb1 = _cy_data["dCy_sb1"]

_interp_cy = RegularGridInterpolator(
    (_alpha1, _beta1, _fi1), _Cy1, method="cubic", bounds_error=False
)
_interp_cy_nos = RegularGridInterpolator(
    (_alpha2, _beta1), _Cy_nos1, method="cubic", bounds_error=False
)
_interp_cywz = PchipInterpolator(_alpha1, _Cywz1, extrapolate=False)
_interp_cywz_nos = PchipInterpolator(_alpha2, _dCywz_nos1, extrapolate=False)
_interp_dcy_sb = PchipInterpolator(_alpha1, _dCy_sb1, extrapolate=False)

_cy_bounds_3d = [
    (float(_alpha1.min()), float(_alpha1.max())),
    (float(_beta1.min()), float(_beta1.max())),
    (float(_fi1.min()), float(_fi1.max())),
]
_cy_bounds_nos = [
    (float(_alpha2.min()), float(_alpha2.max())),
    (float(_beta1.min()), float(_beta1.max())),
]


# ---------- Mz tables ----------

_mz_data = np.load(_AERO_DIR / "getmz.npz")
_mz_alpha1 = _mz_data["alpha1"]
_mz_alpha2 = _mz_data["alpha2"]
_mz_beta1 = _mz_data["beta1"]
_mz_fi1 = _mz_data["fi1"]
_mz_fi2 = _mz_data["fi2"]
_mz1 = _mz_data["mz1"]
_mz_nos1 = _mz_data["mz_nos1"]
_mzwz1 = _mz_data["mzwz1"]
_dmzwz_nos1 = _mz_data["dmzwz_nos1"]
_dmz1 = _mz_data["dmz1"]
_dmz_sb1 = _mz_data["dmz_sb1"]
_eta_fi1 = _mz_data["eta_fi1"]
_dmz_ds1 = _mz_data["dmz_ds1"]

_interp_mz = RegularGridInterpolator(
    (_mz_alpha1, _mz_beta1, _mz_fi1), _mz1, method="cubic", bounds_error=False
)
_interp_mz_nos = RegularGridInterpolator(
    (_mz_alpha2, _mz_beta1), _mz_nos1, method="cubic", bounds_error=False
)
_interp_dmz = PchipInterpolator(_mz_alpha1, _dmz1, extrapolate=False)
_interp_mzwz = PchipInterpolator(_mz_alpha1, _mzwz1, extrapolate=False)
_interp_mzwz_nos = PchipInterpolator(_mz_alpha2, _dmzwz_nos1, extrapolate=False)
_interp_dmz_sb = PchipInterpolator(_mz_alpha1, _dmz_sb1, extrapolate=False)
_interp_eta_fi = PchipInterpolator(_mz_fi1, _eta_fi1, extrapolate=False)
_interp_dmz_ds = RegularGridInterpolator(
    (_mz_alpha1, _mz_fi2), _dmz_ds1, method="cubic", bounds_error=False
)

_mz_bounds_3d = [
    (float(_mz_alpha1.min()), float(_mz_alpha1.max())),
    (float(_mz_beta1.min()), float(_mz_beta1.max())),
    (float(_mz_fi1.min()), float(_mz_fi1.max())),
]
_mz_bounds_nos = [
    (float(_mz_alpha2.min()), float(_mz_alpha2.max())),
    (float(_mz_beta1.min()), float(_mz_beta1.max())),
]
_mz_bounds_ds = [
    (float(_mz_alpha1.min()), float(_mz_alpha1.max())),
    (float(_mz_fi2.min()), float(_mz_fi2.max())),
]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


_DEG25 = math.radians(25)
_DEG60 = math.radians(60)


def get_cy(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Normal-force coefficient. Mirrors longitudinal/matlab_code/GetCy.m."""
    cy = _clamped_lookup(_interp_cy, np.array([alpha, beta, fi]), _cy_bounds_3d)
    cy0 = _clamped_lookup(_interp_cy, np.array([alpha, beta, 0.0]), _cy_bounds_3d)
    cy_nos = _clamped_lookup(_interp_cy_nos, np.array([alpha, beta]), _cy_bounds_nos)
    a1 = _clip(alpha, float(_alpha1.min()), float(_alpha1.max()))
    a2 = _clip(alpha, float(_alpha2.min()), float(_alpha2.max()))
    cywz = float(_interp_cywz(a1)) + float(_interp_cywz_nos(a2)) * (dnos / _DEG25)
    dcy_sb = float(_interp_dcy_sb(a1))

    dcy_nos = cy_nos - cy0
    return cy + dcy_nos * (dnos / _DEG25) + cywz * ((wz * ba) / (2.0 * V)) + dcy_sb * (sb / _DEG60)


def get_mz(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Pitch-moment coefficient. Mirrors longitudinal/matlab_code/GetMz.m."""
    mz = _clamped_lookup(_interp_mz, np.array([alpha, beta, fi]), _mz_bounds_3d)
    mz0 = _clamped_lookup(_interp_mz, np.array([alpha, beta, 0.0]), _mz_bounds_3d)
    mz_nos = _clamped_lookup(_interp_mz_nos, np.array([alpha, beta]), _mz_bounds_nos)
    a1 = _clip(alpha, float(_mz_alpha1.min()), float(_mz_alpha1.max()))
    a2 = _clip(alpha, float(_mz_alpha2.min()), float(_mz_alpha2.max()))
    fi_clip = _clip(fi, float(_mz_fi1.min()), float(_mz_fi1.max()))
    dmz = float(_interp_dmz(a1))
    mzwz = float(_interp_mzwz(a1)) + float(_interp_mzwz_nos(a2)) * (dnos / _DEG25)
    dmz_sb = float(_interp_dmz_sb(a1))
    eta_fi = float(_interp_eta_fi(fi_clip))
    dmz_ds = _clamped_lookup(_interp_dmz_ds, np.array([alpha, fi]), _mz_bounds_ds)

    dmz_nos = mz_nos - mz0
    return (
        mz * eta_fi
        + dmz_nos * (dnos / _DEG25)
        + dmz
        + mzwz * ((wz * ba) / (2.0 * V))
        + dmz_sb * (sb / _DEG60)
        + dmz_ds
    )
