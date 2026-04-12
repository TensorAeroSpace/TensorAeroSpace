"""Aerodynamic coefficient functions for the F-16 longitudinal model.

Direct port of GetCy.m and GetMz.m. Tables are loaded once at import.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

AERO_TABLE_DIR = Path(__file__).parent / "aero_tables"


def _interp1(interp, value, lo, hi) -> float:
    """1-D interpolator query, scalar-clipped. Returns Python float."""
    return float(interp(min(max(value, lo), hi)))


def _interp2(interp, x0, x1, lo0, hi0, lo1, hi1) -> float:
    """2-D RegularGridInterpolator query, scalar-clipped per axis."""
    pt = np.array([[min(max(x0, lo0), hi0), min(max(x1, lo1), hi1)]])
    return float(interp(pt)[0])


def _interp3(interp, x0, x1, x2, lo0, hi0, lo1, hi1, lo2, hi2) -> float:
    """3-D RegularGridInterpolator query, scalar-clipped per axis."""
    pt = np.array([[min(max(x0, lo0), hi0),
                    min(max(x1, lo1), hi1),
                    min(max(x2, lo2), hi2)]])
    return float(interp(pt)[0])


# ---------- Cy tables ----------

_cy_data = np.load(AERO_TABLE_DIR / "getcy.npz")
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

_alpha1_lo, _alpha1_hi = float(_alpha1.min()), float(_alpha1.max())
_alpha2_lo, _alpha2_hi = float(_alpha2.min()), float(_alpha2.max())
_beta1_lo, _beta1_hi = float(_beta1.min()), float(_beta1.max())
_fi1_lo, _fi1_hi = float(_fi1.min()), float(_fi1.max())


# ---------- Mz tables ----------

_mz_data = np.load(AERO_TABLE_DIR / "getmz.npz")
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

_mz_alpha1_lo, _mz_alpha1_hi = float(_mz_alpha1.min()), float(_mz_alpha1.max())
_mz_alpha2_lo, _mz_alpha2_hi = float(_mz_alpha2.min()), float(_mz_alpha2.max())
_mz_beta1_lo, _mz_beta1_hi = float(_mz_beta1.min()), float(_mz_beta1.max())
_mz_fi1_lo, _mz_fi1_hi = float(_mz_fi1.min()), float(_mz_fi1.max())
_mz_fi2_lo, _mz_fi2_hi = float(_mz_fi2.min()), float(_mz_fi2.max())


_NOS_NORM = math.radians(25)  # slat deflection normalization (max ±25°)
_SB_NORM = math.radians(60)   # speedbrake deflection normalization (max ±60°)


def get_cy(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Normal-force coefficient. Mirrors longitudinal/matlab_code/GetCy.m."""
    cy = _interp3(_interp_cy, alpha, beta, fi,
                  _alpha1_lo, _alpha1_hi, _beta1_lo, _beta1_hi, _fi1_lo, _fi1_hi)
    cy0 = _interp3(_interp_cy, alpha, beta, 0.0,
                   _alpha1_lo, _alpha1_hi, _beta1_lo, _beta1_hi, _fi1_lo, _fi1_hi)
    cy_nos = _interp2(_interp_cy_nos, alpha, beta,
                      _alpha2_lo, _alpha2_hi, _beta1_lo, _beta1_hi)
    cywz = (_interp1(_interp_cywz, alpha, _alpha1_lo, _alpha1_hi)
            + _interp1(_interp_cywz_nos, alpha, _alpha2_lo, _alpha2_hi) * (dnos / _NOS_NORM))
    dcy_sb = _interp1(_interp_dcy_sb, alpha, _alpha1_lo, _alpha1_hi)

    dcy_nos = cy_nos - cy0
    return cy + dcy_nos * (dnos / _NOS_NORM) + cywz * ((wz * ba) / (2.0 * V)) + dcy_sb * (sb / _SB_NORM)


def get_mz(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Pitch-moment coefficient. Mirrors longitudinal/matlab_code/GetMz.m."""
    mz = _interp3(_interp_mz, alpha, beta, fi,
                  _mz_alpha1_lo, _mz_alpha1_hi, _mz_beta1_lo, _mz_beta1_hi,
                  _mz_fi1_lo, _mz_fi1_hi)
    mz0 = _interp3(_interp_mz, alpha, beta, 0.0,
                   _mz_alpha1_lo, _mz_alpha1_hi, _mz_beta1_lo, _mz_beta1_hi,
                   _mz_fi1_lo, _mz_fi1_hi)
    mz_nos = _interp2(_interp_mz_nos, alpha, beta,
                      _mz_alpha2_lo, _mz_alpha2_hi, _mz_beta1_lo, _mz_beta1_hi)
    dmz = _interp1(_interp_dmz, alpha, _mz_alpha1_lo, _mz_alpha1_hi)
    mzwz = (_interp1(_interp_mzwz, alpha, _mz_alpha1_lo, _mz_alpha1_hi)
            + _interp1(_interp_mzwz_nos, alpha, _mz_alpha2_lo, _mz_alpha2_hi) * (dnos / _NOS_NORM))
    dmz_sb = _interp1(_interp_dmz_sb, alpha, _mz_alpha1_lo, _mz_alpha1_hi)
    eta_fi = _interp1(_interp_eta_fi, fi, _mz_fi1_lo, _mz_fi1_hi)
    dmz_ds = _interp2(_interp_dmz_ds, alpha, fi,
                      _mz_alpha1_lo, _mz_alpha1_hi, _mz_fi2_lo, _mz_fi2_hi)

    dmz_nos = mz_nos - mz0
    return (
        mz * eta_fi
        + dmz_nos * (dnos / _NOS_NORM)
        + dmz
        + mzwz * ((wz * ba) / (2.0 * V))
        + dmz_sb * (sb / _SB_NORM)
        + dmz_ds
    )
