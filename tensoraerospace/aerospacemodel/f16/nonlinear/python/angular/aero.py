"""Aero coefficient functions for F-16 angular model.

Direct port of angular/matlab_code/Get{Cx,Cy,Cz,Mx,My,Mz}.m.
GetThrust, EnginePowerLevel, body2wind, wind2body are out-of-scope
for the angular ODE and are intentionally omitted.

Tables are loaded once at import time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

AERO_TABLE_DIR = Path(__file__).parent / "aero_tables"


# ---------------------------------------------------------------------------
# Helpers (same signature as longitudinal aero.py)
# ---------------------------------------------------------------------------

def _interp1(interp, value, lo, hi) -> float:
    """1-D PCHIP query, scalar-clipped. Returns Python float."""
    return float(interp(min(max(value, lo), hi)))


def _interp2(interp, x0, x1, lo0, hi0, lo1, hi1) -> float:
    """2-D RegularGridInterpolator query, clipped per axis."""
    pt = np.array([[min(max(x0, lo0), hi0), min(max(x1, lo1), hi1)]])
    return float(interp(pt)[0])


def _interp3(interp, x0, x1, x2, lo0, hi0, lo1, hi1, lo2, hi2) -> float:
    """3-D RegularGridInterpolator query, clipped per axis."""
    pt = np.array([[min(max(x0, lo0), hi0),
                    min(max(x1, lo1), hi1),
                    min(max(x2, lo2), hi2)]])
    return float(interp(pt)[0])


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
_NOS_NORM = math.radians(25)   # slat / leading-edge flap normalisation
_SB_NORM = math.radians(60)    # speedbrake normalisation
_DEL_NORM = math.radians(20)   # aileron table reference (Czdel20 / mxdel20 / mydel20)
_DRN_NORM = math.radians(-30)  # rudder table reference (Czdrn30 / mxdrn30 / mydrn30) — negative!


# ---------------------------------------------------------------------------
# Cx tables  (GetCx.m)
# ---------------------------------------------------------------------------
_cx = np.load(AERO_TABLE_DIR / "getcx.npz")
_cx_alpha1 = _cx["alpha1"]
_cx_alpha2 = _cx["alpha2"]
_cx_beta1 = _cx["beta1"]
_cx_fi1 = _cx["fi1"]

_interp_cx = RegularGridInterpolator(
    (_cx_alpha1, _cx_beta1, _cx_fi1), _cx["Cx1"], method="cubic", bounds_error=False
)
_interp_cx_nos = RegularGridInterpolator(
    (_cx_alpha2, _cx_beta1), _cx["Cx_nos1"], method="cubic", bounds_error=False
)
_interp_cxwz = PchipInterpolator(_cx_alpha1, _cx["Cxwz1"], extrapolate=False)
_interp_cxwz_nos = PchipInterpolator(_cx_alpha2, _cx["dCxwz_nos1"], extrapolate=False)
_interp_dcx_sb = PchipInterpolator(_cx_alpha1, _cx["dCx_sb1"], extrapolate=False)

_cx_a1_lo, _cx_a1_hi = float(_cx_alpha1.min()), float(_cx_alpha1.max())
_cx_a2_lo, _cx_a2_hi = float(_cx_alpha2.min()), float(_cx_alpha2.max())
_cx_b_lo, _cx_b_hi = float(_cx_beta1.min()), float(_cx_beta1.max())
_cx_fi_lo, _cx_fi_hi = float(_cx_fi1.min()), float(_cx_fi1.max())


# ---------------------------------------------------------------------------
# Cy tables  (GetCy.m — identical structure to Cx)
# ---------------------------------------------------------------------------
_cy = np.load(AERO_TABLE_DIR / "getcy.npz")
_cy_alpha1 = _cy["alpha1"]
_cy_alpha2 = _cy["alpha2"]
_cy_beta1 = _cy["beta1"]
_cy_fi1 = _cy["fi1"]

_interp_cy = RegularGridInterpolator(
    (_cy_alpha1, _cy_beta1, _cy_fi1), _cy["Cy1"], method="cubic", bounds_error=False
)
_interp_cy_nos = RegularGridInterpolator(
    (_cy_alpha2, _cy_beta1), _cy["Cy_nos1"], method="cubic", bounds_error=False
)
_interp_cywz = PchipInterpolator(_cy_alpha1, _cy["Cywz1"], extrapolate=False)
_interp_cywz_nos = PchipInterpolator(_cy_alpha2, _cy["dCywz_nos1"], extrapolate=False)
_interp_dcy_sb = PchipInterpolator(_cy_alpha1, _cy["dCy_sb1"], extrapolate=False)

_cy_a1_lo, _cy_a1_hi = float(_cy_alpha1.min()), float(_cy_alpha1.max())
_cy_a2_lo, _cy_a2_hi = float(_cy_alpha2.min()), float(_cy_alpha2.max())
_cy_b_lo, _cy_b_hi = float(_cy_beta1.min()), float(_cy_beta1.max())
_cy_fi_lo, _cy_fi_hi = float(_cy_fi1.min()), float(_cy_fi1.max())


# ---------------------------------------------------------------------------
# Cz tables  (GetCz.m — no fi axis; has lateral-rate and aileron/rudder terms)
# ---------------------------------------------------------------------------
_cz = np.load(AERO_TABLE_DIR / "getcz.npz")
_cz_alpha1 = _cz["alpha1"]
_cz_alpha2 = _cz["alpha2"]
_cz_beta1 = _cz["beta1"]

_interp_cz = RegularGridInterpolator(
    (_cz_alpha1, _cz_beta1), _cz["Cz1"], method="cubic", bounds_error=False
)
_interp_cz_nos = RegularGridInterpolator(
    (_cz_alpha2, _cz_beta1), _cz["Cz_nos1"], method="cubic", bounds_error=False
)
_interp_czdel20 = RegularGridInterpolator(
    (_cz_alpha1, _cz_beta1), _cz["Czdel20"], method="cubic", bounds_error=False
)
_interp_czdel20_nos = RegularGridInterpolator(
    (_cz_alpha2, _cz_beta1), _cz["Czdel20_nos"], method="cubic", bounds_error=False
)
_interp_czdrn30 = RegularGridInterpolator(
    (_cz_alpha1, _cz_beta1), _cz["Czdrn30"], method="cubic", bounds_error=False
)
_interp_czwx = PchipInterpolator(_cz_alpha1, _cz["Czwx1"], extrapolate=False)
_interp_czwx_nos = PchipInterpolator(_cz_alpha2, _cz["dCzwx_nos1"], extrapolate=False)
_interp_czwy = PchipInterpolator(_cz_alpha1, _cz["Czwy1"], extrapolate=False)
_interp_czwy_nos = PchipInterpolator(_cz_alpha2, _cz["dCzwy_nos1"], extrapolate=False)

_cz_a1_lo, _cz_a1_hi = float(_cz_alpha1.min()), float(_cz_alpha1.max())
_cz_a2_lo, _cz_a2_hi = float(_cz_alpha2.min()), float(_cz_alpha2.max())
_cz_b_lo, _cz_b_hi = float(_cz_beta1.min()), float(_cz_beta1.max())


# ---------------------------------------------------------------------------
# Mx tables  (GetMx.m — uses fi2 axis [-25,0,25 deg])
# ---------------------------------------------------------------------------
_mx = np.load(AERO_TABLE_DIR / "getmx.npz")
_mx_alpha1 = _mx["alpha1"]
_mx_alpha2 = _mx["alpha2"]
_mx_beta1 = _mx["beta1"]
_mx_fi2 = _mx["fi2"]

_interp_mx = RegularGridInterpolator(
    (_mx_alpha1, _mx_beta1, _mx_fi2), _mx["mx1"], method="linear", bounds_error=False
)
_interp_mx_nos = RegularGridInterpolator(
    (_mx_alpha2, _mx_beta1), _mx["mx_nos1"], method="cubic", bounds_error=False
)
_interp_mxdel20 = RegularGridInterpolator(
    (_mx_alpha1, _mx_beta1), _mx["mxdel20"], method="cubic", bounds_error=False
)
_interp_mxdel20_nos = RegularGridInterpolator(
    (_mx_alpha2, _mx_beta1), _mx["mxdel20_nos"], method="cubic", bounds_error=False
)
_interp_mxdrn30 = RegularGridInterpolator(
    (_mx_alpha1, _mx_beta1), _mx["mxdrn30"], method="cubic", bounds_error=False
)
_interp_dmxbt = PchipInterpolator(_mx_alpha1, _mx["dmxbt1"], extrapolate=False)
_interp_mxwx = PchipInterpolator(_mx_alpha1, _mx["mxwx1"], extrapolate=False)
_interp_mxwx_nos = PchipInterpolator(_mx_alpha2, _mx["dmxwx_nos1"], extrapolate=False)
_interp_mxwy = PchipInterpolator(_mx_alpha1, _mx["mxwy1"], extrapolate=False)
_interp_mxwy_nos = PchipInterpolator(_mx_alpha2, _mx["dmxwy_nos1"], extrapolate=False)

_mx_a1_lo, _mx_a1_hi = float(_mx_alpha1.min()), float(_mx_alpha1.max())
_mx_a2_lo, _mx_a2_hi = float(_mx_alpha2.min()), float(_mx_alpha2.max())
_mx_b_lo, _mx_b_hi = float(_mx_beta1.min()), float(_mx_beta1.max())
_mx_fi2_lo, _mx_fi2_hi = float(_mx_fi2.min()), float(_mx_fi2.max())


# ---------------------------------------------------------------------------
# My tables  (GetMy.m — same structure as Mx)
# ---------------------------------------------------------------------------
_my = np.load(AERO_TABLE_DIR / "getmy.npz")
_my_alpha1 = _my["alpha1"]
_my_alpha2 = _my["alpha2"]
_my_beta1 = _my["beta1"]
_my_fi2 = _my["fi2"]

_interp_my = RegularGridInterpolator(
    (_my_alpha1, _my_beta1, _my_fi2), _my["my1"], method="linear", bounds_error=False
)
_interp_my_nos = RegularGridInterpolator(
    (_my_alpha2, _my_beta1), _my["my_nos1"], method="cubic", bounds_error=False
)
_interp_mydel20 = RegularGridInterpolator(
    (_my_alpha1, _my_beta1), _my["mydel20"], method="cubic", bounds_error=False
)
_interp_mydel20_nos = RegularGridInterpolator(
    (_my_alpha2, _my_beta1), _my["mydel20_nos"], method="cubic", bounds_error=False
)
_interp_mydrn30 = RegularGridInterpolator(
    (_my_alpha1, _my_beta1), _my["mydrn30"], method="cubic", bounds_error=False
)
_interp_dmybt = PchipInterpolator(_my_alpha1, _my["dmybt1"], extrapolate=False)
_interp_mywx = PchipInterpolator(_my_alpha1, _my["mywx1"], extrapolate=False)
_interp_mywx_nos = PchipInterpolator(_my_alpha2, _my["dmywx_nos1"], extrapolate=False)
_interp_mywy = PchipInterpolator(_my_alpha1, _my["mywy1"], extrapolate=False)
_interp_mywy_nos = PchipInterpolator(_my_alpha2, _my["dmywy_nos1"], extrapolate=False)

_my_a1_lo, _my_a1_hi = float(_my_alpha1.min()), float(_my_alpha1.max())
_my_a2_lo, _my_a2_hi = float(_my_alpha2.min()), float(_my_alpha2.max())
_my_b_lo, _my_b_hi = float(_my_beta1.min()), float(_my_beta1.max())
_my_fi2_lo, _my_fi2_hi = float(_my_fi2.min()), float(_my_fi2.max())


# ---------------------------------------------------------------------------
# Mz tables  (GetMz.m — uses fi1 axis [-25,-10,0,10,25] and fi2 axis for dmz_ds)
# ---------------------------------------------------------------------------
_mz = np.load(AERO_TABLE_DIR / "getmz.npz")
_mz_alpha1 = _mz["alpha1"]
_mz_alpha2 = _mz["alpha2"]
_mz_beta1 = _mz["beta1"]
_mz_fi1 = _mz["fi1"]
_mz_fi2 = _mz["fi2"]

_interp_mz = RegularGridInterpolator(
    (_mz_alpha1, _mz_beta1, _mz_fi1), _mz["mz1"], method="cubic", bounds_error=False
)
_interp_mz_nos = RegularGridInterpolator(
    (_mz_alpha2, _mz_beta1), _mz["mz_nos1"], method="cubic", bounds_error=False
)
_interp_dmz = PchipInterpolator(_mz_alpha1, _mz["dmz1"], extrapolate=False)
_interp_mzwz = PchipInterpolator(_mz_alpha1, _mz["mzwz1"], extrapolate=False)
_interp_mzwz_nos = PchipInterpolator(_mz_alpha2, _mz["dmzwz_nos1"], extrapolate=False)
_interp_dmz_sb = PchipInterpolator(_mz_alpha1, _mz["dmz_sb1"], extrapolate=False)
_interp_eta_fi = PchipInterpolator(_mz_fi1, _mz["eta_fi1"], extrapolate=False)
_interp_dmz_ds = RegularGridInterpolator(
    (_mz_alpha1, _mz_fi2), _mz["dmz_ds1"], method="cubic", bounds_error=False
)

_mz_a1_lo, _mz_a1_hi = float(_mz_alpha1.min()), float(_mz_alpha1.max())
_mz_a2_lo, _mz_a2_hi = float(_mz_alpha2.min()), float(_mz_alpha2.max())
_mz_b_lo, _mz_b_hi = float(_mz_beta1.min()), float(_mz_beta1.max())
_mz_fi1_lo, _mz_fi1_hi = float(_mz_fi1.min()), float(_mz_fi1.max())
_mz_fi2_lo, _mz_fi2_hi = float(_mz_fi2.min()), float(_mz_fi2.max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cx(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Longitudinal-force coefficient Cx. Mirrors GetCx.m.

    Parameters mirror matlab: (alpha, beta, fi=stab, dnos=lef, Wz, V, ba, sb).
    """
    cx = _interp3(_interp_cx, alpha, beta, fi,
                  _cx_a1_lo, _cx_a1_hi, _cx_b_lo, _cx_b_hi, _cx_fi_lo, _cx_fi_hi)
    cx0 = _interp3(_interp_cx, alpha, beta, 0.0,
                   _cx_a1_lo, _cx_a1_hi, _cx_b_lo, _cx_b_hi, _cx_fi_lo, _cx_fi_hi)
    cx_nos = _interp2(_interp_cx_nos, alpha, beta,
                      _cx_a2_lo, _cx_a2_hi, _cx_b_lo, _cx_b_hi)
    cxwz = (_interp1(_interp_cxwz, alpha, _cx_a1_lo, _cx_a1_hi)
            + _interp1(_interp_cxwz_nos, alpha, _cx_a2_lo, _cx_a2_hi) * (dnos / _NOS_NORM))
    dcx_sb = _interp1(_interp_dcx_sb, alpha, _cx_a1_lo, _cx_a1_hi)

    dcx_nos = cx_nos - cx0
    return (cx
            + dcx_nos * (dnos / _NOS_NORM)
            + cxwz * ((wz * ba) / (2.0 * V))
            + dcx_sb * (sb / _SB_NORM))


def get_cy(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Normal-force coefficient Cy. Mirrors GetCy.m.

    Same argument layout and assembly as GetCx.
    """
    cy = _interp3(_interp_cy, alpha, beta, fi,
                  _cy_a1_lo, _cy_a1_hi, _cy_b_lo, _cy_b_hi, _cy_fi_lo, _cy_fi_hi)
    cy0 = _interp3(_interp_cy, alpha, beta, 0.0,
                   _cy_a1_lo, _cy_a1_hi, _cy_b_lo, _cy_b_hi, _cy_fi_lo, _cy_fi_hi)
    cy_nos = _interp2(_interp_cy_nos, alpha, beta,
                      _cy_a2_lo, _cy_a2_hi, _cy_b_lo, _cy_b_hi)
    cywz = (_interp1(_interp_cywz, alpha, _cy_a1_lo, _cy_a1_hi)
            + _interp1(_interp_cywz_nos, alpha, _cy_a2_lo, _cy_a2_hi) * (dnos / _NOS_NORM))
    dcy_sb = _interp1(_interp_dcy_sb, alpha, _cy_a1_lo, _cy_a1_hi)

    dcy_nos = cy_nos - cy0
    return (cy
            + dcy_nos * (dnos / _NOS_NORM)
            + cywz * ((wz * ba) / (2.0 * V))
            + dcy_sb * (sb / _SB_NORM))


def get_cz(alpha: float, beta: float, drn: float, del_: float, dnos: float,
           wx: float, wy: float, V: float, l: float) -> float:
    """Lateral-force coefficient Cz. Mirrors GetCz.m.

    Note: ``del`` is a Python keyword so the parameter is named ``del_``.
    """
    cz = _interp2(_interp_cz, alpha, beta,
                  _cz_a1_lo, _cz_a1_hi, _cz_b_lo, _cz_b_hi)
    cz_nos = _interp2(_interp_cz_nos, alpha, beta,
                      _cz_a2_lo, _cz_a2_hi, _cz_b_lo, _cz_b_hi)
    czdel = _interp2(_interp_czdel20, alpha, beta,
                     _cz_a1_lo, _cz_a1_hi, _cz_b_lo, _cz_b_hi)
    czdel_nos = _interp2(_interp_czdel20_nos, alpha, beta,
                         _cz_a2_lo, _cz_a2_hi, _cz_b_lo, _cz_b_hi)
    czdrn = _interp2(_interp_czdrn30, alpha, beta,
                     _cz_a1_lo, _cz_a1_hi, _cz_b_lo, _cz_b_hi)

    dcz_nos = cz_nos - cz
    dczdel = czdel - cz
    dczdel_nos = czdel_nos - cz_nos - dczdel
    dczdrn = czdrn - cz

    czwx = (_interp1(_interp_czwx, alpha, _cz_a1_lo, _cz_a1_hi)
            + _interp1(_interp_czwx_nos, alpha, _cz_a2_lo, _cz_a2_hi) * (dnos / _NOS_NORM))
    czwy = (_interp1(_interp_czwy, alpha, _cz_a1_lo, _cz_a1_hi)
            + _interp1(_interp_czwy_nos, alpha, _cz_a2_lo, _cz_a2_hi) * (dnos / _NOS_NORM))

    return (cz
            + dcz_nos * (dnos / _NOS_NORM)
            + (dczdel + dczdel_nos * (dnos / _NOS_NORM)) * (del_ / _DEL_NORM)
            + dczdrn * (drn / _DRN_NORM)
            + czwx * ((wx * l) / (2.0 * V))
            + czwy * ((wy * l) / (2.0 * V)))


def get_mx(alpha: float, beta: float, fi: float, drn: float, del_: float,
           dnos: float, wx: float, wy: float, V: float, l: float) -> float:
    """Roll-moment coefficient mx. Mirrors GetMx.m.

    ``fi`` here is the stabiliser deflection mapped onto the fi2 grid [-25,0,25 deg].
    """
    mx = _interp3(_interp_mx, alpha, beta, fi,
                  _mx_a1_lo, _mx_a1_hi, _mx_b_lo, _mx_b_hi, _mx_fi2_lo, _mx_fi2_hi)
    mx0 = _interp3(_interp_mx, alpha, beta, 0.0,
                   _mx_a1_lo, _mx_a1_hi, _mx_b_lo, _mx_b_hi, _mx_fi2_lo, _mx_fi2_hi)
    mx_nos = _interp2(_interp_mx_nos, alpha, beta,
                      _mx_a2_lo, _mx_a2_hi, _mx_b_lo, _mx_b_hi)
    mxdel = _interp2(_interp_mxdel20, alpha, beta,
                     _mx_a1_lo, _mx_a1_hi, _mx_b_lo, _mx_b_hi)
    mxdel_nos = _interp2(_interp_mxdel20_nos, alpha, beta,
                         _mx_a2_lo, _mx_a2_hi, _mx_b_lo, _mx_b_hi)
    mxdrn = _interp2(_interp_mxdrn30, alpha, beta,
                     _mx_a1_lo, _mx_a1_hi, _mx_b_lo, _mx_b_hi)

    dmx_nos = mx_nos - mx0
    dmxdel = mxdel - mx0
    dmxdel_nos = mxdel_nos - mx_nos - dmxdel
    dmxdrn = mxdrn - mx0

    mxwx = (_interp1(_interp_mxwx, alpha, _mx_a1_lo, _mx_a1_hi)
            + _interp1(_interp_mxwx_nos, alpha, _mx_a2_lo, _mx_a2_hi) * (dnos / _NOS_NORM))
    mxwy = (_interp1(_interp_mxwy, alpha, _mx_a1_lo, _mx_a1_hi)
            + _interp1(_interp_mxwy_nos, alpha, _mx_a2_lo, _mx_a2_hi) * (dnos / _NOS_NORM))
    dmxbt = _interp1(_interp_dmxbt, alpha, _mx_a1_lo, _mx_a1_hi)

    return (mx
            + dmx_nos * (dnos / _NOS_NORM)
            + (dmxdel + dmxdel_nos * (dnos / _NOS_NORM)) * (del_ / _DEL_NORM)
            + dmxdrn * (drn / _DRN_NORM)
            + mxwx * ((wx * l) / (2.0 * V))
            + mxwy * ((wy * l) / (2.0 * V))
            + dmxbt * beta)


def get_my(alpha: float, beta: float, fi: float, drn: float, del_: float,
           dnos: float, wx: float, wy: float, V: float, l: float) -> float:
    """Yaw-moment coefficient my. Mirrors GetMy.m.

    Same structure as GetMx, with ``my*`` tables.
    """
    my = _interp3(_interp_my, alpha, beta, fi,
                  _my_a1_lo, _my_a1_hi, _my_b_lo, _my_b_hi, _my_fi2_lo, _my_fi2_hi)
    my0 = _interp3(_interp_my, alpha, beta, 0.0,
                   _my_a1_lo, _my_a1_hi, _my_b_lo, _my_b_hi, _my_fi2_lo, _my_fi2_hi)
    my_nos = _interp2(_interp_my_nos, alpha, beta,
                      _my_a2_lo, _my_a2_hi, _my_b_lo, _my_b_hi)
    mydel = _interp2(_interp_mydel20, alpha, beta,
                     _my_a1_lo, _my_a1_hi, _my_b_lo, _my_b_hi)
    mydel_nos = _interp2(_interp_mydel20_nos, alpha, beta,
                         _my_a2_lo, _my_a2_hi, _my_b_lo, _my_b_hi)
    mydrn = _interp2(_interp_mydrn30, alpha, beta,
                     _my_a1_lo, _my_a1_hi, _my_b_lo, _my_b_hi)

    dmy_nos = my_nos - my0
    dmydel = mydel - my0
    dmydel_nos = mydel_nos - my_nos - dmydel
    dmydrn = mydrn - my0

    mywx = (_interp1(_interp_mywx, alpha, _my_a1_lo, _my_a1_hi)
            + _interp1(_interp_mywx_nos, alpha, _my_a2_lo, _my_a2_hi) * (dnos / _NOS_NORM))
    mywy = (_interp1(_interp_mywy, alpha, _my_a1_lo, _my_a1_hi)
            + _interp1(_interp_mywy_nos, alpha, _my_a2_lo, _my_a2_hi) * (dnos / _NOS_NORM))
    dmybt = _interp1(_interp_dmybt, alpha, _my_a1_lo, _my_a1_hi)

    return (my
            + dmy_nos * (dnos / _NOS_NORM)
            + (dmydel + dmydel_nos * (dnos / _NOS_NORM)) * (del_ / _DEL_NORM)
            + dmydrn * (drn / _DRN_NORM)
            + mywx * ((wx * l) / (2.0 * V))
            + mywy * ((wy * l) / (2.0 * V))
            + dmybt * beta)


def get_mz(alpha: float, beta: float, fi: float, dnos: float,
           wz: float, V: float, ba: float, sb: float) -> float:
    """Pitch-moment coefficient mz. Mirrors GetMz.m.

    ``fi`` is on the fi1 axis [-25,-10,0,10,25 deg]; ``ba`` is the wing span.
    """
    mz = _interp3(_interp_mz, alpha, beta, fi,
                  _mz_a1_lo, _mz_a1_hi, _mz_b_lo, _mz_b_hi, _mz_fi1_lo, _mz_fi1_hi)
    mz0 = _interp3(_interp_mz, alpha, beta, 0.0,
                   _mz_a1_lo, _mz_a1_hi, _mz_b_lo, _mz_b_hi, _mz_fi1_lo, _mz_fi1_hi)
    mz_nos = _interp2(_interp_mz_nos, alpha, beta,
                      _mz_a2_lo, _mz_a2_hi, _mz_b_lo, _mz_b_hi)
    dmz = _interp1(_interp_dmz, alpha, _mz_a1_lo, _mz_a1_hi)
    mzwz = (_interp1(_interp_mzwz, alpha, _mz_a1_lo, _mz_a1_hi)
            + _interp1(_interp_mzwz_nos, alpha, _mz_a2_lo, _mz_a2_hi) * (dnos / _NOS_NORM))
    dmz_sb = _interp1(_interp_dmz_sb, alpha, _mz_a1_lo, _mz_a1_hi)
    eta_fi = _interp1(_interp_eta_fi, fi, _mz_fi1_lo, _mz_fi1_hi)
    dmz_ds = _interp2(_interp_dmz_ds, alpha, fi,
                      _mz_a1_lo, _mz_a1_hi, _mz_fi2_lo, _mz_fi2_hi)

    dmz_nos = mz_nos - mz0
    return (mz * eta_fi
            + dmz_nos * (dnos / _NOS_NORM)
            + dmz
            + mzwz * ((wz * ba) / (2.0 * V))
            + dmz_sb * (sb / _SB_NORM)
            + dmz_ds)
