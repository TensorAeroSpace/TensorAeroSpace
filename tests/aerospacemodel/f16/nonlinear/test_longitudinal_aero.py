import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal import aero

_AERO_DIR = aero._AERO_DIR


def test_get_cy_at_zero_state_returns_finite():
    val = aero.get_cy(alpha=0.0, beta=0.0, fi=0.0, dnos=0.0, wz=0.0,
                     V=150.0, ba=3.45, sb=0.0)
    assert math.isfinite(val)


def test_get_cy_grid_node_matches_table_value():
    """At a grid node with no secondary contributions, get_cy should equal
    the corresponding Cy1 table entry (cubic spline interpolates exactly at
    data points to within scipy's rounding tolerance)."""
    raw = np.load(_AERO_DIR / "getcy.npz")
    alpha1 = raw["alpha1"]
    beta1 = raw["beta1"]
    fi1 = raw["fi1"]
    Cy1 = raw["Cy1"]

    i_alpha = int(np.argmin(np.abs(alpha1 - 0.0)))
    i_beta = int(np.argmin(np.abs(beta1 - 0.0)))
    i_fi = int(np.argmin(np.abs(fi1 - 0.0)))
    expected_base = Cy1[i_alpha, i_beta, i_fi]

    val = aero.get_cy(
        alpha=float(alpha1[i_alpha]), beta=float(beta1[i_beta]), fi=float(fi1[i_fi]),
        dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0,
    )
    # At this exact node, Cy0 == Cy1[i,j,k] so dCy_nos contribution is 0,
    # and wz=sb=0. So the result should equal Cy_nos1 at (alpha, beta) only
    # via the dCy_nos = Cy_nos - Cy0 term, which should be small/zero at the
    # node. Allow some slack because Cy0 evaluates the 3-D table at fi=0,
    # which is a different node — they only match if i_fi already corresponds
    # to fi=0. Pick the alpha/beta indices such that this holds.
    assert val == pytest.approx(expected_base, rel=1e-4, abs=1e-4)


def test_get_cy_pitch_rate_contribution_is_linear_in_wz():
    """The Cywz term is added linearly: doubling wz doubles the increment."""
    a = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=0.0,
                   V=150.0, ba=3.45, sb=0.0)
    b = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=0.5,
                   V=150.0, ba=3.45, sb=0.0)
    c = aero.get_cy(alpha=0.1, beta=0.0, fi=0.0, dnos=0.0, wz=1.0,
                   V=150.0, ba=3.45, sb=0.0)
    assert (c - a) == pytest.approx(2.0 * (b - a), rel=1e-9)


def test_get_mz_at_grid_node_matches_table():
    raw = np.load(_AERO_DIR / "getmz.npz")
    alpha1 = raw["alpha1"]
    beta1 = raw["beta1"]
    fi1 = raw["fi1"]
    mz1 = raw["mz1"]
    eta_fi1 = raw["eta_fi1"]
    dmz1 = raw["dmz1"]

    i_alpha = int(np.argmin(np.abs(alpha1 - 0.0)))
    i_beta = int(np.argmin(np.abs(beta1 - 0.0)))
    i_fi = int(np.argmin(np.abs(fi1 - 0.0)))
    expected = mz1[i_alpha, i_beta, i_fi] * eta_fi1[i_fi] + dmz1[i_alpha]

    val = aero.get_mz(
        alpha=float(alpha1[i_alpha]), beta=float(beta1[i_beta]), fi=float(fi1[i_fi]),
        dnos=0.0, wz=0.0, V=150.0, ba=3.45, sb=0.0,
    )
    assert val == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_clamp_out_of_bounds_inputs_does_not_blow_up():
    val = aero.get_cy(
        alpha=math.radians(200.0), beta=0.0, fi=0.0, dnos=0.0, wz=0.0,
        V=150.0, ba=3.45, sb=0.0,
    )
    assert math.isfinite(val)
