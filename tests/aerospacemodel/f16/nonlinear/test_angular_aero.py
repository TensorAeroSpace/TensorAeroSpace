"""Tests for the angular aero coefficient functions.

For each of the 6 functions we test:
1. Finite output at zero-state
2. Grid-node match: at a known grid corner with all secondary terms zero,
   the assembled output should reproduce the raw table value within 1e-3.

If a grid-node test fails by more than 1e-2 that indicates an assembly or
table indexing bug and should be reported as BLOCKED.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.python.angular.aero import (
    AERO_TABLE_DIR,
    get_cx,
    get_cy,
    get_cz,
    get_mx,
    get_my,
    get_mz,
)

# ---- common parameter values used as stand-ins when terms drop out ----
_V = 120.0   # m/s
_BA = 3.45   # wing span
_L = 9.144   # characteristic length

# ---- grid coordinates used in grid-node tests ----
# alpha1[5] = 5 deg, beta1[9] = 0 rad (middle of beta grid), fi=0
_A5 = math.radians(5.0)
_B0 = 0.0


# ===========================================================================
# Cx
# ===========================================================================

class TestGetCx:

    def test_finite_at_zero_state(self):
        val = get_cx(0.0, 0.0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """At alpha=5deg (alpha1[5]), beta=0 (beta1[9]), fi=0 (fi1[2]),
        with dnos=wz=sb=0 all correction terms drop out.
        Expected = Cx1[5,9,2] = 0.0066
        """
        d = np.load(AERO_TABLE_DIR / "getcx.npz")
        expected = float(d["Cx1"][5, 9, 2])   # fi1[2] = 0
        val = get_cx(_A5, _B0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert val == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# Cy
# ===========================================================================

class TestGetCy:

    def test_finite_at_zero_state(self):
        val = get_cy(0.0, 0.0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """Cy1[5,9,2] at fi=0 should be reproduced when dnos=wz=sb=0."""
        d = np.load(AERO_TABLE_DIR / "getcy.npz")
        expected = float(d["Cy1"][5, 9, 2])
        val = get_cy(_A5, _B0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert val == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# Cz
# ===========================================================================

class TestGetCz:

    def test_finite_at_zero_state(self):
        val = get_cz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """Cz has no fi axis.  At alpha=5deg (alpha1[5]), beta=0 (beta1[9])
        with drn=del_=dnos=wx=wy=0, all correction terms drop out.
        Expected = Cz1[5,9] = 0.0
        """
        d = np.load(AERO_TABLE_DIR / "getcz.npz")
        expected = float(d["Cz1"][5, 9])
        val = get_cz(_A5, _B0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert val == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# Mx
# ===========================================================================

class TestGetMx:

    def test_finite_at_zero_state(self):
        val = get_mx(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """mx1 uses fi2 axis [-25,0,25 deg]; fi2[1]=0.
        At alpha=5deg (alpha1[5]), beta=0 (beta1[9]), fi=0 (fi2[1]),
        with drn=del_=dnos=wx=wy=0 and beta=0 (so dmxbt*beta=0):
        mx = mx1[5,9,1] = 0.0
        """
        d = np.load(AERO_TABLE_DIR / "getmx.npz")
        expected = float(d["mx1"][5, 9, 1])   # fi2[1] = 0
        val = get_mx(_A5, _B0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert val == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# My
# ===========================================================================

class TestGetMy:

    def test_finite_at_zero_state(self):
        val = get_my(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """my1 uses fi2 axis; fi2[1]=0.
        At alpha=5deg, beta=0, fi=0 with drn=del_=dnos=wx=wy=0 and beta=0:
        my = my1[5,9,1] = 0.0
        """
        d = np.load(AERO_TABLE_DIR / "getmy.npz")
        expected = float(d["my1"][5, 9, 1])
        val = get_my(_A5, _B0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _V, _L)
        assert val == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# Mz
# ===========================================================================

class TestGetMz:

    def test_finite_at_zero_state(self):
        val = get_mz(0.0, 0.0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert math.isfinite(val)

    def test_grid_node_matches_table(self):
        """At alpha=5deg (alpha1[5]), beta=0, fi=0 (fi1[2]),
        dnos=0, wz=0, sb=0.

        Assembly (from GetMz.m):
          mz = mz1[5,9,2] * eta_fi(0) + 0 + dmz1[5] + 0 + 0 + dmz_ds[5,fi2=0]
          fi1[2]=0 → eta_fi=1.0
          fi2[2]=0 → dmz_ds[5,2]=0.0
          => mz = -0.0498 * 1.0 + 0.019 + 0.0 = -0.0308
        """
        d = np.load(AERO_TABLE_DIR / "getmz.npz")
        mz_table = float(d["mz1"][5, 9, 2])
        eta = float(d["eta_fi1"][2])       # fi1[2] = 0 → 1.0
        dmz = float(d["dmz1"][5])
        dmz_ds = float(d["dmz_ds1"][5, 2]) # fi2[2] = 0
        expected = mz_table * eta + dmz + dmz_ds

        val = get_mz(_A5, _B0, 0.0, 0.0, 0.0, _V, _BA, 0.0)
        assert val == pytest.approx(expected, abs=1e-3)
