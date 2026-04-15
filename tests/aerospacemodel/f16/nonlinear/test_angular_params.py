import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
    default_parameters,
)


def test_defaults_match_matlab():
    p = default_parameters()
    assert p.m == pytest.approx(9295.44)
    assert p.l == pytest.approx(9.144)
    assert p.S == pytest.approx(27.87)
    assert p.bA == pytest.approx(3.45)
    assert p.Jx == pytest.approx(12874.8)
    assert p.Jy == pytest.approx(85552.1)
    assert p.Jz == pytest.approx(75673.6)
    assert p.Jxy == pytest.approx(1331.4)
    assert p.Jyz == 0
    assert p.Jxz == 0
    assert p.rcgx == pytest.approx(-0.05 * 3.45)
    assert p.hEx == 0.0
    assert p.Tstab == pytest.approx(0.03)
    assert p.Xistab == pytest.approx(0.707)
    assert p.maxabsstab == pytest.approx(math.radians(25))
    assert p.maxabsdstab == pytest.approx(math.radians(60))
    assert p.Tail == pytest.approx(0.02)
    assert p.Xiail == pytest.approx(0.707)
    assert p.maxabsail == pytest.approx(math.radians(21.5))
    assert p.maxabsdail == pytest.approx(math.radians(80))
    assert p.Tdir == pytest.approx(0.03)
    assert p.Xidir == pytest.approx(0.707)
    assert p.maxabsdir == pytest.approx(math.radians(30))
    assert p.maxabsddir == pytest.approx(math.radians(120))
    assert p.lef == 0.0
    assert p.sb == 0.0
    assert p.g == pytest.approx(9.80665)
    assert p.Oy == 3000.0
    assert p.V == 120.0


def test_dynamic_pressure_at_3000m_120ms():
    p = default_parameters()
    # rho(3000) ~ 0.9091; q = 0.5*rho*120^2 ~= 6545
    assert p.q == pytest.approx(6545.5, rel=1e-3)
