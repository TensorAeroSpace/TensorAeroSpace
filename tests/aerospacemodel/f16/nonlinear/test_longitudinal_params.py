import math

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.python.longitudinal.params import (
    F16LongParameters,
    default_parameters,
)


def test_default_matches_matlab_constants():
    p = default_parameters()
    assert p.m == pytest.approx(9295.44)
    assert p.S == pytest.approx(27.87)
    assert p.bA == pytest.approx(3.45)
    assert p.Jz == pytest.approx(75673.6)
    assert p.rcgx == pytest.approx(-0.05 * 3.45)
    assert p.Tstab == pytest.approx(0.03)
    assert p.Xistab == pytest.approx(0.707)
    assert p.maxabsstab == pytest.approx(math.radians(25))
    assert p.maxabsdstab == pytest.approx(math.radians(60))
    assert p.lef == 0.0
    assert p.sb == 0.0
    assert p.g == pytest.approx(9.80665)
    assert p.Oy == 3000.0
    assert p.V == 150.0


def test_dynamic_pressure_matches_isa_atmosphere_at_3000m():
    p = default_parameters()
    # rho(3000m) ~ 0.9091, V=150, q ~ 10227 Pa
    assert p.q == pytest.approx(10227.6, rel=1e-3)


def test_can_override_altitude_and_velocity():
    p = F16LongParameters(Oy=0.0, V=200.0)
    # at sea level rho=1.225, q = 0.5 * 1.225 * 200^2 = 24500
    assert p.q == pytest.approx(24500.0, rel=1e-3)
