"""Regression tests for the derivative used by AIDI."""

import numpy as np
import pytest

from tensoraerospace.agent.aidi.sensor_filter import LowPassDerivative


def test_low_pass_derivative_on_ramp_returns_slope_eventually():
    """For x(t) = a·t + b, the derivative should converge to a."""
    d = LowPassDerivative(n=1, dt=0.01, cutoff_hz=20.0)
    out = None
    for k in range(200):
        out = d.step(np.array([2.0 * k * 0.01 + 1.0]))
    # After warm-up the filter has tracked the constant slope.
    assert abs(float(out[0]) - 2.0) < 1e-2


def test_low_pass_derivative_first_call_returns_zero():
    d = LowPassDerivative(n=3, dt=0.01, cutoff_hz=10.0)
    out = d.step(np.array([5.0, -1.0, 2.0]))
    np.testing.assert_allclose(out, np.zeros(3))


def test_low_pass_derivative_reset_clears_state():
    d = LowPassDerivative(n=1, dt=0.01, cutoff_hz=50.0)
    for _ in range(10):
        d.step(np.array([1.0]))
    d.reset()
    out = d.step(np.array([1.0]))
    np.testing.assert_allclose(out, np.zeros(1))


def test_low_pass_derivative_validates_inputs():
    with pytest.raises(ValueError, match="dt"):
        LowPassDerivative(n=1, dt=0.0)
    with pytest.raises(ValueError, match="cutoff_hz"):
        LowPassDerivative(n=1, dt=0.01, cutoff_hz=-1)
    d = LowPassDerivative(n=2, dt=0.01)
    with pytest.raises(ValueError, match="length"):
        d.step(np.array([1.0]))
