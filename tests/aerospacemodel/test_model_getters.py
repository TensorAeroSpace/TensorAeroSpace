"""Tests for get_output and get_control getter methods across models.

Verifies that:
- get_output(to_deg=True) reads from output_history (not empty state_history).
- get_control(to_rad=True) reads from store_input (not store_states).
"""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747 import LongitudinalB747


def _make_b747_with_steps(n_steps=10, control_value=0.1):
    """Create a B747 model and run a few simulation steps."""
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    model = LongitudinalB747(x0=x0, number_time_steps=n_steps, dt=0.01)
    for _ in range(n_steps - 1):
        model.run_step(np.array([control_value]))
    return model


class TestGetOutputConversions:
    """get_output with to_deg/to_rad should read from output_history."""

    def test_get_output_to_deg_returns_data(self):
        model = _make_b747_with_steps()
        result = model.get_output("theta", to_deg=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_get_output_to_rad_returns_data(self):
        model = _make_b747_with_steps()
        result = model.get_output("theta", to_rad=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_get_output_to_deg_matches_manual_conversion(self):
        model = _make_b747_with_steps()
        raw = model.get_output("theta")
        deg = model.get_output("theta", to_deg=True)
        np.testing.assert_allclose(deg, np.rad2deg(raw), rtol=1e-12)

    def test_get_output_to_rad_matches_manual_conversion(self):
        model = _make_b747_with_steps()
        raw = model.get_output("theta")
        rad = model.get_output("theta", to_rad=True)
        np.testing.assert_allclose(rad, np.deg2rad(raw), rtol=1e-12)


class TestGetControlConversions:
    """get_control with to_rad should read from store_input, not store_states."""

    def test_get_control_to_rad_returns_data(self):
        model = _make_b747_with_steps(control_value=0.2)
        result = model.get_control("ele", to_rad=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_get_control_to_deg_returns_data(self):
        model = _make_b747_with_steps(control_value=0.2)
        result = model.get_control("ele", to_deg=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_get_control_to_rad_matches_manual_conversion(self):
        model = _make_b747_with_steps(control_value=0.2)
        raw = model.get_control("ele")
        rad = model.get_control("ele", to_rad=True)
        np.testing.assert_allclose(rad, np.deg2rad(raw), rtol=1e-12)

    def test_get_control_to_deg_matches_manual_conversion(self):
        model = _make_b747_with_steps(control_value=0.2)
        raw = model.get_control("ele")
        deg = model.get_control("ele", to_deg=True)
        np.testing.assert_allclose(deg, np.rad2deg(raw), rtol=1e-12)

    def test_get_control_to_rad_returns_control_not_state(self):
        """Regression: to_rad must read store_input, not store_states."""
        model = _make_b747_with_steps(control_value=0.3)
        ctrl_raw = model.get_control("ele")
        ctrl_rad = model.get_control("ele", to_rad=True)
        # If the bug existed, ctrl_rad would be deg2rad of store_states[0]
        # (the 'u' state), not store_input[0] (the 'ele' control).
        # Verify they are consistent with each other.
        np.testing.assert_allclose(ctrl_rad, np.deg2rad(ctrl_raw), rtol=1e-12)
