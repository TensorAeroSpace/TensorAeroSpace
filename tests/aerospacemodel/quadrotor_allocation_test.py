"""Tests for the X-config quadrotor allocation matrix."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.quadrotor import (
    XConfigAllocator,
    default_allocator,
)


def _allocator() -> XConfigAllocator:
    return default_allocator()


def test_mix_unmix_roundtrip_identity():
    """unmix(mix(ω²)) == ω² for any non-degenerate input."""
    a = _allocator()
    rng = np.random.default_rng(0)
    for _ in range(20):
        omega2 = rng.uniform(0, 1e6, size=4)
        v = a.mix(omega2)
        recovered = a.unmix(v)
        assert np.allclose(recovered, omega2, rtol=1e-12)


def test_unmix_mix_roundtrip_identity():
    """mix(unmix(v)) == v for any **realisable** v.

    Realisable: positive T plus small torques such that all four
    omega² stay non-negative (otherwise mix() rightly refuses the input).
    """
    a = _allocator()
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = rng.uniform(5.0, 30.0)  # N — well above hover thrust
        # Limit τ magnitudes so each rotor stays positive.
        # Crude bound: |τ_x|, |τ_y| < T·a, |τ_z| · a/k_M·k_T < T·... — use small fractions.
        tau_x = rng.uniform(-0.1, 0.1)
        tau_y = rng.uniform(-0.1, 0.1)
        tau_z = rng.uniform(-0.05, 0.05)
        v = np.array([T, tau_x, tau_y, tau_z])
        omega2 = a.unmix(v)
        if np.any(omega2 < 0):
            continue  # skip non-realisable random sample
        recovered = a.mix(omega2)
        assert np.allclose(recovered, v, rtol=1e-12)


def test_hover_is_symmetric():
    """Pure thrust → all 4 rotors at the same speed²."""
    a = _allocator()
    T_only = np.array([10.0, 0.0, 0.0, 0.0])
    omega2 = a.unmix(T_only)
    assert np.allclose(omega2, omega2[0])  # all four equal
    assert omega2[0] > 0


def test_pure_roll_torque_excites_paired_rotors():
    """+τ_x with T=0, τ_y=τ_z=0 → motors 1+4 down, 2+3 up (or vice versa)."""
    a = _allocator()
    roll_only = np.array([0.0, 1.0, 0.0, 0.0])
    omega2 = a.unmix(roll_only)
    # by the mixing matrix structure (column signs for τ_x row):
    #   M1: -, M2: +, M3: +, M4: -
    # so flipping for unmix: ω²₁ < 0, ω²₂ > 0, ω²₃ > 0, ω²₄ < 0
    assert omega2[0] < 0 and omega2[3] < 0
    assert omega2[1] > 0 and omega2[2] > 0
    # Symmetric pairs: (1, 4) equal magnitude, (2, 3) equal magnitude
    assert np.isclose(omega2[0], omega2[3])
    assert np.isclose(omega2[1], omega2[2])


def test_pure_pitch_torque_excites_paired_rotors():
    """+τ_y → front motors (1,3) up, rear motors (2,4) down."""
    a = _allocator()
    pitch_only = np.array([0.0, 0.0, 1.0, 0.0])
    omega2 = a.unmix(pitch_only)
    # tau_y row: M1=+, M2=-, M3=+, M4=-, so unmix sign flip:
    #   ω²₁ > 0, ω²₂ < 0, ω²₃ > 0, ω²₄ < 0
    assert omega2[0] > 0 and omega2[2] > 0
    assert omega2[1] < 0 and omega2[3] < 0
    # Symmetric: (1, 3) equal, (2, 4) equal
    assert np.isclose(omega2[0], omega2[2])
    assert np.isclose(omega2[1], omega2[3])


def test_pure_yaw_torque_excites_ccw_pair():
    """+τ_z → CCW pair (M1, M2) up, CW pair (M3, M4) down."""
    a = _allocator()
    yaw_only = np.array([0.0, 0.0, 0.0, 1.0])
    omega2 = a.unmix(yaw_only)
    assert omega2[0] > 0 and omega2[1] > 0
    assert omega2[2] < 0 and omega2[3] < 0
    assert np.isclose(omega2[0], omega2[1])
    assert np.isclose(omega2[2], omega2[3])


def test_mix_thrust_scales_linearly_with_total_omega_squared():
    """T = k_T · Σ ω_i² (linear in total)."""
    a = _allocator()
    omega2 = np.full(4, 1e5)
    v = a.mix(omega2)
    expected_T = a.k_T * omega2.sum()
    assert np.isclose(v[0], expected_T)
    # equal-speed hover gives zero torques
    assert np.allclose(v[1:], 0.0)


def test_saturate_clips_to_bounds():
    a = _allocator()
    omega2 = np.array([-100.0, 50.0, 1e10, 200.0])
    out = a.saturate(omega2, omega_min=0.0, omega_max=1000.0)
    assert out[0] == 0.0
    assert out[1] == 50.0
    assert out[2] == 1000.0**2
    assert out[3] == 200.0


def test_saturate_with_min_floor():
    """Non-zero omega_min — useful for keeping motors spinning at idle."""
    a = _allocator()
    omega2 = np.array([0.0, 100.0, 1e10, 50.0])
    out = a.saturate(omega2, omega_min=10.0, omega_max=1000.0)
    assert out[0] == 100.0  # omega_min² = 100
    assert out[1] == 100.0
    assert out[3] == 100.0


def test_negative_kT_raises():
    with pytest.raises(ValueError, match="k_T must be positive"):
        XConfigAllocator(k_T=-1e-6)


def test_invalid_omega_squared_shape_raises():
    a = _allocator()
    with pytest.raises(ValueError, match="must have 4 elements"):
        a.mix(np.array([1.0, 2.0]))


def test_negative_omega_squared_raises():
    a = _allocator()
    with pytest.raises(ValueError, match="non-negative"):
        a.mix(np.array([-1.0, 1.0, 1.0, 1.0]))


def test_invalid_saturation_bounds_raises():
    a = _allocator()
    with pytest.raises(ValueError, match="omega_max must be"):
        a.saturate(np.zeros(4), omega_min=100.0, omega_max=50.0)


def test_default_allocator_returns_well_formed_instance():
    a = default_allocator()
    assert a.k_T > 0 and a.k_M > 0 and a.arm_length > 0
    # Allocation matrix should be 4×4 and invertible
    assert a.matrix.shape == (4, 4)
    assert a.matrix_inverse.shape == (4, 4)


def test_round_trip_with_saturated_input():
    """If we unmix → saturate → mix, the actual virtual delivered may
    differ from the request (this is the "lost authority" scenario).
    Verify it doesn't crash and the result is at least sensible.
    """
    a = _allocator()
    # Request more thrust than 4 rotors at omega_max can deliver
    huge = np.array([1e5, 0.0, 0.0, 0.0])
    omega2 = a.unmix(huge)
    omega2_sat = a.saturate(omega2, omega_min=0.0, omega_max=1000.0)
    delivered = a.mix(omega2_sat)
    # Should still be a finite vector
    assert np.all(np.isfinite(delivered))
    # Saturated thrust should be less than requested
    assert delivered[0] < huge[0]
