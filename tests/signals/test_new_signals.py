"""Tests for new signal generation functions."""

import numpy as np
import pytest

from tensoraerospace.signals.standard import (
    chirp,
    damped_sinusoid,
    doublet,
    exponential,
    gaussian_pulse,
    multi_step,
    multisine,
    pulse,
    ramp,
    sawtooth,
    square_wave,
    triangular_wave,
)


def test_ramp():
    """Test ramp signal generation."""
    # Normal input data
    tp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    slope = 1.0
    time_start = 1.0
    result = ramp(tp, slope, time_start)
    expected = slope * np.maximum(tp - time_start, 0)
    assert np.allclose(result, expected)

    # Different slope
    slope = 2.5
    time_start = 0.5
    result = ramp(tp, slope, time_start)
    expected = slope * np.maximum(tp - time_start, 0)
    assert np.allclose(result, expected)

    # Zero slope
    slope = 0.0
    result = ramp(tp, slope, time_start)
    assert np.all(result == 0.0)

    # Negative slope
    slope = -1.5
    time_start = 2.0
    result = ramp(tp, slope, time_start)
    expected = slope * np.maximum(tp - time_start, 0)
    assert np.allclose(result, expected)


def test_pulse():
    """Test pulse signal generation."""
    # Normal input data
    tp = np.linspace(0, 10, 101)
    amplitude = 5.0
    time_start = 3.0
    width = 2.0
    result = pulse(tp, amplitude, time_start, width)

    # Check pulse is zero before start
    assert np.all(result[tp < time_start] == 0)
    # Check pulse has correct amplitude during pulse
    pulse_region = (tp >= time_start) & (tp < time_start + width)
    assert np.all(result[pulse_region] == amplitude)
    # Check pulse is zero after end
    assert np.all(result[tp >= time_start + width] == 0)

    # Edge case: zero amplitude
    result = pulse(tp, 0.0, time_start, width)
    assert np.all(result == 0)

    # Edge case: zero width
    result = pulse(tp, amplitude, time_start, 0.0)
    assert np.all(result == 0)


def test_square_wave():
    """Test square wave signal generation."""
    # Normal input data
    tp = np.linspace(0, 10, 1000)
    frequency = 1.0
    amplitude = 3.0
    duty_cycle = 0.5
    result = square_wave(tp, frequency, amplitude, duty_cycle)

    # Check amplitude values
    assert np.all((result == 0) | (result == amplitude))
    # Check max and min values
    assert result.max() == amplitude
    assert result.min() == 0

    # Different duty cycle
    duty_cycle = 0.25
    result = square_wave(tp, frequency, amplitude, duty_cycle)
    assert np.all((result == 0) | (result == amplitude))

    # Zero frequency
    result = square_wave(tp, 0.0, amplitude, duty_cycle)
    assert np.all(result == amplitude)


def test_sawtooth():
    """Test sawtooth signal generation."""
    # Normal input data
    tp = np.linspace(0, 2, 1000)
    frequency = 1.0
    amplitude = 3.0
    result = sawtooth(tp, frequency, amplitude)

    # Check range
    assert result.max() <= amplitude
    assert result.min() >= -amplitude

    # Zero frequency
    result = sawtooth(tp, 0.0, amplitude)
    assert np.all(result == -amplitude)

    # Zero amplitude
    result = sawtooth(tp, frequency, 0.0)
    assert np.all(result == 0)


def test_triangular_wave():
    """Test triangular wave signal generation."""
    # Normal input data
    tp = np.linspace(0, 2, 1000)
    frequency = 1.0
    amplitude = 4.0
    result = triangular_wave(tp, frequency, amplitude)

    # Check range
    assert result.max() <= amplitude
    assert result.min() >= -amplitude

    # Check symmetry property
    phase = (tp * frequency) % 1.0
    expected = amplitude * (2 * np.abs(2 * phase - 1) - 1)
    assert np.allclose(result, expected)


def test_chirp():
    """Test chirp signal generation."""
    # Linear chirp
    tp = np.linspace(0, 10, 1000)
    f0 = 0.1
    f1 = 2.0
    amplitude = 2.0
    result = chirp(tp, f0, f1, amplitude, method="linear")

    # Check amplitude bounds
    assert np.abs(result).max() <= amplitude
    assert np.abs(result).min() >= 0

    # Exponential chirp
    result = chirp(tp, f0, f1, amplitude, method="exponential")
    assert np.abs(result).max() <= amplitude

    # Empty array
    result = chirp(np.array([]), f0, f1, amplitude)
    assert len(result) == 0

    # Invalid method
    with pytest.raises(ValueError):
        chirp(tp, f0, f1, amplitude, method="invalid")


def test_doublet():
    """Test doublet signal generation."""
    # Normal input data
    tp = np.linspace(0, 10, 1001)
    amplitude = 10.0
    time_start = 3.0
    width = 1.0
    result = doublet(tp, amplitude, time_start, width)

    # Check signal is zero before start
    assert np.all(result[tp < time_start] == 0)

    # Check positive pulse
    positive_region = (tp >= time_start) & (tp < time_start + width)
    assert np.all(result[positive_region] == amplitude)

    # Check negative pulse
    negative_region = (tp >= time_start + width) & (tp < time_start + 2 * width)
    assert np.all(result[negative_region] == -amplitude)

    # Check signal is zero after doublet
    assert np.all(result[tp >= time_start + 2 * width] == 0)

    # Check integral is approximately zero (property of doublet)
    dt = tp[1] - tp[0]
    integral = np.sum(result) * dt
    assert np.abs(integral) < 0.1


def test_multi_step():
    """Test multi-step signal generation."""
    # Normal input data
    tp = np.linspace(0, 20, 2001)
    step_times = [2, 5, 8, 12, 16]
    step_values = [1, 2, -1, 3, -2]
    result = multi_step(tp, step_times, step_values)

    # Check initial value
    assert result[0] == 0

    # Check cumulative sum property
    for i, (time, value) in enumerate(zip(step_times, step_values)):
        idx = np.where(tp >= time)[0][0]
        expected_value = sum(step_values[: i + 1])
        assert np.isclose(result[idx], expected_value)

    # Mismatched lengths
    with pytest.raises(ValueError):
        multi_step(tp, [1, 2, 3], [1, 2])


def test_exponential():
    """Test exponential signal generation."""
    # Normal input data
    tp = np.linspace(0, 10, 1001)
    amplitude = 10.0
    time_constant = 2.0
    time_start = 3.0
    result = exponential(tp, amplitude, time_constant, time_start)

    # Check signal is zero before start
    assert np.all(result[tp < time_start] == 0)

    # Check signal approaches amplitude
    assert result[-1] < amplitude
    assert result[-1] > 0.95 * amplitude  # Should be close to final value

    # Check time constant property (63.2% at one time constant)
    idx_tau = np.argmin(np.abs(tp - (time_start + time_constant)))
    expected_at_tau = amplitude * (1 - np.exp(-1))
    assert np.isclose(result[idx_tau], expected_at_tau, rtol=0.05)

    # Zero time constant should give step-like behavior
    result = exponential(tp, amplitude, 1e-10, time_start)
    # Exclude the exact time_start point as it might be transitional
    assert np.all(result[tp > time_start] > 0.99 * amplitude)


def test_gaussian_pulse():
    """Test Gaussian pulse signal generation."""
    # Normal input data
    tp = np.linspace(0, 20, 2001)
    amplitude = 8.0
    center = 10.0
    width = 1.5
    result = gaussian_pulse(tp, amplitude, center, width)

    # Check maximum at center
    center_idx = np.argmin(np.abs(tp - center))
    assert result[center_idx] == pytest.approx(amplitude, rel=0.01)

    # Check symmetry around center
    left_idx = center_idx - 100
    right_idx = center_idx + 100
    if left_idx >= 0 and right_idx < len(result):
        assert np.isclose(result[left_idx], result[right_idx], rtol=0.01)

    # Check signal decreases away from center
    assert result[0] < result[center_idx]
    assert result[-1] < result[center_idx]

    # Check amplitude
    assert result.max() == pytest.approx(amplitude, rel=0.01)


def test_multisine():
    """Test multi-sine signal generation."""
    # Normal input data
    tp = np.linspace(0, 10, 1000)
    frequencies = [0.5, 1.0, 2.0]
    amplitudes = [2.0, 1.5, 1.0]
    phases = [0, np.pi / 4, np.pi / 2]
    result = multisine(tp, frequencies, amplitudes, phases)

    # Check it's a sum of sinusoids
    expected = np.zeros_like(tp)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        expected += amp * np.sin(2 * np.pi * freq * tp + phase)
    assert np.allclose(result, expected)

    # Default phases (all zeros)
    result_no_phase = multisine(tp, frequencies, amplitudes)
    expected_no_phase = np.zeros_like(tp)
    for freq, amp in zip(frequencies, amplitudes):
        expected_no_phase += amp * np.sin(2 * np.pi * freq * tp)
    assert np.allclose(result_no_phase, expected_no_phase)

    # Mismatched lengths
    with pytest.raises(ValueError):
        multisine(tp, [0.5, 1.0], [2.0, 1.5, 1.0])

    with pytest.raises(ValueError):
        multisine(tp, frequencies, amplitudes, [0, np.pi / 4])


def test_chirp_equal_frequencies_exponential():
    """Test chirp with f0 == f1 (exponential) returns valid constant-frequency sinusoid."""
    tp = np.linspace(0, 10, 1000)
    f0 = 2.0
    result = chirp(tp, f0=f0, f1=f0, amplitude=3.0, method="exponential")
    # Should not raise and should return a valid array
    assert result.shape == tp.shape
    assert np.all(np.isfinite(result))
    # Should match a constant-frequency sinusoid
    expected = 3.0 * np.sin(2 * np.pi * f0 * tp)
    assert np.allclose(result, expected)


def test_chirp_f0_non_positive_raises():
    """Test chirp raises ValueError when f0 <= 0."""
    tp = np.linspace(0, 10, 100)
    with pytest.raises(ValueError, match="f0 must be positive"):
        chirp(tp, f0=0.0, f1=1.0)
    with pytest.raises(ValueError, match="f0 must be positive"):
        chirp(tp, f0=-1.0, f1=1.0)


def test_damped_sinusoid():
    """Test damped sinusoid signal generation."""
    # Normal input data
    tp = np.linspace(0, 20, 2001)
    frequency = 1.0
    amplitude = 5.0
    damping = 0.3
    time_start = 2.0
    result = damped_sinusoid(tp, frequency, amplitude, damping, time_start)

    # Check signal is zero before start
    assert np.all(result[tp < time_start] == 0)

    # Check amplitude decreases with time (envelope decay)
    t_after_start = tp[tp >= time_start]
    result_after_start = result[tp >= time_start]

    # Check envelope at different times
    if len(t_after_start) > 100:
        early_max = np.abs(result_after_start[:100]).max()
        late_max = np.abs(result_after_start[-100:]).max()
        assert late_max < early_max  # Signal should decay

    # Check oscillation frequency (approximate)
    # Find peaks
    sign_changes = np.diff(np.sign(np.diff(result_after_start)))
    peaks = np.where(sign_changes < 0)[0]
    if len(peaks) > 1:
        # Average period in samples
        avg_period = np.mean(np.diff(peaks))
        dt = tp[1] - tp[0]
        measured_freq = 1.0 / (avg_period * dt)
        assert np.isclose(measured_freq, frequency, rtol=0.1)

    # Zero damping should give pure sinusoid
    result_no_damp = damped_sinusoid(tp, frequency, amplitude, 0.0, time_start)
    t_late = tp[-100:]
    result_late = result_no_damp[-100:]
    # Should still have full amplitude oscillations
    assert np.abs(result_late).max() > 0.9 * amplitude
