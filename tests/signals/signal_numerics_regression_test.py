"""Signal amplitudes and frequency sweeps must not depend on time dtype/origin."""

import numpy as np
import pytest
from scipy.signal import chirp as scipy_chirp

from tensoraerospace.signals.standard import chirp, constant_line, multi_step, multisine


@pytest.mark.parametrize(
    "generator",
    [
        lambda t: constant_line(t, 0.25),
        lambda t: multi_step(t, [1, 3], [0.25, -0.5]),
        lambda t: multisine(t, [0.125, 0.25], [0.25, 0.75]),
    ],
)
def test_fractional_signal_on_integer_time_grid(generator):
    times = np.arange(8)
    np.testing.assert_allclose(generator(times), generator(times.astype(float)))


@pytest.mark.parametrize(
    "method,reference_method", [("linear", "linear"), ("exponential", "logarithmic")]
)
@pytest.mark.parametrize("origin", [-20.0, 0.0, 10.0])
def test_chirp_frequency_sweep_uses_elapsed_time(method, reference_method, origin):
    elapsed = np.linspace(0, 2, 129)
    actual = chirp(elapsed + origin, f0=0.2, f1=3.0, amplitude=0.7, method=method)
    expected = 0.7 * scipy_chirp(
        elapsed, f0=0.2, f1=3.0, t1=2.0, method=reference_method, phi=-90
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_near_constant_exponential_chirp_does_not_collapse_to_zero():
    times = np.linspace(0, 100, 1001)
    actual = chirp(times, f0=1.0, f1=1.0 + 1e-14, method="exponential")
    np.testing.assert_allclose(actual, np.sin(2 * np.pi * times), atol=5e-11)


@pytest.mark.parametrize("end_frequency", [0.0, -1.0])
def test_exponential_chirp_rejects_nonpositive_end_frequency(end_frequency):
    with pytest.raises(ValueError, match="f1"):
        chirp(np.linspace(0, 2, 10), f1=end_frequency, method="exponential")
