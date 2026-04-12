"""General purpose utilities for TensorAeroSpace library.

This module contains general purpose helper functions used in various parts
of the library. Includes functions for working with time intervals, data
conversion and other useful utilities.

Main functions:
    - generate_time_period: Generate time period with specified sampling frequency
    - convert_tp_to_sec_tp: Convert time interval to seconds
"""

from typing import Sequence

import numpy as np


def generate_time_period(tn: float = 20.0, dt: float = 0.01) -> np.ndarray:
    """Generate time period in seconds with sampling frequency dt.

    Returns an array of actual time values in seconds, from 0 to ``tn``
    inclusive, with step ``dt``.

    Args:
        tn (float): Simulation duration in seconds. Defaults to 20.0.
        dt (float): Sampling period in seconds. Defaults to 0.01.

    Returns:
        np.ndarray: Time period in seconds ``[0, dt, 2*dt, ..., tn]``.
    """
    t0 = 0.0
    # Returns actual time in seconds: [0, dt, 2*dt, ..., tn]
    n_steps = int(round((tn - t0) / dt)) + 1
    return np.linspace(t0, tn, n_steps)


def convert_tp_to_sec_tp(tp, dt=None) -> np.ndarray:
    """Backward-compat identity function.

    Since :func:`generate_time_period` now returns actual seconds, this
    function is effectively a no-op kept for API compatibility.

    Args:
        tp: Time period (already in seconds).
        dt: Ignored. Kept for backward-compatible signature.

    Returns:
        np.ndarray: ``tp`` converted to a numpy array.
    """
    return np.asarray(tp)


def validate_time_series_alignment(*series: Sequence) -> int:
    """Ensure that every provided time series has the same length.

    Args:
        *series: Arbitrary number of sequences (lists, numpy arrays, tensors).

    Returns:
        int: Common length of all sequences.

    Raises:
        TypeError: If one of the arguments does not define ``__len__``.
        ValueError: If at least two sequences have different lengths.
    """

    if not series:
        raise ValueError("At least one sequence is required for validation.")

    lengths = []
    for idx, seq in enumerate(series):
        try:
            length = len(seq)
        except TypeError as exc:  # pragma: no cover - defensive branch
            raise TypeError(f"Argument #{idx} does not define __len__().") from exc
        lengths.append(length)

    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            "Mismatched time series lengths detected: "
            + ", ".join(f"seq{idx}={length}" for idx, length in enumerate(lengths))
        )

    return lengths[0]
