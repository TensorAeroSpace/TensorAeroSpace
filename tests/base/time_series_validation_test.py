import numpy as np
import pytest

from tensoraerospace.utils import validate_time_series_alignment


def test_validate_time_series_alignment_success():
    length = validate_time_series_alignment(
        [0, 1, 2],
        np.array([10.0, 11.0, 12.0]),
        range(3),
    )
    assert length == 3


def test_validate_time_series_alignment_raises_on_mismatch():
    with pytest.raises(ValueError, match="Mismatched time series lengths"):
        validate_time_series_alignment([0, 1], np.array([1, 2, 3]))


def test_validate_time_series_alignment_requires_at_least_one_series():
    with pytest.raises(ValueError, match="At least one sequence"):
        validate_time_series_alignment()
