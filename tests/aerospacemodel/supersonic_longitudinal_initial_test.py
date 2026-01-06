import numpy as np
import pytest

from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal import initial


def test_set_initial_state_returns_list_and_updates():
    new_theta = np.deg2rad(2.0)
    out = initial.set_initial_state({"theta": new_theta})
    assert isinstance(out, list)
    # theta is first entry
    assert out[0][0] == pytest.approx(new_theta)


def test_set_initial_state_invalid_key_raises():
    with pytest.raises(Exception):
        initial.set_initial_state({"bad": 1.0})


def test_set_initial_state_no_matlab_dependency():
    # ensure module import/export works without matlab installed (function returns list)
    out = initial.set_initial_state({})
    assert isinstance(out, list)
