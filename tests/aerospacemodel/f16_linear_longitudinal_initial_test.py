import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.linear.longitudinal import initial


def test_set_initial_state_updates_and_returns_list():
    new_alpha = np.deg2rad(3.0)
    out = initial.set_initial_state({"alpha": new_alpha})
    assert isinstance(out, list)
    # alpha is second entry in list
    assert out[1][0] == pytest.approx(new_alpha)


def test_set_initial_state_invalid_key():
    with pytest.raises(Exception):
        initial.set_initial_state({"bad": 1.0})
