import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.utils import (
    control2dict,
    output2dict,
    state2dict,
)


def test_state2dict_shapes_and_keys():
    state = [
        [0.0, 1.0, 2.0],
        [10.0, 11.0, 12.0],
    ]
    keys = ["a", "b"]
    out = state2dict(state, keys)
    assert set(out.keys()) == set(keys)
    assert out["a"].shape[0] == 2
    assert out["b"].shape[0] == 2


def test_control2dict_shapes_and_keys():
    control = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]
    keys = ["u1", "u2"]
    out = control2dict(control, keys)
    assert set(out.keys()) == set(keys)
    for k in keys:
        assert out[k].shape[0] == 2


def test_output2dict_mapping():
    output = np.array([1.0, 2.0, 3.0])
    labels = ["y1", "y2", "y3"]
    out = output2dict(output, labels)
    assert out == {"y1": 1.0, "y2": 2.0, "y3": 3.0}
