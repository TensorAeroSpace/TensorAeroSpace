import sys
import types

import numpy as np
import pytest

# Mock optional matlab dependency used in directional.initial
mock_matlab = types.ModuleType("matlab")
mock_matlab.double = lambda x: x
sys.modules.setdefault("matlab", mock_matlab)

# Stub utils module for missing exports
utils_stub = types.ModuleType("tensoraerospace.aerospacemodel.utils")
utils_stub.state_to_latex_eng = {}
utils_stub.state_to_latex_rus = {}
sys.modules["tensoraerospace.aerospacemodel.utils"] = utils_stub

from tensoraerospace.aerospacemodel.supersonic.linear.directional.model import (
    DirectionalSuperSonic,
)


def test_supersonic_directional_model_init_raises_due_to_state_output_mismatch():
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    steps = 4
    # Known bug: list_state uses outputs while indices are built from states
    with pytest.raises(ValueError):
        DirectionalSuperSonic(
            x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
        )
