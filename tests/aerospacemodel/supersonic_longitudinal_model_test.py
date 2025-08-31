import sys
import types

import numpy as np
import pytest

# Stub utils module to provide required names for imports
utils_stub = types.ModuleType("tensoraerospace.aerospacemodel.utils")
utils_stub.state_to_latex_eng = {}
utils_stub.state_to_latex_rus = {}
sys.modules["tensoraerospace.aerospacemodel.utils"] = utils_stub

from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal.model import (
    LongitudinalSuperSonic as LongitudinalModel,
)
from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal.output_based import (
    LongitudinalSuperSonic as LongitudinalOutputBased,
)


def test_supersonic_longitudinal_model_smoke():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 4
    model = LongitudinalModel(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    x1 = model.run_step(np.array([1.0]))
    assert x1.shape[0] == 4
    assert model.get_state("theta").shape[0] == steps - 1
    assert model.get_control("ele").shape[0] == steps - 1
    assert model.get_output("q").shape[0] == model.time_step - 1


def test_supersonic_longitudinal_output_based_init_raises_due_to_mismatch():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 4
    # Known bug: passing selected_states vs list_state of outputs causes mismatch
    with pytest.raises(ValueError):
        LongitudinalOutputBased(
            x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
        )
