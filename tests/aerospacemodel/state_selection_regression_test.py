"""State selection must preserve model metadata, requested order and Gym shapes."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747.linear.longitudinal import LongitudinalB747
from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.comsat import ComSat
from tensoraerospace.aerospacemodel.elv import ELVRocket
from tensoraerospace.aerospacemodel.f4c import LongitudinalF4C
from tensoraerospace.aerospacemodel.geosat import GeoSat
from tensoraerospace.aerospacemodel.lapan import LAPAN
from tensoraerospace.aerospacemodel.rocket import MissileModel
from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal.model import (
    LongitudinalSuperSonic,
)
from tensoraerospace.aerospacemodel.uav import LongitudinalUAV
from tensoraerospace.aerospacemodel.ultrastick import Ultrastick
from tensoraerospace.aerospacemodel.x15.linear.longitudinal import LongitudinalX15
from tensoraerospace.envs.b747 import LinearLongitudinalB747

MODELS = [
    (LongitudinalB747, 4, 1),
    (LongitudinalF4C, 4, 1),
    (LongitudinalUAV, 4, 1),
    (LongitudinalX15, 4, 1),
    (LongitudinalSuperSonic, 4, 1),
    (MissileModel, 4, 1),
    (LAPAN, 4, 1),
    (ELVRocket, 3, 1),
    (GeoSat, 3, 1),
    (ComSat, 3, 1),
    (Ultrastick, 5, 2),
]


@pytest.mark.parametrize(
    "cls,n_states,n_inputs", MODELS, ids=[item[0].__name__ for item in MODELS]
)
def test_selected_states_follow_requested_order(cls, n_states, n_inputs):
    initial = np.arange(1, n_states + 1, dtype=float).reshape(-1, 1) * 0.01
    full = cls(initial.copy(), number_time_steps=4)
    requested = [full.selected_states[-1], full.selected_states[0]]
    selected = cls(initial.copy(), number_time_steps=4, selected_state_output=requested)
    for _ in range(3):
        action = np.full(n_inputs, 0.01)
        expected = full.run_step(action)[[-1, 0]]
        actual = selected.run_step(action)
        assert actual.shape == ((2,) if cls is Ultrastick else (2, 1))
        np.testing.assert_allclose(actual, expected)
    assert selected.list_state == full.selected_states
    assert selected.control_list == full.selected_input


def test_base_selection_keeps_metadata_and_existing_history():
    model = ModelBase(np.zeros(2))
    model.list_state = ["alpha", "q"]
    model.control_list = ["ele"]
    model.state_history = {"alpha": np.array([0.1])}
    model.control_history = {"ele": np.array([0.2])}
    model.store_outputs = np.array([[0.3]])
    model.yt = np.array([0.4])
    model.ut = np.array([0.5])
    for requested, indices in [(["q"], [1]), (None, [0, 1])]:
        model._initialize_selected_state_index(requested, model.list_state)
        assert model.selected_state_index == indices
        assert model.list_state == ["alpha", "q"]
        assert model.control_list == ["ele"]
        np.testing.assert_array_equal(model.state_history["alpha"], [0.1])
        np.testing.assert_array_equal(model.control_history["ele"], [0.2])
        np.testing.assert_array_equal(model.store_outputs, [[0.3]])
        np.testing.assert_array_equal(model.yt, [0.4])
        np.testing.assert_array_equal(model.ut, [0.5])


def test_b747_subset_observations_match_space_and_physical_states():
    initial = np.array([0.01, 0.02, 0.03, 0.04])
    env = LinearLongitudinalB747(
        initial,
        reference_signal=np.zeros((1, 5)),
        number_time_steps=5,
        state_space=["theta", "q"],
        output_space=["theta", "q"],
    )
    try:
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
        np.testing.assert_allclose(observation, np.rad2deg(initial[[3, 2]]), rtol=1e-6)
        observation, *_ = env.step(np.array([1.0], dtype=np.float32))
        assert env.observation_space.contains(observation)
        np.testing.assert_allclose(
            observation,
            np.rad2deg(env.model.xt.reshape(-1)[[3, 2]]),
            rtol=1e-6,
        )
    finally:
        env.close()
