"""Callers must not be able to rewrite a plant's trajectory through returned arrays."""

import importlib

import numpy as np
import pytest

CASES = [
    ("b747.nonlinear", "NonlinearB747", 12, 4),
    ("b737.nonlinear", "NonlinearB737", 12, 4),
    ("x15.nonlinear", "NonlinearX15", 13, 4),
    ("skywalker_x8.nonlinear", "NonlinearSkywalkerX8", 12, 3),
    ("aai_shadow.nonlinear", "NonlinearAAIShadow", 12, 4),
    ("quadrotor.nonlinear", "NonlinearQuadrotor", 12, 4),
    ("f16.nonlinear.longitudinal", "LongitudinalF16", 4, 1),
    ("f16.nonlinear.angular", "AngularF16", 14, 3),
]


def build(case):
    package, cls, size, actions = case
    initial = np.zeros(size)
    if package not in ("quadrotor.nonlinear",) and size in (12, 13):
        initial[0], initial[11] = 100.0, -1000.0
    if size == 13:
        initial[12] = 1000.0
    module = importlib.import_module(f"tensoraerospace.aerospacemodel.{package}.model")
    return getattr(module, cls)(initial, dt=0.001), initial, np.zeros(actions)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
def test_initial_state_is_owned_by_model(case):
    model, initial, _ = build(case)
    expected = initial.copy()
    initial[:] = 123.0
    np.testing.assert_array_equal(model.current_state, expected)
    model.restart()
    np.testing.assert_array_equal(model.current_state, expected)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("accessor", ["run_step", "current_state"])
def test_observation_mutation_does_not_change_future_trajectory(case, accessor):
    model, _, action = build(case)
    reference, _, _ = build(case)
    actual = model.run_step(action)
    reference.run_step(action)
    exposed = actual if accessor == "run_step" else model.current_state
    exposed[...] = 123.0
    np.testing.assert_array_equal(model.current_state, reference.current_state)
    np.testing.assert_array_equal(model.run_step(action), reference.run_step(action))
