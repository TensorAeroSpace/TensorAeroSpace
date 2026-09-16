"""A restarted plant must reproduce a fresh trajectory and live histories."""

import importlib

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747.linear.longitudinal import LongitudinalB747
from tensoraerospace.aerospacemodel.comsat import ComSat
from tensoraerospace.aerospacemodel.f16.linear.angular.model import AngularF16
from tensoraerospace.aerospacemodel.ultrastick import Ultrastick

NONLINEAR = [
    ("b747", "NonlinearB747", 12, 4, "b747_ode_6dof"),
    ("b737", "NonlinearB737", 12, 4, "b737_ode_6dof"),
    ("x15", "NonlinearX15", 13, 4, "x15_ode_6dof"),
    ("skywalker_x8", "NonlinearSkywalkerX8", 12, 3, "x8_ode_6dof"),
    ("aai_shadow", "NonlinearAAIShadow", 12, 4, "shadow_ode_6dof"),
    ("quadrotor", "NonlinearQuadrotor", 12, 4, "quadrotor_ode_6dof"),
    ("f16", "LongitudinalF16", 4, 1, "f16_ode_long"),
    ("f16", "AngularF16", 14, 3, "f16_ode_6dof"),
]


def make_nonlinear(case, **kwargs):
    package, name, n_states, n_actions, _ = case
    suffix = ".longitudinal" if n_states == 4 else ".angular" if n_states == 14 else ""
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{package}.nonlinear{suffix}.model"
    )
    initial = np.zeros(n_states)
    if n_states in (12, 13) and package != "quadrotor":
        initial[0], initial[11] = 100.0, -1000.0
    if n_states == 13:
        initial[12] = 1000.0
    return (
        getattr(module, name)(initial, dt=0.001, integrator="rk4", **kwargs),
        module,
        np.zeros(n_actions),
    )


@pytest.mark.parametrize("case", NONLINEAR, ids=[case[1] for case in NONLINEAR])
def test_nonlinear_restart_preserves_metadata_and_reproduces_trajectory(case):
    model, _, action = make_nonlinear(case)
    states, controls = model.list_state.copy(), model.control_list.copy()
    initial = model.current_state.copy()
    first = [model.run_step(action).copy() for _ in range(3)]
    model.get_state(states[0])
    model.get_control(controls[0])
    model.restart()
    assert model.list_state == states
    assert model.control_list == controls
    np.testing.assert_array_equal(model.current_state, initial)
    assert model.get_control(controls[0]).size == 0
    for expected in first:
        np.testing.assert_array_equal(model.run_step(action), expected)
    assert model.get_state(states[0]).size == 3
    assert model.get_control(controls[0]).size == 3


@pytest.mark.parametrize(
    "cls,n_states,n_actions",
    [
        (LongitudinalB747, 4, 1),
        (AngularF16, 11, 3),
        (ComSat, 3, 1),
        (Ultrastick, 5, 2),
    ],
)
def test_linear_restart_restores_state_time_and_preallocated_buffers(
    cls, n_states, n_actions
):
    model = cls(np.full((n_states, 1), 0.01), 4)
    action = np.full((n_actions, 1), 0.01)
    first = [model.run_step(action).copy() for _ in range(3)]
    model.restart()
    assert model.time_step == 0
    np.testing.assert_array_equal(model.xt, model.x0)
    assert not model.store_input.any()
    assert not model.store_outputs.any()
    np.testing.assert_array_equal(
        model.store_states[:, 0], np.asarray(model.x0).ravel()
    )
    assert not model.store_states[:, 1:].any()
    for expected in first:
        np.testing.assert_array_equal(model.run_step(action), expected)


@pytest.mark.parametrize("case", NONLINEAR, ids=[case[1] for case in NONLINEAR])
def test_history_queries_refresh_after_more_steps(case):
    model, _, action = make_nonlinear(case)
    state_name, control_name = model.list_state[0], model.control_list[0]
    model.run_step(action)
    assert model.get_state(state_name).size == 1
    assert model.get_control(control_name).size == 1
    model.run_step(action)
    model.run_step(action)
    expected_states = np.asarray(model.x_history).reshape(4, -1)[:3, 0]
    np.testing.assert_array_equal(model.get_state(state_name), expected_states)
    assert model.get_control(control_name).size == 3


@pytest.mark.parametrize("case", NONLINEAR, ids=[case[1] for case in NONLINEAR])
def test_integrator_starts_at_t0_and_advances_by_dt(case, monkeypatch):
    model, module, action = make_nonlinear(case, t0=2.0)
    initial = model.current_state.copy()

    def time_dependent_rhs(state, control, time, params):
        derivative = np.zeros_like(state)
        derivative[0] = time
        return derivative

    monkeypatch.setattr(module, case[4], time_dependent_rhs)
    for step in range(1, 4):
        actual = model.run_step(action).ravel()
        elapsed = step * model.dt
        expected = initial[0] + model.t0 * elapsed + 0.5 * elapsed**2
        assert actual[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("case", NONLINEAR, ids=[case[1] for case in NONLINEAR])
def test_control_history_owns_each_applied_command(case):
    model, _, action = make_nonlinear(case)
    action[0] = 0.01
    model.run_step(action)
    action[0] = 0.02
    model.run_step(action)
    np.testing.assert_array_equal(
        model.get_control(model.control_list[0]), [0.01, 0.02]
    )
