"""Nonlinear F-16 commands and actual actuator states obey physical limits."""

import numpy as np
import pytest
from scipy.linalg import expm

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    ControlFailure,
    DamageState,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

CASES = [
    ("longitudinal", 0, 2, "stab"),
    ("angular", 0, 8, "stab"),
    ("angular", 1, 10, "ail"),
    ("angular", 2, 12, "dir"),
]


def make_model(kind, dt=0.005, integrator="rk4"):
    cls, size = (AngularF16, 14) if kind == "angular" else (LongitudinalF16, 4)
    return cls(np.zeros(size), dt=dt, integrator=integrator)


@pytest.mark.parametrize("kind,channel,index,name", CASES)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_full_travel_and_reversal_obey_position_and_rate_limits(
    kind, channel, index, name, sign, integrator
):
    model = make_model(kind, integrator=integrator)
    limit = getattr(model.param, "maxabs" + name)
    rate_limit = getattr(model.param, "maxabsd" + name)
    previous = 0.0
    for step in range(360):
        command = np.zeros(model.action_space_length)
        command[channel] = sign * limit * (1 if step < 160 else -1)
        model.run_step(command)
        position, rate = model.current_state[index : index + 2]
        assert abs(position) <= limit + 1e-14
        assert abs(rate) <= rate_limit + 1e-14
        assert abs(position - previous) <= model.dt * rate_limit + 1e-14
        previous = position
    assert sign * previous < -0.7 * limit  # leaves the stop and reverses


@pytest.mark.parametrize("kind,channel,index,name", CASES)
def test_small_signal_preserves_analytic_second_order_servo(kind, channel, index, name):
    model = make_model(kind, dt=0.001)
    t = getattr(model.param, "T" + name)
    damping = getattr(model.param, "Xi" + name)
    matrix = np.array([[0, 1], [-1 / t**2, -2 * damping / t]])
    equilibrium = np.array([0.001, 0])
    for step in range(1, 101):
        command = np.zeros(model.action_space_length)
        command[channel] = equilibrium[0]
        model.run_step(command)
        expected = equilibrium - expm(matrix * (step * model.dt)) @ equilibrium
        np.testing.assert_allclose(
            model.current_state[index : index + 2], expected, atol=3e-9, rtol=0
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_split_stabilator_cannot_bypass_surface_travel_limit(sign):
    huge = AngularF16(np.zeros(14), split_stab=True, integrator="rk4")
    bounded = AngularF16(np.zeros(14), split_stab=True, integrator="rk4")
    command = sign * np.array([10.0, -10.0, 0, 0])
    huge.run_step(command)
    bounded.run_step(np.clip(command, -huge.param.maxabsstab, huge.param.maxabsstab))
    np.testing.assert_array_equal(huge.current_state, bounded.current_state)
    np.testing.assert_array_equal(huge.u_history, bounded.u_history)


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_efficiency_loss_cannot_be_cancelled_by_an_oversized_command(kind):
    huge = make_model(kind)
    bounded = make_model(kind)
    for model in (huge, bounded):
        model.damage_state = DamageState(section_loss={}, control_failures={})
        model.damage_state.set_control_failure(
            "stab_left", ControlFailure(mode="efficiency_loss", efficiency=0.25)
        )
    command = np.zeros(huge.action_space_length)
    command[0] = 10
    huge.run_step(command)
    command[0] = bounded.param.maxabsstab
    bounded.run_step(command)
    np.testing.assert_array_equal(huge.current_state, bounded.current_state)
    assert huge.u_history[-1][0, 0] == pytest.approx(0.25 * huge.param.maxabsstab)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("thrust", ["constant", "control"])
def test_environment_box_and_conversion_match_each_surface(split, thrust):
    env = NonlinearAngularF16(np.zeros(14), 10, split_stab=split, thrust_mode=thrust)
    env.reset()
    expected = [25, 25, 21.5, 30] if split else [25, 21.5, 30]
    if thrust == "control":
        expected.append(130000)
    np.testing.assert_allclose(env.action_space.high, expected, rtol=0, atol=1e-12)
    env.step(np.array(expected) * 2)
    surfaces = env.model.u_history[-1].reshape(-1)
    n_surface = 4 if split else 3
    np.testing.assert_allclose(
        np.rad2deg(surfaces[:n_surface]), expected[:n_surface], atol=1e-12
    )
