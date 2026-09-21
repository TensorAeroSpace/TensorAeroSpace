"""Aerodynamic fault physics, encoder feedback and exact event timing."""

from dataclasses import replace

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness
from tensoraerospace.aerospacemodel.b737.nonlinear._integrators import rk4
from tensoraerospace.aerospacemodel.b737.nonlinear.dynamics import b737_ode_6dof
from tensoraerospace.agent.aa_indi import FlightMeasurement
from tensoraerospace.agent.aa_indi.kinematics import aircraft_kinematics
from tensoraerospace.benchmark import B737PitchStepBenchmark


def setup(fault, *, substeps=1):
    cfg = B737PitchStepBenchmark(
        duration=2, step_time=1, elevator_fault=fault, integration_substeps=substeps
    )
    env, tr, action = cfg.make_env()
    state, _ = env.reset(seed=47)
    return cfg, env, tr, action, state


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_aerodynamic_effectiveness_changes_input_gain_not_encoder(eta):
    fault = ElevatorEffectiveness(0, eta)
    _, env, _, action, state = setup(fault)
    params = env.model.param
    h = 1e-5
    perturbation = np.array([h, 0, 0, 0])
    gain = (
        env.model.dynamics(state, action + perturbation, time=0)[4]
        - env.model.dynamics(state, action - perturbation, time=0)[4]
    ) / (2 * h)
    nominal_gain = (
        b737_ode_6dof(state, action + perturbation, 0, params)[4]
        - b737_ode_6dof(state, action - perturbation, 0, params)[4]
    ) / (2 * h)
    assert gain == pytest.approx(eta * nominal_gain, abs=1e-10)
    next_state, *_ = env.step(action)
    applied = env.model.u_history[-1].ravel()
    np.testing.assert_array_equal(applied, action)
    packet = FlightMeasurement.from_model(env.model, surface_indices=(0,))
    np.testing.assert_array_equal(packet.surface_position, action[:1])
    actual_rates = aircraft_kinematics(
        np.r_[next_state[:3] * 0.3048, next_state[6:9]],
        packet.imu,
        np.zeros(6),
        gravity=params.g_ft_s2 * 0.3048,
    )
    expected = env.model.dynamics(next_state, applied)
    np.testing.assert_allclose(actual_rates[:3], expected[:3] * 0.3048, atol=1e-12)
    np.testing.assert_allclose(actual_rates[3:], expected[6:9], atol=1e-12)
    env.close()


def test_event_inside_rk4_interval_splits_healthy_and_faulty_dynamics():
    fault = ElevatorEffectiveness(0.007, 0.5)
    _, env, _, action, state = setup(fault)
    before = rk4(b737_ode_6dof, state, action, 0, 0.007, env.model.param)
    aero_action = action.copy()
    aero_action[0] *= 0.5
    expected = rk4(b737_ode_6dof, before, aero_action, 0.007, 0.013, env.model.param)
    actual, *_ = env.step(action)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    env.close()


def test_endpoint_event_cannot_change_pre_fault_trajectory():
    cfg, damaged, _, action, _ = setup(ElevatorEffectiveness(1.0, 0.5))
    healthy, _, _ = replace(cfg, elevator_fault=None).make_env()
    healthy.reset(seed=47)
    for _ in range(50):
        x1, *_ = healthy.step(action)
        x2, *_ = damaged.step(action)
        np.testing.assert_allclose(x2, x1, atol=1e-10, rtol=0)
    x1, *_ = healthy.step(action)
    x2, *_ = damaged.step(action)
    assert abs(x1[4] - x2[4]) > 1e-5
    healthy.close()
    damaged.close()


def test_unity_effectiveness_preserves_native_model_for_whole_episode():
    cfg, damaged, _, action, _ = setup(ElevatorEffectiveness(0.347, 1.0))
    healthy, _, _ = replace(cfg, elevator_fault=None).make_env()
    healthy.reset(seed=47)
    for k in range(cfg.steps):
        command = action.copy()
        command[0] += np.deg2rad(0.1) * np.sin(k * cfg.dt)
        x1, *_ = healthy.step(command)
        x2, *_ = damaged.step(command)
        np.testing.assert_allclose(x2, x1, atol=2e-9, rtol=0)
    healthy.close()
    damaged.close()


def test_fault_reset_restores_initial_state_and_time():
    _, env, _, action, initial = setup(ElevatorEffectiveness(0.03, 0.5))
    first = [env.step(action)[0] for _ in range(5)]
    reset, _ = env.reset(seed=47)
    np.testing.assert_array_equal(reset, initial)
    second = [env.step(action)[0] for _ in range(5)]
    np.testing.assert_array_equal(first, second)
    env.close()
