"""Physical-unit, timing and step-assessment checks for the B737 notebooks."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b737.nonlinear import b737_ode_6dof
from tensoraerospace.aerospacemodel.b737.nonlinear.aero import AeroState, b737_aero
from tensoraerospace.aerospacemodel.b737.nonlinear.params import isa_density_slug_ft3
from tensoraerospace.agent.aa_indi import AircraftGeometry, FlightMeasurement
from tensoraerospace.agent.aa_indi.kinematics import aircraft_kinematics, body_to_ned
from tensoraerospace.benchmark import B737PitchStepBenchmark


@pytest.fixture
def plant():
    experiment = B737PitchStepBenchmark()
    env, trim, action = experiment.make_env()
    state, _ = env.reset(seed=experiment.seed)
    yield experiment, env, trim, action, state
    env.close()


@pytest.mark.parametrize(
    "attitude", [[0, 0, 0], [0.12, 0.08, -0.2], [-0.2, -0.05, 0.7]]
)
def test_synthetic_imu_recovers_body_and_navigation_kinematics(plant, attitude):
    _, env, _, action, state = plant
    state[1] = 3.0
    state[3:6] = [0.02, -0.03, 0.01]
    state[6:9] = attitude
    action = action + [0.003, 0.002, -0.001, 0.0]
    params = env.unwrapped.model.param
    packet = FlightMeasurement.from_model(
        env.model, state=state, applied_action=action, time=1.0, surface_indices=(0,)
    )
    rates = aircraft_kinematics(
        np.r_[state[:3] * 0.3048, state[6:9]],
        packet.imu,
        np.zeros(6),
        gravity=params.g_ft_s2 * 0.3048,
    )
    derivative = b737_ode_6dof(state, action, 1.0, params)
    np.testing.assert_allclose(rates[:3], derivative[:3] * 0.3048, atol=1e-12)
    np.testing.assert_allclose(rates[3:], derivative[6:9], atol=1e-12)
    np.testing.assert_allclose(
        packet.ground_velocity, derivative[9:12] * 0.3048, atol=1e-12
    )
    np.testing.assert_array_equal(packet.surface_position, action[:1])
    assert packet.airspeed == pytest.approx(np.linalg.norm(state[:3]) * 0.3048)
    assert packet.density == pytest.approx(0.652977, rel=1e-6)


def test_trim_specific_force_has_gravity_removed_in_rotated_body_frame(plant):
    _, env, _, action, state = plant
    params = env.unwrapped.model.param
    packet = FlightMeasurement.from_model(
        env.model, state=state, applied_action=action, time=0, surface_indices=(0,)
    )
    expected = -body_to_ned(state[6:9]).T @ np.array([0, 0, params.g_ft_s2 * 0.3048])
    np.testing.assert_allclose(packet.specific_force, expected, atol=1e-8)
    assert packet.specific_force[2] < -9.7  # Down-positive body frame.


def test_si_inertia_reconstructs_b737_aerodynamic_moments_with_cross_coupling(plant):
    _, env, _, action, state = plant
    params = env.unwrapped.model.param
    state[1] = 5.0
    state[3:6] = [0.03, -0.02, 0.04]
    action = action + [0.01, 0.02, -0.015, 0.0]
    speed = np.linalg.norm(state[:3])
    aero = b737_aero(
        AeroState(
            alpha=np.arctan2(state[2], state[0]),
            beta=np.arcsin(state[1] / speed),
            V=speed,
            p=state[3],
            q=state[4],
            r=state[5],
            altitude_ft=-state[11],
            de=action[0],
            da=action[1],
            dr=action[2],
        ),
        params,
    )
    packet = FlightMeasurement.from_model(
        env.model, state=state, applied_action=action, time=0, surface_indices=(0,)
    )
    derivative = b737_ode_6dof(state, action, 0, params)
    actual = AircraftGeometry.from_parameters(params).coefficients(
        state[3:6],
        derivative[3:6],
        packet.density,
        packet.airspeed,
    )
    rho = isa_density_slug_ft3(-state[11])
    scale = (
        0.5
        * rho
        * speed**2
        * params.S_ft2
        * np.array([params.b_ft, params.cbar_ft, params.b_ft])
    )
    expected = np.array([aero.l, aero.m, aero.n]) / scale
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_sensor_surface_feedback_uses_the_clipped_plant_input(plant):
    experiment, env, _, action, _ = plant
    action = action.copy()
    action[0] = 2.0  # Intentionally outside the physical actuator range.
    state, *_ = env.step(action)
    applied = env.unwrapped.model.u_history[-1].ravel()
    packet = FlightMeasurement.from_model(env.model, surface_indices=(0,))
    assert packet.surface_position[0] == pytest.approx(env.action_space.high[0])
    assert packet.surface_position[0] != action[0]
    # iADP must receive this actual surface minus its trim, not the request.


def test_reference_and_horizon_have_matching_sample_times():
    experiment = B737PitchStepBenchmark(duration=20, step_time=5)
    index = round(experiment.step_time / experiment.dt)
    assert len(experiment.time) == experiment.steps + 1
    assert experiment.time[-1] == 20
    assert experiment.reference[index - 1] == 0
    assert experiment.reference[index] == pytest.approx(np.deg2rad(1))
    with pytest.raises(ValueError, match="integer multiples"):
        B737PitchStepBenchmark(step_time=15.001)


def test_benchmark_distinguishes_settling_from_command_tracking():
    experiment = B737PitchStepBenchmark(duration=20, step_time=5)
    states = np.zeros((experiment.steps + 1, 12))
    states[:, 0], states[:, 11] = 650, -20_000
    states[:, 7] = np.deg2rad(2) + 0.8 * experiment.reference
    actions = np.zeros((experiment.steps, 4))
    windows, physical = experiment.evaluate(states, actions)
    for metrics in windows.values():
        assert metrics["settling_time"] == 0.0  # Constant final OUTPUT.
        assert metrics["command_settling_time"] is None  # Always 20% below command.
        assert metrics["command_overshoot"] == 0.0
        assert np.rad2deg(metrics["static_error"]) == pytest.approx(0.2)
        # Native benchmark uses a rectangle sum including the endpoint sample.
        assert np.rad2deg(metrics["iae"]) == pytest.approx(0.2 * (15 + experiment.dt))
    assert physical["Final pitch error [deg]"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="complete finite trajectory"):
        experiment.evaluate(states[:-1], actions[:-1])


def test_early_truncation_cannot_be_reported_as_a_completed_run(plant):
    experiment, _, _, _, state = plant
    with pytest.raises(RuntimeError, match="before the requested horizon"):
        experiment.validate_transition(state, False, True, 5)
    experiment.validate_transition(state, False, True, experiment.steps - 1)
    with pytest.raises(RuntimeError, match="configured horizon"):
        experiment.validate_transition(state, False, False, experiment.steps - 1)
