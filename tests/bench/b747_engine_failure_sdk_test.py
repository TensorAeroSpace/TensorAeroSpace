"""Causal faults and common actuator constraints in the B747 comparison."""

from dataclasses import replace

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747.nonlinear.dynamics import b747_ode_6dof
from tensoraerospace.agent.aa_indi import FlightMeasurement
from tensoraerospace.agent.aa_indi.kinematics import aircraft_kinematics
from tensoraerospace.benchmark import B747EngineFailureBenchmark


@pytest.mark.parametrize("algorithm", B747EngineFailureBenchmark.algorithms)
def test_fault_is_causal_and_all_controllers_obey_same_surface_limits(algorithm):
    cfg = B747EngineFailureBenchmark(
        duration=4, dt=0.02, fault_time=2, engine_fraction=0
    )
    healthy = cfg.run(algorithm, fault=False)
    faulty = cfg.run(algorithm, fault=True)
    before = round(cfg.fault_time / cfg.dt)
    np.testing.assert_array_equal(
        healthy["states"][: before + 1], faulty["states"][: before + 1]
    )
    np.testing.assert_array_equal(
        healthy["actions"][:before], faulty["actions"][:before]
    )
    assert np.linalg.norm(healthy["states"][-1] - faulty["states"][-1]) > 1e-4
    assert len(faulty["states"]) == cfg.steps + 1
    assert len(faulty["events"]) == 1
    assert {
        k: faulty["events"][0][k] for k in ("time", "engine_id", "thrust_fraction")
    } == {"time": 2, "engine_id": 1, "thrust_fraction": 0}
    actual = faulty["actions"][:, 1:3]
    assert abs(actual).max() <= np.deg2rad(cfg.surface_limit_deg) + 1e-12
    assert (
        abs(np.diff(np.vstack([np.zeros(2), actual]), axis=0)).max()
        <= np.deg2rad(cfg.surface_rate_deg_s) * cfg.dt + 1e-12
    )
    if algorithm == "AA-INDI":
        assert faulty["updates"] == [cfg.steps] * 3
        assert np.isfinite(faulty["learning"]).all()
        assert faulty["learning"][-1, 0] > 0


def test_sensor_specific_force_contains_engine_loss_and_matches_kinematics():
    cfg = B747EngineFailureBenchmark(
        duration=2, dt=0.02, fault_time=0, engine_fraction=0
    )
    env = cfg.make_env(fault=True)
    state, _ = env.reset(seed=1)
    tr = cfg.nominal_trim()
    action = np.array([tr.elevator_rad, 0, 0, tr.throttle])
    state, *_ = env.step(action)
    params = env.model.param
    packet = FlightMeasurement.from_model(env.model, surface_indices=(1, 2))
    derivative = b747_ode_6dof(state, action, cfg.dt, params)
    reconstructed = aircraft_kinematics(
        np.r_[state[:3] * 0.3048, state[6:9]],
        packet.imu,
        np.zeros(6),
        gravity=params.g_ft_s2 * 0.3048,
    )
    np.testing.assert_allclose(reconstructed[:3], derivative[:3] * 0.3048, atol=1e-12)
    np.testing.assert_allclose(reconstructed[3:], derivative[6:9], atol=1e-12)
    assert derivative[5] < 0  # Left engine loss yaws toward the failed engine.
    assert derivative[0] < 0  # Reduced total thrust decelerates the aircraft.
    np.testing.assert_array_equal(packet.surface_position, action[1:3])
    env.close()


@pytest.mark.parametrize("integral", [False, True])
def test_nominal_lqr_gains_stabilize_their_discrete_design_model(integral):
    cfg = B747EngineFailureBenchmark()
    tr = cfg.nominal_trim()
    A, B = cfg.nominal_model().lateral_linearization(
        [tr.elevator_rad, 0, 0, tr.throttle]
    )
    if not integral:
        A, B = A[:5, :5], B[:5]
    K = cfg.make_lqr(integral=integral).K
    assert abs(np.linalg.eigvals(A - B @ K)).max() < 1


def test_refined_physics_preserves_engine_event_and_response():
    cfg = B747EngineFailureBenchmark(
        duration=4, dt=0.02, fault_time=2, engine_fraction=0
    )
    ordinary = cfg.run("AA-INDI", fault=True)
    refined = replace(cfg, substeps=2).run("AA-INDI", fault=True)
    assert ordinary["events"] == refined["events"]
    np.testing.assert_allclose(
        ordinary["states"][:, [6, 8]], refined["states"][:, [6, 8]], atol=1e-6, rtol=0
    )


def test_tracking_metric_does_not_accept_constant_offset_as_recovery():
    states = np.zeros((101, 12))
    states[:, 0] = 674.0
    states[:, 11] = -20000.0
    states[:, 8] = np.deg2rad(0.2)
    result = B747EngineFailureBenchmark().evaluate(
        states, np.zeros((100, 4)), start=0, end=2
    )
    assert result["heading_rmse_deg"] == pytest.approx(0.2)
    assert result["recovery_s"] is None
    assert result["angle_iae_deg_s"] == pytest.approx(0.4)
