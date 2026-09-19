"""Native model analysis is side-effect free and scheduled faults are causal."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747.nonlinear._integrators import rk4
from tensoraerospace.aerospacemodel.b747.nonlinear.dynamics import b747_ode_6dof
from tensoraerospace.agent.aa_indi import FlightMeasurement
from tensoraerospace.benchmark import B737PitchStepBenchmark, B747EngineFailureBenchmark


@pytest.mark.parametrize("aircraft", ["b737", "b747"])
def test_analysis_and_sensor_adapter_do_not_advance_or_mutate_model(aircraft):
    if aircraft == "b737":
        env, tr, action = B737PitchStepBenchmark().make_env()
    else:
        cfg = B747EngineFailureBenchmark()
        env, tr = cfg.make_env(fault=False), cfg.nominal_trim()
        action = np.array([tr.elevator_rad, 0, 0, tr.throttle])
    state, _ = env.reset()
    history = np.array(env.model.x_history)
    with pytest.raises(RuntimeError, match="No applied action"):
        FlightMeasurement.from_model(env.model)
    A, B = env.model.linearize(state, action)
    assert A.shape == (12, 12) and B.shape == (12, 4)
    FlightMeasurement.from_model(env.model, applied_action=action)
    assert env.model.current_time == 0
    np.testing.assert_array_equal(history, env.model.x_history)
    assert len(env.model.u_history) == 0
    env.step(action)
    packet = FlightMeasurement.from_model(env.model)
    assert packet.time == pytest.approx(env.dt)
    np.testing.assert_array_equal(packet.surface_position, env.model.applied_action[:3])
    env.close()


def test_off_grid_b747_engine_event_splits_integration_at_its_exact_time():
    cfg = B747EngineFailureBenchmark(duration=1, fault_time=0.007)
    env = cfg.make_env(fault=True)
    state, _ = env.reset()
    tr = cfg.nominal_trim()
    action = np.array([tr.elevator_rad, 0, 0, tr.throttle])
    healthy = cfg.nominal_model()
    expected = rk4(b747_ode_6dof, state, action, 0, 0.007, healthy.param)
    from tensoraerospace.aerospacemodel.b747.nonlinear.damage import EngineFailureEvent
    from tensoraerospace.aerospacemodel.b747.nonlinear.damage.state import (
        B747DamageState,
    )

    healthy.param.damage_state = B747DamageState.healthy()
    EngineFailureEvent(trigger_time=0.007, engine_id=1, thrust_fraction=0).apply(
        healthy.param.damage_state
    )
    expected = rk4(b747_ode_6dof, expected, action, 0.007, 0.013, healthy.param)
    actual, _, _, _, info = env.step(action)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert env.damage_events_log[0]["time"] == 0.007
    assert info["damage_events_triggered"]
    env.close()
