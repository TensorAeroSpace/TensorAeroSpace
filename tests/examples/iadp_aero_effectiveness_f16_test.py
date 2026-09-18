"""Check aerodynamic effectiveness physics, servo feedback and fault timing."""

from functools import partial

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_aero_effectiveness_f16 as example,
)
from example.reinforcement_learning.incremental_adp import (
    example_iadp_tuned_f16 as tuned,
)
from tensoraerospace.aerospacemodel.f16.nonlinear._integrators import rk4
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.aero import (
    get_cy,
    get_mz,
)


@pytest.mark.parametrize("eta", [0.0, 0.4, 0.7, 1.0])
def test_effectiveness_scales_aerodynamic_force_and_cg_moment_only(eta):
    p = example.default_parameters()
    x = np.array([0.09, 0.01, -0.07, 0.02])
    u = np.array([-0.1])
    cy0 = get_cy(x[0], 0, 0, p.lef, x[1], p.V, p.bA, p.sb)
    cm0 = get_mz(x[0], 0, 0, p.lef, x[1], p.V, p.bA, p.sb)
    cy = cy0 + eta * (get_cy(x[0], 0, x[2], p.lef, x[1], p.V, p.bA, p.sb) - cy0)
    cm = cm0 + eta * (get_mz(x[0], 0, x[2], p.lef, x[1], p.V, p.bA, p.sb) - cm0)
    force = p.q * p.S * cy
    moment = p.q * p.S * p.bA * cm + p.rcgx * force
    actual = example.effectiveness_rhs(x, u, 0, p, effectiveness=eta)
    np.testing.assert_allclose(
        actual[:2],
        [x[1] - (force - p.m * p.g) / (p.m * p.V), moment / p.Jz],
        atol=1e-14,
        rtol=1e-12,
    )
    native = example.f16_ode_long(x, u, 0, p)
    np.testing.assert_array_equal(actual[2:], native[2:])
    if eta == 1:
        np.testing.assert_array_equal(actual, native)


def test_effectiveness_changes_gain_without_faking_surface_measurement():
    p = example.default_parameters()
    x = np.array([0.08, 0.01, -0.08, 0.0])
    derivatives = []
    for eta in (1.0, 0.7):
        plus, minus = x.copy(), x.copy()
        plus[2] += 1e-5
        minus[2] -= 1e-5
        derivatives.append(
            (
                example.effectiveness_rhs(plus, [0], 0, p, effectiveness=eta)[1]
                - example.effectiveness_rhs(minus, [0], 0, p, effectiveness=eta)[1]
            )
            / 2e-5
        )
    assert derivatives[0] < 0
    assert derivatives[1] / derivatives[0] == pytest.approx(0.7)


def test_interior_event_splits_physics_before_changing_effectiveness():
    x = np.array([0.08, 0.01, -0.08, 0.0])
    events = []
    model = example.EffectivenessModel(
        x,
        dt=0.02,
        integrator="rk4",
        aero_fault=example.AeroFault(0.007, 0.7),
        on_fault=events.append,
    )
    u = np.array([-0.08])
    before = rk4(example.f16_ode_long, x, u, 0, 0.007, model.param)
    expected = rk4(
        partial(example.effectiveness_rhs, effectiveness=0.7),
        before,
        u,
        0.007,
        0.013,
        model.param,
    )
    model.run_step(u)
    np.testing.assert_allclose(model.current_state, expected, atol=1e-14, rtol=0)
    model.run_step(u)
    assert len(events) == 1
    assert events[0].time_s == 0.007


def test_healthy_equivalence_prefault_history_and_reset():
    cfg = tuned.baseline.Experiment(duration=1.0, fault_time=0.4)
    x, _, _ = tuned.baseline.trim_and_agent(cfg)
    reference = tuned.comparison.reference_signal(cfg)
    native = tuned.baseline.make_environment(cfg, x, reference, fault=False)
    same = example.make_environment(
        cfg, x, reference, fault=False, aero_fault=example.AeroFault(0.4, 0.7)
    )
    faulty = example.make_environment(
        cfg, x, reference, fault=True, aero_fault=example.AeroFault(0.4, 0.7)
    )
    try:
        for env in (native, same, faulty):
            env.reset()
        for k in range(cfg.steps):
            command = np.array([0.1 * np.sin(k * 0.1)])
            native.step(command)
            same.step(command)
            faulty.step(command)
            np.testing.assert_allclose(
                same.model.current_state, native.model.current_state, atol=1e-12, rtol=0
            )
            if k < cfg.fault_step:
                np.testing.assert_allclose(
                    faulty.model.current_state,
                    native.model.current_state,
                    atol=1e-12,
                    rtol=0,
                )
        assert same.damage_events_log == []
        assert len(faulty.damage_events_log) == 1
        assert not np.allclose(faulty.model.current_state, native.model.current_state)
        # Servo follows the same command independently of aerodynamic forces.
        np.testing.assert_allclose(
            faulty.model.current_state[2:],
            native.model.current_state[2:],
            atol=1e-14,
            rtol=0,
        )
        faulty.reset()
        assert not faulty.model.fault_applied
        assert faulty.model.effectiveness == 1
        assert faulty.damage_events_log == []
    finally:
        for env in (native, same, faulty):
            env.close()


def test_agent_does_not_use_the_aerodynamic_fault_schedule():
    from dataclasses import replace

    cfg = tuned.baseline.Experiment(duration=2.0, fault_time=0.4)
    factory = partial(example.make_environment, aero_fault=example.AeroFault(0.8, 0.7))
    first, _ = tuned.rollout(
        cfg, example.selected_tuning(), fault=True, environment_factory=factory
    )
    second, _ = tuned.rollout(
        replace(cfg, fault_time=1.4),
        example.selected_tuning(),
        fault=True,
        environment_factory=factory,
    )
    np.testing.assert_array_equal(first, second)


def test_selected_controller_keeps_both_learning_loops_active_after_fault():
    cfg = tuned.baseline.Experiment(duration=8.0, fault_time=1.0)
    factory = partial(example.make_environment, aero_fault=example.AeroFault(1.0, 0.7))
    rows = []
    _, diagnostics = tuned.rollout(
        cfg,
        example.selected_tuning(),
        fault=True,
        environment_factory=factory,
        telemetry=rows,
    )
    data = np.asarray(rows)
    assert np.ptp(data[data[:, 0] > 6.0, 10]) > 0
    assert np.ptp(data[data[:, 0] > 6.0, 11]) > 0
    assert diagnostics["model_change_after_event"] > 0
    assert diagnostics["critic_change_after_event"] > 0
    assert example.selected_tuning().blend > 0


def test_expanded_pid_bounds_are_supported_without_changing_default_search():
    default = tuned.comparison.PID_SEARCH_BOUNDS
    _, report = tuned.comparison.tune_pid(
        max_evaluations=4, gain_magnitude_bounds=example.PID_GAIN_BOUNDS
    )
    np.testing.assert_array_equal(
        report["gain_magnitude_bounds"], example.PID_GAIN_BOUNDS
    )
    assert not report["fault_used"]
    assert report["evaluations"] == 4
    assert tuned.comparison.PID_SEARCH_BOUNDS == default
