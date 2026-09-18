"""Step metrics must not conceal tracking bias or change the plant physics."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_step_response_b747 as steps,
    example_etdhp_vs_pid_b747 as demo,
)


@pytest.mark.parametrize("offset,amplitude", [(0, 1), (6096, 10), (205.4352, -1)])
def test_first_order_rise_and_settling_are_offset_and_sign_invariant(offset, amplitude):
    t = np.arange(0, 102.001, 0.01)
    tau = np.maximum(t - 2, 0)
    y = offset + amplitude * (1 - np.exp(-tau / 3))
    m = steps.step_metrics(
        t,
        y,
        step_time=2,
        initial_reference=offset,
        final_reference=offset + amplitude,
        tail_seconds=20,
        confirmation_seconds=10,
    )
    assert m["rise_10_90_s"] == pytest.approx(3 * np.log(9), abs=0.011)
    assert m["settling_2pct_s"] == pytest.approx(-3 * np.log(0.02), abs=0.011)
    assert m["settling_5pct_s"] == pytest.approx(-3 * np.log(0.05), abs=0.011)
    assert m["overshoot_pct"] == pytest.approx(0)
    assert m["iae"] == pytest.approx(3 * abs(amplitude), rel=1e-4)
    assert m["ise"] == pytest.approx(1.5 * amplitude**2, rel=1e-4)
    assert m["itae"] == pytest.approx(9 * abs(amplitude), rel=1e-4)
    assert abs(m["tail_bias"]) < 1e-9


def test_second_order_overshoot_and_peak_time():
    t = np.arange(0, 102.001, 0.01)
    tau = np.maximum(t - 2, 0)
    zeta, wn = 0.3, 1.0
    wd = wn * np.sqrt(1 - zeta**2)
    y = 1 - np.exp(-zeta * wn * tau) * (
        np.cos(wd * tau) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * tau)
    )
    m = steps.step_metrics(t, y, step_time=2, initial_reference=0, final_reference=1)
    assert m["overshoot_pct"] == pytest.approx(
        100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)), abs=0.001
    )
    assert m["peak_time_s"] == pytest.approx(np.pi / wd, abs=0.01)


def test_constant_wrong_output_cannot_be_reported_as_settled():
    t = np.arange(0, 102.001, 0.1)
    m = steps.step_metrics(
        t,
        np.full_like(t, 6096),
        step_time=2,
        initial_reference=6096,
        final_reference=6106,
    )
    assert m["settling_2pct_s"] is None and m["settling_5pct_s"] is None
    assert m["rise_10_90_s"] is None
    assert m["tail_bias"] == -10
    assert m["iae"] == pytest.approx(1000)


def test_late_entry_needs_a_confirmation_window_and_oscillations_do_not_settle():
    t = np.arange(0, 102.001, 0.1)
    for y in (np.where(t > 100, 1.0, 0.0), np.where(t < 2, 0.0, 1 + 0.1 * np.sin(t))):
        m = steps.step_metrics(
            t, y, step_time=2, initial_reference=0, final_reference=1
        )
        assert m["settling_2pct_s"] is None and m["settling_5pct_s"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_metric_input_is_rejected(bad):
    with pytest.raises(ValueError):
        steps.step_metrics(
            [0, 1, 2], [0, bad, 1], step_time=1, initial_reference=0, final_reference=1
        )


def test_reference_is_causal_and_changes_only_the_selected_channel():
    p = steps.Protocol()
    for channel in range(4):
        np.testing.assert_array_equal(
            steps.reference_at(p.step_time_s - p.dt, channel, p),
            steps.NOMINAL_REFERENCE,
        )
        delta = steps.reference_at(p.step_time_s, channel, p) - steps.NOMINAL_REFERENCE
        assert np.count_nonzero(delta) == 1
        assert delta[channel] == pytest.approx(steps.STEP_AMPLITUDES[channel])


def test_pid_measured_derivative_has_no_reference_kick():
    obs = demo.nominal_trim().to_state()
    pid = demo.LateralPID([0, 0, 10, 0, 0, 10], 0.05)
    np.testing.assert_array_equal(
        pid.command(obs, roll_ref_deg=0.5, heading_ref_deg=1), [0, 0]
    )
    lon = demo.LongitudinalHold(0.05)
    lon.command(obs, height_ref_ft=demo.ALTITUDE + 10)
    assert (
        lon.hdot == 0
    )  # altitude derivative uses the measurement, not the reference error


@pytest.mark.parametrize("algorithm", ["etdhp", "pid"])
def test_reference_extension_preserves_original_zero_reference_trajectory(algorithm):
    p = steps.Protocol(
        warmup_s=1,
        step_time_s=0.2,
        response_s=1,
        fault_time_s=0.1,
        tail_s=0.2,
        confirmation_s=0.2,
    )
    template, _ = demo.make_agent()
    state = steps.TrialState(p, algorithm, fault=True, template=template)
    cfg = replace(state.cfg, duration=1)
    original, _ = demo.rollout(
        cfg, algorithm, fault=True, gains=demo.NOMINAL_PID_GAINS, template=template
    )
    actual = []
    for k in range(cfg.steps):
        state.advance(steps.NOMINAL_REFERENCE)
        actual.append(state.record((k + 1) * p.dt, steps.NOMINAL_REFERENCE))
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual[:, 2:4], original[:, 1:3], rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        actual[:, 4], (original[:, 4] + demo.SPEED) * 0.3048, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        actual[:, 5], (original[:, 5] + demo.ALTITUDE) * 0.3048, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(actual[:, 11:15], original[:, 10:14], rtol=0, atol=1e-11)
    clone = copy.deepcopy(state)
    assert clone.step == state.step
    assert clone.env.damage_events_log == state.env.damage_events_log
    for x, y in zip(clone.weights(), state.weights()):
        np.testing.assert_array_equal(x, y)
