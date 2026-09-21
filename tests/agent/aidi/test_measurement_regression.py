"""AIDI must identify matched actuator/acceleration measurements (Ul Haq §III.B)."""

import copy

import numpy as np
import pytest

from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, LinearOnboardCE
from tensoraerospace.agent.aidi.scaling_rls import ScalingRLS


def observation(omega=None):
    return dict(
        omega=np.zeros(3) if omega is None else np.asarray(omega),
        alpha=0.05,
        beta=0.0,
        theta=0.05,
        phi=0.0,
        V=200.0,
        n_z=1.0,
    )


REF = dict(C_star=1.0, phi_cmd=0.0, beta_cmd=0.0, V_cmd=200.0)
G = np.diag([2.0, -3.0, 1.5])


def make_agent():
    return AIDIAgent(
        3,
        3,
        LinearOnboardCE(G),
        AIDIConfig(
            dt=0.01, sensor_cutoff_hz=2.0, u_magnitude_limit=1.0, u_rate_limit=10.0
        ),
    )


def test_first_measured_transition_is_not_discarded():
    agent = make_agent()
    agent.predict(observation([0.2, 0.3, -0.1]), REF)
    agent.learn(observation([0.21, 0.28, -0.07]), REF)
    np.testing.assert_allclose(
        agent.deriv.last_output, agent.deriv._alpha * np.array([1.0, -2.0, 3.0])
    )


@pytest.mark.parametrize("feedback", [False, True])
def test_filter_delay_and_actuator_lag_are_not_identified_as_damage(feedback):
    agent = make_agent()
    rng = np.random.default_rng(37)
    omega, applied = np.zeros(3), np.zeros(3)
    for _ in range(400):
        agent.allocator.allocate = lambda *args: rng.uniform(-0.03, 0.03, 3)
        command = agent.predict(observation(omega), REF)
        applied = 0.7 * applied + 0.3 * command if feedback else command
        omega = omega + agent.cfg.dt * (G @ applied)
        kwargs = {"applied_action": applied} if feedback else {}
        agent.learn(observation(omega), REF, **kwargs)
    np.testing.assert_allclose(np.diag(agent.rls.theta), 1.0, rtol=0, atol=1e-12)
    assert agent.rls.num_updates == 399


def test_incremental_control_uses_synchronized_actuator_baseline():
    agent = make_agent()
    agent.predict(observation(), REF)
    agent.learn(observation([0.002, 0, 0]), REF, applied_action=[0.1, 0, 0])
    agent.allocator.allocate = lambda *args: np.zeros(3)
    command = agent.predict(observation([0.002, 0, 0]), REF)
    np.testing.assert_allclose(command, [agent.deriv._alpha * 0.1, 0, 0])
    assert np.max(abs(command - agent._u_prev)) <= agent.cfg.dt * agent.cfg.u_rate_limit


def test_initial_actuator_position_is_preserved_and_validated():
    agent = make_agent()
    agent.reset(initial_action=[-0.08, 0.01, 0])
    agent.allocator.allocate = lambda *args: np.zeros(3)
    np.testing.assert_allclose(agent.predict(observation(), REF), [-0.08, 0.01, 0])
    with pytest.raises(ValueError):
        agent.reset(initial_action=[np.nan, 0, 0])
    np.testing.assert_allclose(agent._u_prev, [-0.08, 0.01, 0])


def test_native_state_fallback_maps_body_axes_and_attitude():
    agent = make_agent()
    obs = observation([0.1, 0.2, 0.3])
    obs["phi"] = 0.4
    agent.reset(initial_action=[-0.08, 0.01, 0.02])
    x = agent._build_state_vector(obs)
    np.testing.assert_array_equal(x[[2, 4, 3]] * [1, 1, -1], obs["omega"])
    assert x[5] == obs["phi"] and x[7] == obs["theta"]
    np.testing.assert_allclose(x[[8, 10, 12]], [-0.08, 0.01, 0.02])


def test_frozen_identifier_still_advances_measurements():
    agent = make_agent()
    before = copy.deepcopy(agent.rls.__dict__)
    for k in range(5):
        agent.predict(observation([0.01 * k, 0, 0]), REF)
        agent.learn(observation([0.01 * (k + 1), 0, 0]), REF, adapt=False)
    for key, value in before.items():
        np.testing.assert_equal(agent.rls.__dict__[key], value)
    assert agent._step == 5 and agent.deriv.last_output[0] > 0


@pytest.mark.parametrize("bad", [[np.nan, 0, 0], [np.inf, 0, 0], [1, 2]])
def test_invalid_feedback_does_not_mutate_loop_or_estimator(bad):
    agent = make_agent()
    agent.predict(observation(), REF)
    before = copy.deepcopy(agent)
    with pytest.raises(ValueError):
        agent.learn(observation(), REF, applied_action=bad)
    np.testing.assert_equal(agent.deriv.__dict__, before.deriv.__dict__)
    np.testing.assert_equal(agent.rls.__dict__, before.rls.__dict__)
    np.testing.assert_array_equal(agent._u_prev, before._u_prev)


def test_learn_requires_an_unconsumed_command():
    agent = make_agent()
    with pytest.raises(RuntimeError):
        agent.learn(observation(), REF)
    agent.predict(observation(), REF)
    agent.learn(observation(), REF)
    with pytest.raises(RuntimeError):
        agent.learn(observation(), REF)


@pytest.mark.parametrize("pending", [False, True])
def test_checkpoint_continuation_preserves_filtered_feedback(tmp_path, pending):
    agent = make_agent()
    omega = np.zeros(3)
    for k in range(8):
        agent.predict(observation(omega), REF)
        applied = np.array([0.01 * k, -0.005 * k, 0])
        omega = omega + 0.01 * G @ applied
        agent.learn(observation(omega), REF, applied_action=applied)
    if pending:
        agent.predict(observation(omega), REF)
    folder = agent.save(tmp_path)
    restored = AIDIAgent.from_pretrained(folder, onboard_ce=LinearOnboardCE(G))
    for k in range(8, 18):
        if not (k == 8 and pending):
            np.testing.assert_array_equal(
                agent.predict(observation(omega), REF),
                restored.predict(observation(omega), REF),
            )
        applied = np.array([0.01 * k, -0.005 * k, 0])
        omega = omega + 0.01 * G @ applied
        agent.learn(observation(omega), REF, applied_action=applied)
        restored.learn(observation(omega), REF, applied_action=applied)
        np.testing.assert_array_equal(agent.rls.theta, restored.rls.theta)
        np.testing.assert_array_equal(agent.rls.P, restored.rls.P)


def test_scaling_rls_retains_original_algorithm_one_gain_and_forgetting():
    rls = ScalingRLS(
        1,
        2,
        sigma0=2.0,
        memory_length=10,
        lambda_min=0.5,
        lambda_max=0.99,
        cov_trace_bound=1e6,
    )
    rls.P[0] = np.array([[3.0, 0.2], [0.2, 2.0]])
    rls.last_lambda[:] = 0.8
    phi = np.array([0.3, -0.7])
    previous = rls.P[0].copy()
    residual = 0.5 - phi.sum()
    gain = previous @ phi / (0.8 + phi @ previous @ phi)
    lam = np.clip(1 - (1 - phi @ gain) * residual**2 / 40, 0.5, 0.99)
    rls.update(phi, [0.5], np.ones((1, 2)))
    np.testing.assert_allclose(rls.theta[0], 1 + gain * residual)
    np.testing.assert_allclose(
        rls.P[0], (previous - np.outer(gain, phi) @ previous) / lam
    )
    assert rls.last_lambda[0] == pytest.approx(lam)


def test_command_slew_limit_is_respected_with_lagging_actuator_feedback():
    agent = make_agent()
    previous_command = np.zeros(3)
    agent.allocator.allocate = lambda *args: np.full(3, 3.0)
    for _ in range(10):
        command = agent.predict(observation(), REF)
        assert np.max(abs(command - previous_command)) <= 0.1 + 1e-12
        agent.learn(observation(), REF, applied_action=0.25 * command)
        previous_command = command
    # Command gain loss must not create an artificial feedback-centred ceiling.
    np.testing.assert_allclose(command, np.ones(3))


def test_legacy_checkpoint_retains_identifier_and_reprimes_measurements(tmp_path):
    from pathlib import Path

    agent = make_agent()
    for k in range(5):
        agent.predict(observation([k * 0.01, 0, 0]), REF)
        agent.learn(observation([(k + 1) * 0.01, 0, 0]), REF)
    folder = Path(agent.save(tmp_path))
    with np.load(folder / "loop_state.npz") as data:
        legacy = {
            key: data[key].copy()
            for key in data.files
            if key not in {"u_filtered", "pending_transition"}
        }
    np.savez(folder / "loop_state.npz", **legacy)
    restored = AIDIAgent.from_pretrained(folder, onboard_ce=LinearOnboardCE(G))
    np.testing.assert_array_equal(restored.rls.theta, agent.rls.theta)
    np.testing.assert_array_equal(restored.rls.P, agent.rls.P)
    assert restored.deriv._prev_x is None
    restored.predict(observation([0.05, 0, 0]), REF)
    restored.learn(observation([0.06, 0, 0]), REF)
    # No artificial identification increment bridges two filter conventions.
    np.testing.assert_array_equal(restored.rls.theta, agent.rls.theta)
    np.testing.assert_allclose(
        restored.deriv.last_output, [restored.deriv._alpha, 0, 0]
    )


def test_native_rate_mapping_satisfies_paper_euler_kinematics():
    """Ul Haq eq. (3): yaw sign is required at nonzero roll and pitch."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )

    agent = make_agent()
    obs = observation([0.11, 0.23, -0.17])
    obs.update(phi=0.31, theta=-0.21)
    native = agent._build_state_vector(obs)
    dx = f16_ode_6dof(native, np.zeros(3), 0, default_parameters())
    p, q, r = obs["omega"]
    phi, theta = obs["phi"], obs["theta"]
    expected = [
        p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r,
        np.cos(phi) * q - np.sin(phi) * r,
        (np.sin(phi) * q + np.cos(phi) * r) / np.cos(theta),
    ]
    # Native heading has the opposite orientation, too.
    np.testing.assert_allclose([dx[5], dx[7], -dx[6]], expected, atol=1e-14)
