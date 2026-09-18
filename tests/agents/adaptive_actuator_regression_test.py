"""Adaptive identification must use aligned measurements and applied controls."""

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.envs.b747 import LinearLongitudinalB747
from tensoraerospace.envs.lapan import LinearLongitudinalLAPAN


def iadp(**kwargs):
    options = dict(
        dt=0.1,
        model_learning_only_steps=1000,
        excitation_signal=np.array([[2.0], [-2.0], [1.0]]),
        u_rate_limit=1.0,
        u_magnitude_limit=1.0,
        policy_eval_warmup_updates=10000,
    )
    options.update(kwargs)
    return IADPAgent(1, 1, IADPConfig(**options))


def aaindi(**kwargs):
    options = dict(
        dt=0.01,
        G_init=np.array([[2.0]]),
        sensor_cutoff_hz=2.0,
        enable_bias_correction=False,
        vff_cov_init=1e4,
        vff_forgetting_min=1.0,
        vff_forgetting_max=1.0,
        ref_error_kp=0.6,
        u_rate_limit=10000.0,
    )
    options.update(kwargs)
    return AAINDIAgent(1, 1, AAINDIConfig(**options))


def test_iadp_excitation_obeys_rate_limit_from_first_step():
    agent = iadp()
    previous = np.zeros(1)
    for k in range(3):
        command = agent.predict(np.array([0.2]), np.zeros(1), k)
        assert np.max(abs(command - previous)) <= 0.1 + 1e-12
        agent.learn(np.array([0.2]), np.zeros(1), k)
        previous = command


def test_iadp_first_transition_does_not_fabricate_state_increment():
    agent = iadp(model_learning_only_steps=0)
    for _ in range(2):
        theta, covariance = agent.rls.theta.copy(), agent.rls.Phi.copy()
        agent.predict(np.array([4.0]), np.zeros(1))
        agent.learn(np.array([3.2]), np.zeros(1))
        np.testing.assert_array_equal(agent.rls.theta, theta)
        np.testing.assert_array_equal(agent.rls.Phi, covariance)
        assert len(agent._window) == 1  # The valid absolute transition still trains P.
        agent.reset()


def test_iadp_identifies_actual_actuator_not_requested_command():
    rng = np.random.default_rng(123)
    schedule = rng.uniform(-1, 1, (1000, 1))
    agent = iadp(
        excitation_signal=schedule, u_rate_limit=1000, gamma_rls=1.0, phi_init=1e6
    )
    state, applied = np.array([0.3]), np.zeros(1)
    for k in range(1000):
        command = agent.predict(state, np.zeros(1), k)
        applied = np.clip(command, applied - 0.02, applied + 0.02)
        next_state = 0.8 * state + 0.5 * applied
        expected_cost = state[0] ** 2 + applied[0] ** 2
        metrics = agent.learn(next_state, np.zeros(1), k, applied_action=applied)
        state = next_state
    assert agent.F[0, 0] == pytest.approx(0.8, abs=1e-3)
    assert agent.G[0, 0] == pytest.approx(0.5, abs=1e-3)
    assert metrics["cost"] == pytest.approx(expected_cost)


def test_aaindi_differentiates_the_first_real_transition():
    agent = aaindi(sensor_cutoff_hz=50)
    agent.predict(np.array([1.0]), np.zeros(1))
    agent.learn(np.array([1.03]), np.zeros(1))
    np.testing.assert_allclose(agent.deriv.last_output, [3.0])


@pytest.mark.parametrize("cutoff", [1.0, 3.0, 10.0])
def test_aaindi_identification_matches_the_measurement_filter(cutoff):
    agent = aaindi(sensor_cutoff_hz=cutoff, G_init=np.array([[0.5]]))
    rng = np.random.default_rng(17)
    state = np.zeros(1)
    for k in range(1000):
        agent.predict(state, np.zeros(1), k)
        applied = rng.uniform(-1, 1, 1)
        # Isolate identification while using the public transition API.
        agent._last_u_cmd = applied.copy()
        state = state + agent.cfg.dt * 2 * applied
        agent.learn(state, np.zeros(1), k)
    assert agent.rls.G[0, 0] == pytest.approx(2.0, abs=0.002)


def test_aaindi_control_uses_synchronized_baseline():
    agent = aaindi(ref_error_kp=0)
    state = np.zeros(1)
    agent.predict(state, np.zeros(1))
    agent._last_u_cmd = np.array([1.0])
    state += 0.02
    agent.learn(state, np.zeros(1))
    # nu=0, q_dot_filtered = G * u_filtered: cancellation should request neutral.
    np.testing.assert_allclose(agent.predict(state, np.zeros(1)), [0.0], atol=1e-12)


def test_aaindi_uses_applied_input_feedback():
    agent = aaindi()
    agent.predict(np.zeros(1), np.ones(1))
    agent.learn(np.array([0.002]), np.ones(1), applied_action=np.array([0.1]))
    np.testing.assert_allclose(agent._u_prev, [0.1])


def test_aaindi_pi_uses_bias_corrected_measurement():
    agent = aaindi(enable_bias_correction=True, ref_error_kp=1.0)
    agent.bias_est._bias[:] = 0.2
    command = agent.predict(np.array([0.2]), np.zeros(1))
    np.testing.assert_allclose(command, [0], atol=1e-14)


@pytest.mark.parametrize("pending", [False, True])
def test_aaindi_checkpoint_replays_identification_and_control(tmp_path, pending):
    agent = aaindi()
    state = np.zeros(1)
    reference = np.ones(1) * 0.2
    for k in range(15):
        u = agent.predict(state, reference, k)
        state += 0.01 * 2 * u
        agent.learn(state, reference, k)
    if pending:
        u = agent.predict(state, reference, 15)
    restored = AAINDIAgent.from_pretrained(agent.save(tmp_path))
    for k in range(15, 20):
        if not (pending and k == 15):
            u = agent.predict(state, reference, k)
            np.testing.assert_array_equal(restored.predict(state, reference, k), u)
        state += 0.01 * 2 * u
        expected = agent.learn(state, reference, k)
        assert restored.learn(state, reference, k) == expected
        np.testing.assert_array_equal(restored.rls.theta, agent.rls.theta)
        np.testing.assert_array_equal(restored.rls.P, agent.rls.P)


@pytest.mark.parametrize("factory", [iadp, aaindi])
@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [0, 1]])
def test_invalid_feedback_does_not_mutate_adaptation(factory, bad):
    agent = factory()
    agent.predict(np.zeros(1), np.zeros(1))
    theta = agent.rls.theta.copy()
    with pytest.raises(ValueError, match="applied_action"):
        agent.learn(np.zeros(1), np.zeros(1), applied_action=np.array(bad))
    np.testing.assert_array_equal(agent.rls.theta, theta)
    assert agent._step == 0


@pytest.mark.parametrize(
    "environment,expected",
    [(LinearLongitudinalB747, 0.6), (LinearLongitudinalLAPAN, 3.0)],
)
def test_linear_env_reports_applied_action_in_action_space_units(environment, expected):
    env = environment(np.zeros(4), np.zeros((1, 5)), 5, dt=0.01)
    _, _, _, _, info = env.step([25.0])
    np.testing.assert_allclose(info["applied_action"], [expected], rtol=1e-6)
    np.testing.assert_allclose(
        info["applied_action"], np.rad2deg(env.model.store_input[:, 0]), rtol=1e-6
    )
