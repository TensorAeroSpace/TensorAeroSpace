"""Adaptive identification must use aligned measurements and applied controls."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("factory", [iadp])
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
