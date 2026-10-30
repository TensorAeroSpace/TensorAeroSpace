import numpy as np
import pytest

from tensoraerospace.agent.lqr import LQRAgent, LQRMRACAgent, LQRMRACConfig


def make_agent(**settings):
    baseline = LQRAgent([[1.0]], [[1.0]], [[1.0]], [[0.1]])
    defaults = dict(
        dt=0.05,
        adaptation_gain=10.0,
        normalization=0.01,
        parameter_limit=5.0,
        adaptive_limit=2.0,
    )
    defaults.update(settings)
    return LQRMRACAgent(baseline, LQRMRACConfig(**defaults))


def rollout(agent, uncertain, steps=150):
    state = np.array([1.0])
    agent.reset(state)
    errors = []
    for _ in range(steps):
        command = agent.predict(state)
        next_state = state + (0.5 if uncertain else 1.0) * command
        if uncertain:
            next_state += 0.2
        agent.update(state, next_state)
        errors.append(abs(float(next_state[0])))
        state = next_state
    return np.asarray(errors)


def test_disabled_adaptation_exactly_matches_lqr():
    agent = make_agent(adaptation_enabled=False)
    state = np.array([0.4])
    reference = np.array([0.1])
    np.testing.assert_array_equal(
        agent.predict(state, reference), agent.baseline.predict(state, reference)
    )
    agent.update(state, np.array([0.2]), reference)
    np.testing.assert_array_equal(agent.theta, np.zeros((2, 1)))


def test_healthy_reference_model_produces_no_adaptation():
    agent = make_agent()
    state = np.array([1.0])
    agent.reset(state)
    for _ in range(10):
        command = agent.predict(state)
        next_state = agent.A @ state + agent.B @ command
        agent.update(state, next_state)
        state = next_state
    np.testing.assert_allclose(agent.theta, 0.0, atol=1e-12)
    np.testing.assert_allclose(agent.diagnostics["reference_error"], 0.0, atol=1e-12)


def test_online_adaptation_reduces_matched_uncertainty_error():
    frozen = rollout(make_agent(adaptation_enabled=False), uncertain=True)
    online = rollout(make_agent(), uncertain=True)
    assert np.sum(online[-50:] ** 2) < 0.05 * np.sum(frozen[-50:] ** 2)


def test_deadzone_prevents_small_error_update():
    agent = make_agent(deadzone=2.0)
    agent.theta[0, 0] = 0.1
    before = agent.theta.copy()
    agent.update(np.zeros(1), np.array([0.1]))
    np.testing.assert_array_equal(agent.theta, before)


def test_projection_and_adaptive_command_are_bounded():
    agent = make_agent(parameter_limit=0.01, adaptive_limit=0.02)
    state = np.array([100.0])
    agent.update(state, np.array([100.0]))
    assert np.linalg.norm(agent.theta[:, 0]) <= 0.01 + 1e-12
    assert np.linalg.norm(agent.diagnostics["adaptive_command"]) <= 0.02
    assert np.linalg.norm(agent.predict(state) - agent.baseline.predict(state)) <= 0.02


def test_reset_restores_initial_state():
    agent = make_agent()
    agent.update(np.array([1.0]), np.array([2.0]))
    agent.reset(np.array([0.3]))
    np.testing.assert_array_equal(agent.theta, np.zeros((2, 1)))
    np.testing.assert_array_equal(agent.reference_state, [0.3])


@pytest.mark.parametrize(
    "settings",
    [
        {"dt": 0},
        {"adaptation_gain": -1},
        {"deadzone": -1},
        {"normalization": 0},
        {"parameter_limit": np.nan},
        {"adaptive_limit": 0},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        make_agent(**settings)


def test_invalid_state_is_rejected():
    agent = make_agent()
    with pytest.raises(ValueError, match="state"):
        agent.predict([1.0, 2.0])
    with pytest.raises(ValueError, match="next_state"):
        agent.update([1.0], [np.nan])
