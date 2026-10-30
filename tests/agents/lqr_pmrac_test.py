from types import SimpleNamespace

import numpy as np
import pytest

from tensoraerospace.agent.lqr import LQRAgent, LQRPMRACAgent, LQRPMRACConfig


def make_agent(**settings):
    baseline = LQRAgent([[1.0]], [[1.0]], [[1.0]], [[0.1]])
    defaults = dict(
        dt=0.05,
        adaptation_gain=10.0,
        normalization=0.01,
        parameter_limit=5.0,
        adaptive_limit=2.0,
        rbf_centers=np.array([-1.0, 0.0, 1.0]),
        rbf_width=0.5,
    )
    defaults.update(settings)
    return LQRPMRACAgent(baseline, LQRPMRACConfig(**defaults))


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
    np.testing.assert_array_equal(agent.state_gains, np.zeros((1, 1)))
    np.testing.assert_array_equal(agent.theta, np.zeros((3, 1)))


def test_exact_model_keeps_both_errors_and_parameters_at_zero():
    agent = make_agent()
    state = np.array([1.0])
    agent.reset(state)
    for _ in range(10):
        command = agent.predict(state)
        next_state = agent.A @ state + agent.B @ command
        agent.update(state, next_state)
        state = next_state
    np.testing.assert_allclose(agent.diagnostics["reference_error"], 0.0, atol=1e-12)
    np.testing.assert_allclose(agent.diagnostics["prediction_error"], 0.0, atol=1e-12)
    np.testing.assert_allclose(agent.state_gains, 0.0, atol=1e-12)
    np.testing.assert_allclose(agent.theta, 0.0, atol=1e-12)


def test_predictor_follows_the_declared_difference_equation():
    agent = make_agent()
    state = np.array([0.2])
    next_state = np.array([0.5])
    reference = np.array([0.1])
    old_offset = agent.predictor_state - agent.reference_state
    offset = state - agent.reference_state
    command_input = agent.B @ (agent.K @ reference)
    expected = (
        agent.reference_dynamics @ agent.reference_state
        + command_input
        + agent.predictor_dynamics @ old_offset
        + agent.predictor_error_input @ offset
    )
    agent.update(state, next_state, reference)
    np.testing.assert_allclose(agent.predictor_state, expected)


def test_rbf_features_use_configured_centers_and_width():
    agent = make_agent()
    features = agent._features(np.array([0.0]))
    np.testing.assert_allclose(features, [np.exp(-2.0), 1.0, np.exp(-2.0)])


def test_online_adaptation_reduces_matched_uncertainty_error():
    frozen = rollout(make_agent(adaptation_enabled=False), uncertain=True)
    online = rollout(make_agent(), uncertain=True)
    assert np.sum(online[-50:] ** 2) < 0.1 * np.sum(frozen[-50:] ** 2)


def test_projection_and_adaptive_command_are_bounded():
    agent = make_agent(parameter_limit=0.01, adaptive_limit=0.02)
    state = np.array([100.0])
    agent.update(state, np.array([100.0]))
    assert np.linalg.norm(agent.state_gains[:, 0]) <= 0.01 + 1e-12
    assert np.linalg.norm(agent.theta[:, 0]) <= 0.01 + 1e-12
    assert np.linalg.norm(agent.diagnostics["adaptive_command"]) <= 0.02
    assert np.linalg.norm(agent.predict(state) - agent.baseline.predict(state)) <= 0.02


def test_reset_restores_initial_state():
    agent = make_agent()
    agent.update(np.array([1.0]), np.array([2.0]))
    agent.reset(np.array([0.3]))
    np.testing.assert_array_equal(agent.state_gains, np.zeros((1, 1)))
    np.testing.assert_array_equal(agent.theta, np.zeros((3, 1)))
    np.testing.assert_array_equal(agent.reference_state, [0.3])
    np.testing.assert_array_equal(agent.predictor_state, [0.3])


@pytest.mark.parametrize(
    "settings",
    [
        {"dt": 0},
        {"adaptation_gain": -1},
        {"deadzone": -1},
        {"normalization": 0},
        {"parameter_limit": np.nan},
        {"adaptive_limit": 0},
        {"predictor_ratio": 0},
        {"rbf_width": 0},
        {"rbf_centers": []},
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


def test_projection_bounds_the_complete_gain_vector_per_control_channel():
    baseline = SimpleNamespace(
        A=np.diag([1.0, 0.5]),
        B=np.array([[1.0], [0.0]]),
        K=np.array([[0.5, 0.0]]),
    )
    agent = LQRPMRACAgent(
        baseline,
        LQRPMRACConfig(
            dt=0.05,
            adaptation_gain=10.0,
            normalization=0.01,
            parameter_limit=0.01,
            rbf_centers=np.array([-1.0, 0.0, 1.0]),
        ),
    )
    state = np.array([100.0, 100.0])
    agent.update(state, state)
    assert np.linalg.norm(agent.state_gains[0, :]) <= 0.01 + 1e-12
    assert np.linalg.norm(agent.theta[:, 0]) <= 0.01 + 1e-12


def test_parameter_norm_combines_all_parameters_by_sum_of_squares():
    agent = make_agent()
    agent.state_gains[0, 0] = 3.0
    agent.theta[:, 0] = 4.0
    assert agent.diagnostics["parameter_norm"] == pytest.approx(np.sqrt(57.0))


def test_negative_real_reference_eigenvalue_has_a_clear_error():
    baseline = SimpleNamespace(
        A=np.array([[0.0]]), B=np.array([[1.0]]), K=np.array([[1.0]])
    )
    with pytest.raises(ValueError, match="negative real eigenvalues"):
        LQRPMRACAgent(baseline)


def test_singular_reference_dynamics_has_a_clear_error():
    baseline = SimpleNamespace(
        A=np.array([[0.0]]), B=np.array([[1.0]]), K=np.array([[0.0]])
    )
    with pytest.raises(ValueError, match="nonsingular"):
        LQRPMRACAgent(baseline)
