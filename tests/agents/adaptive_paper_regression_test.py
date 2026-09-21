"""Equation-based regressions for Konatala (2024) and Atmaca (2026).

Konatala Fig. 2 supplies the model bootstrap and fixed-forgetting recursion.
Atmaca Eqs. 54–57 supply gain, residual-weighted forgetting and covariance.
"""

import copy

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi import VFFRLSEstimator
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig, IncrementalRLS


def estimator(kind, scale=100.0):
    if kind == "iadp":
        return IncrementalRLS(1, 1, gamma_rls=1.0, phi_init=scale)
    return VFFRLSEstimator(
        1, 1, forgetting_min=1.0, forgetting_max=1.0, cov_init=scale, seed=0
    )


def covariance(agent):
    return agent.Phi if isinstance(agent, IncrementalRLS) else agent.P


@pytest.mark.parametrize("initial_variance", [1e16, 1e20])
@pytest.mark.parametrize("kind", ["iadp", "aaindi"])
def test_rls_retains_information_after_large_prior(kind, initial_variance):
    agent = estimator(kind, initial_variance)
    agent.theta[:] = 0
    agent.update(np.ones(1), np.ones(1))
    # Independent scalar posterior precision: 1/P_new = 1/P_old + phi**2.
    expected = 1 / (1 / initial_variance + 1)
    assert covariance(agent)[0, 0] == pytest.approx(expected, rel=1e-12)
    agent.update(np.ones(1), np.array([3.0]))
    assert agent.theta[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_aaindi_rls_matches_original_gain_and_forgetting_equations():
    agent = VFFRLSEstimator(
        1,
        2,
        forgetting_min=0.25,
        forgetting_max=1.0,
        eps_sensitivity=2.0,
        cov_init=2.0,
        seed=0,
    )
    agent.theta[:] = 0
    phi = np.array([1.0, 0.5])
    # Atmaca Eq. 54: K = P a.T / (1 + a P a.T), *before* forgetting.
    expected_gain = np.array([4 / 7, 2 / 7])
    # Eq. 55: Sigma_0 = eps_sensitivity**2 = 4, residual = 3.
    expected_lambda = 1 - (2 / 7) * 9 / 4
    expected_covariance = (
        2 * np.eye(2) - np.outer(expected_gain, 2 * phi)
    ) / expected_lambda
    agent.update(phi, np.array([3.0]))
    np.testing.assert_allclose(agent.theta[:, 0], 3 * expected_gain, rtol=1e-12)
    assert agent.last_lambda == pytest.approx(expected_lambda)
    np.testing.assert_allclose(agent.P, expected_covariance, rtol=1e-12)


@pytest.mark.parametrize("kind", ["iadp", "aaindi"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_regressor", [False, True])
def test_nonfinite_identification_sample_does_not_mutate_estimator(
    kind, bad, bad_regressor
):
    agent = estimator(kind)
    previous = copy.deepcopy(agent.__dict__)
    phi = np.array([bad if bad_regressor else 1.0])
    target = np.array([1.0 if bad_regressor else bad])
    with pytest.raises(ValueError, match="finite"):
        agent.update(phi, target)
    for key, value in previous.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(agent.__dict__[key], value)
        else:
            assert agent.__dict__[key] == value


@pytest.mark.parametrize("kind", ["iadp", "aaindi"])
@pytest.mark.parametrize("scale", [np.nan, np.inf, -1.0])
def test_invalid_prior_covariance_rejected(kind, scale):
    with pytest.raises(ValueError):
        estimator(kind, scale)


@pytest.mark.parametrize("kind", ["iadp", "aaindi"])
def test_covariance_overflow_raises_before_poisoning_state(kind):
    agent = (
        IncrementalRLS(1, 2, gamma_rls=0.7)
        if kind == "iadp"
        else VFFRLSEstimator(1, 2, forgetting_min=0.7, forgetting_max=0.7, seed=0)
    )
    # No excitation in the second direction. Standard forgetting can wind up;
    # the implementation must report numerical failure without storing inf/NaN.
    with pytest.raises(FloatingPointError):
        for _ in range(2500):
            agent.update(np.array([1.0, 0.0]), np.array([2.0]))
    assert np.isfinite(covariance(agent)).all()
    assert np.isfinite(agent.theta).all()


def test_iadp_critic_uses_model_prediction_as_in_original_fig2():
    agent = IADPAgent(
        1,
        1,
        IADPConfig(
            F_init=np.diag([0.9, 1.0]),
            G_init=np.zeros((2, 1)),
            phi_init=1e-8,
            gamma_rls=1.0,
            policy_eval_warmup_updates=1000,
        ),
    )
    agent.predict(np.array([1.0]), np.zeros(1), 0)
    agent.learn(np.array([0.9]), np.zeros(1), 0)
    agent.predict(np.array([0.9]), np.zeros(1), 1)
    agent.learn(np.array([1.01]), np.zeros(1), 1)  # True next .81 plus sensor error.
    assert agent._window[-1]["Xnext"][0] == pytest.approx(0.81, abs=1e-8)
    # Actual observations must still drive the identifier and current-state cost.
    assert agent.rls.last_residual[0] == pytest.approx(0.2, abs=1e-8)
    assert agent._window[-1]["cost"] == pytest.approx(0.9**2)


def test_iadp_unregularized_ls_keeps_resolvable_small_feature():
    agent = IADPAgent(1, 1, IADPConfig())
    expected = np.diag([5.0, 2.0])
    for state in [
        np.array([sign * s, x])
        for s, x in [(1e-5, 1.0), (2e-5, 0.8), (3e-5, 0.5)]
        for sign in [-1, 1]
    ]:
        agent._window.append(
            dict(X=state, Xnext=np.zeros(2), cost=float(state @ expected @ state))
        )
    agent._policy_evaluation()
    np.testing.assert_allclose(agent.P, expected, atol=1e-5)


def test_iadp_model_only_phase_does_not_train_or_fill_critic():
    agent = IADPAgent(
        1,
        1,
        IADPConfig(
            dt=0.1,
            model_learning_only_steps=20,
            excitation_signal=np.tile(np.array([[0.2], [-0.1]]), (10, 1)),
            policy_eval_warmup_updates=1,
            policy_eval_every=1,
            P_init=np.array([[4.0, 1.0], [1.0, 3.0]]),
            F_init=np.diag([0.8, 1.0]),
            G_init=np.array([[0.5], [0.0]]),
        ),
    )
    prior = agent.P.copy()
    state = np.array([0.3])
    ref = np.array([0.1])
    for k in range(20):
        command = agent.predict(state, ref, k)
        state = 0.8 * state + 0.5 * command
        agent.learn(state, ref, k)
    assert agent.rls.num_updates == 19
    np.testing.assert_array_equal(agent.P, prior)
    assert len(agent._window) == 0
    for k in range(20, 40):
        command = agent.predict(state, ref, k)
        state = 0.8 * state + 0.5 * command
        agent.learn(state, ref, k)
    assert len(agent._window) == 20
    assert not np.allclose(agent.P, prior)
