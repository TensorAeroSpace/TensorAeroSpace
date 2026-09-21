"""Independent batch-fit oracles and unexcited initialization limits."""

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi import VFFRLSEstimator
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig, IncrementalRLS


@pytest.mark.parametrize("kind", ["iadp", "aaindi"])
@pytest.mark.parametrize("n_output", [1, 3])
def test_recursive_identifier_matches_independent_weighted_batch_fit(kind, n_output):
    """Check 60 updates against one batch solve, including the prior.

    Fixed forgetting discounts history before adding the current sample.
    Atmaca (54)--(57) discounts the posterior after adding it. Hence the
    current sample has weight lambda in AA-INDI, but weight one in iADP.
    The multi-output AA-INDI case checks the library's shared-lambda extension.
    """
    rng = np.random.default_rng(72686)
    n_regressor, count, prior_variance = 4, 60, 3.0
    design = rng.normal(size=(count, n_regressor))
    targets = design @ rng.normal(size=(n_regressor, n_output))
    targets += rng.normal(scale=0.2, size=targets.shape)
    prior = rng.normal(scale=0.1, size=(n_regressor, n_output))
    if kind == "iadp":
        estimator = IncrementalRLS(
            n_output, n_regressor, gamma_rls=0.97, phi_init=prior_variance
        )
    else:
        estimator = VFFRLSEstimator(
            n_output,
            n_regressor,
            cov_init=prior_variance,
            forgetting_min=0.25,
            forgetting_max=1.0,
            eps_sensitivity=np.sqrt(15.0),
            seed=0,
        )
    estimator.theta[:] = prior
    forgetting = []
    for row, target in zip(design, targets):
        estimator.update(row, target)
        forgetting.append(
            estimator.gamma_rls if kind == "iadp" else estimator.last_lambda
        )

    if kind == "iadp":
        weights = 0.97 ** np.arange(count - 1, -1, -1)
        covariance = estimator.Phi
    else:
        weights = np.cumprod(np.asarray(forgetting)[::-1])[::-1]
        covariance = estimator.P
        assert np.ptp(forgetting) > 0.01  # Exercise genuinely variable forgetting.
    prior_weight = np.prod(forgetting) / prior_variance
    batch_design = np.vstack(
        [
            np.sqrt(weights)[:, None] * design,
            np.sqrt(prior_weight) * np.eye(n_regressor),
        ]
    )
    batch_targets = np.vstack(
        [
            np.sqrt(weights)[:, None] * targets,
            np.sqrt(prior_weight) * prior,
        ]
    )
    batch_theta = np.linalg.lstsq(batch_design, batch_targets, rcond=None)[0]
    batch_covariance = np.linalg.solve(
        batch_design.T @ batch_design, np.eye(n_regressor)
    )
    np.testing.assert_allclose(estimator.theta, batch_theta, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(covariance, batch_covariance, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("warm_start_model", [False, True])
def test_iadp_uncoupled_value_without_excitation_cannot_start_tracking(
    warm_start_model,
):
    """Even exact F/G cannot connect a reference to the policy with P=I."""
    config = IADPConfig(P_init=np.eye(2))
    if warm_start_model:
        config.F_init = np.diag([0.9, 1.0])
        config.G_init = np.array([[0.1], [0.0]])
    agent = IADPAgent(1, 1, config)
    initial_g = agent.G.copy()
    state, reference = np.zeros(1), np.ones(1)
    for k in range(200):
        command = agent.predict(state, reference, k)
        state = 0.9 * state + 0.1 * command
        agent.learn(state, reference, k, applied_action=command)
        np.testing.assert_array_equal(command, [0.0])
    np.testing.assert_array_equal(agent.G, initial_g)
    np.testing.assert_array_equal(state, [0.0])
    assert agent.rls.num_updates == 199


def test_iadp_tracking_value_couples_reference_to_control():
    """A tracking-shaped value starts control on the same exact scalar plant."""
    agent = IADPAgent(
        1,
        1,
        IADPConfig(
            F_init=np.diag([0.9, 1.0]),
            G_init=np.array([[0.1], [0.0]]),
            P_init=np.array([[1.0, -1.0], [-1.0, 1.0]]),
        ),
    )
    command = agent.predict(np.zeros(1), np.ones(1))
    # The positive plant gain requires positive input for the positive target.
    assert command[0] > 0.0
