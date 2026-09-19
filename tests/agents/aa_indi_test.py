"""Independent VFF-RLS component tests for AA-INDI."""

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi import VFFRLSEstimator


def test_vff_rls_converges_on_known_linear_system():
    """Synthetic identification task: Δy = G_true · Δu with white-noise Δu."""
    rng = np.random.default_rng(0)
    G_true = np.array([[-2.0, 0.1], [0.05, -1.5]], dtype=np.float64)
    rls = VFFRLSEstimator(
        n_y=2,
        n_u=2,
        forgetting_min=0.8,
        forgetting_max=0.9999,
        eps_sensitivity=5.0,
        cov_init=1e3,
        seed=0,
    )
    for _ in range(500):
        du = rng.normal(size=2) * 0.5
        dy = G_true @ du + rng.normal(size=2) * 1e-3
        rls.update(du, dy)
    np.testing.assert_allclose(rls.G, G_true, atol=5e-2)


def test_vff_rls_lambda_drops_on_large_residuals():
    """Large residuals should pull the forgetting factor toward the lower bound.

    Only the very first update sees the full residual; by the second call the
    linear estimator has already nearly fit it, so we assert the contraction
    on the initial tick.
    """
    rls = VFFRLSEstimator(
        n_y=1,
        n_u=1,
        forgetting_min=0.5,
        forgetting_max=0.999,
        eps_sensitivity=0.1,
    )
    rls.update(np.array([1.0]), np.array([10.0]))
    assert rls.last_lambda <= 0.8
    # After the residual collapses, λ should relax back toward the upper bound.
    for _ in range(20):
        rls.update(np.array([1.0]), np.array([10.0]))
    assert rls.last_lambda >= 0.9


def test_vff_rls_validates_inputs():
    with pytest.raises(ValueError, match="forgetting"):
        VFFRLSEstimator(n_y=1, n_u=1, forgetting_min=1.5, forgetting_max=1.0)
    with pytest.raises(ValueError, match="eps_sensitivity"):
        VFFRLSEstimator(n_y=1, n_u=1, eps_sensitivity=0.0)
    rls = VFFRLSEstimator(n_y=2, n_u=2)
    with pytest.raises(ValueError, match="du"):
        rls.update(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="dy"):
        rls.update(np.zeros(2), np.zeros(3))


def test_vff_rls_reset_covariance_restores_init_scale():
    rls = VFFRLSEstimator(n_y=1, n_u=1, cov_init=50.0)
    rls.update(np.array([1.0]), np.array([0.5]))
    assert not np.allclose(rls.P, 50.0 * np.eye(1))
    rls.reset_covariance()
    np.testing.assert_allclose(rls.P, 50.0 * np.eye(1))


def test_vff_rls_predict_uses_latest_theta():
    rls = VFFRLSEstimator(n_y=1, n_u=1)
    rls.theta = np.array([[0.3]])
    np.testing.assert_allclose(rls.predict(np.array([2.0])), [0.6])
