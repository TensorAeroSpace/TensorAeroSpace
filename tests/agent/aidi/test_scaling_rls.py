"""ScalingRLS unit tests — convergence, fault response, consistency check."""

import numpy as np
import pytest

from tensoraerospace.agent.aidi.scaling_rls import ScalingRLS


def _onboard_G() -> np.ndarray:
    return np.array(
        [[0.10, -2.50, 0.00], [-3.00, 0.05, 0.00], [0.02, 0.00, -1.20]],
        dtype=np.float64,
    )


def test_scaling_rls_init_at_unity():
    rls = ScalingRLS(n_y=3, n_u=3)
    np.testing.assert_array_equal(rls.theta, np.ones((3, 3)))


def test_scaling_rls_converges_to_unity_when_truth_matches_onboard():
    rng = np.random.default_rng(0)
    G = _onboard_G()
    rls = ScalingRLS(
        n_y=3, n_u=3, sigma0=1e-3, memory_length=50, consistency_threshold=10.0
    )

    for _ in range(400):
        du = rng.normal(0.0, 1.0, size=3)
        domega = G @ du + rng.normal(0.0, 1e-4, size=3)
        rls.update(du, domega, G)

    np.testing.assert_allclose(rls.theta, np.ones((3, 3)), atol=0.1)


def test_scaling_rls_converges_to_truth_when_one_surface_lost():
    rng = np.random.default_rng(1)
    G_onboard = _onboard_G()
    truth_scale = np.ones((3, 3))
    truth_scale[:, 0] = 0.25
    G_true = truth_scale * G_onboard
    rls = ScalingRLS(
        n_y=3, n_u=3, sigma0=1e-3, memory_length=50, consistency_threshold=10.0
    )

    for _ in range(800):
        du = rng.normal(0.0, 1.0, size=3)
        domega = G_true @ du + rng.normal(0.0, 1e-4, size=3)
        rls.update(du, domega, G_onboard)

    np.testing.assert_allclose(rls.theta[:, 0], 0.25 * np.ones(3), atol=0.15)
    np.testing.assert_allclose(rls.theta[:, 1:], 1.0, atol=0.15)


def test_scaling_rls_lambda_drops_after_step_fault():
    rng = np.random.default_rng(2)
    G = _onboard_G()
    rls = ScalingRLS(
        n_y=3,
        n_u=3,
        sigma0=1e-3,
        memory_length=50,
        lambda_min=0.5,
        consistency_threshold=10.0,
    )

    for _ in range(200):
        du = rng.normal(0.0, 0.5, size=3)
        rls.update(du, G @ du, G)

    lam_pre = rls.last_lambda.copy()
    fault_truth = np.array([[1.0, 0.3, 1.0]] * 3, dtype=np.float64) * G

    for _ in range(5):
        du = rng.normal(0.0, 1.0, size=3)
        rls.update(du, fault_truth @ du, G)

    assert float(np.min(rls.last_lambda)) < float(np.min(lam_pre))


def test_scaling_rls_consistency_check_collapses_outlier_row():
    rls = ScalingRLS(
        n_y=3, n_u=3, sigma0=1.0, memory_length=10, consistency_threshold=1e-6
    )
    delta_theta_in = np.array(
        [[0.10, 0.00, 0.00], [0.20, 0.00, 0.00], [0.30, 0.00, 0.00]],
        dtype=np.float64,
    )
    out = rls._apply_consistency_check(delta_theta_in)
    expected_col0 = np.full(3, 0.20)
    np.testing.assert_allclose(out[:, 0], expected_col0)
    np.testing.assert_array_equal(out[:, 1:], 0.0)


def test_scaling_rls_rejects_wrong_shape():
    rls = ScalingRLS(n_y=3, n_u=3)
    with pytest.raises(ValueError):
        rls.update(np.zeros(2), np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        rls.update(np.zeros(3), np.zeros(2), np.eye(3))
    with pytest.raises(ValueError):
        rls.update(np.zeros(3), np.zeros(3), np.eye(2))
