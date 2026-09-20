"""RLS covariance must account for the units of each regressor."""

import numpy as np
import pytest

from tensoraerospace.agent.im_gdhp import IncrementalModelRLS


@pytest.mark.parametrize("covariance", [0, -1, np.inf, [1, 2], [1, 2, 3, 4, 5, -1]])
def test_invalid_prior_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="cov_init"):
        IncrementalModelRLS(2, 1, history_length=2, cov_init=covariance)


def test_scaled_prior_is_equivalent_to_identification_in_normalized_units():
    state_scale = np.array([180 / np.pi, 10.0])
    regressor_scale = np.r_[np.tile(state_scale, 2), np.ones(2)]
    diagonal = 2000 * regressor_scale**2
    physical = IncrementalModelRLS(
        2, 1, history_length=2, cov_init=diagonal, theta_init_scale=0
    )
    normalized = IncrementalModelRLS(
        2, 1, history_length=2, cov_init=2000, theta_init_scale=0
    )
    rng = np.random.default_rng(4)
    previous, current, previous_action = np.zeros(2), np.zeros(2), np.zeros(1)
    a = np.array([[0.7, 0.1], [-0.2, 0.5]])
    b = np.array([-0.002, 0.001])
    for _ in range(300):
        action = rng.normal(0, 0.3, size=1)
        following = a @ current + b * action[0]
        scaled_prediction = normalized.predict_next(
            current * state_scale, previous * state_scale, action, previous_action
        )
        prediction = physical.predict_next(current, previous, action, previous_action)
        np.testing.assert_allclose(
            prediction * state_scale, scaled_prediction, rtol=1e-7, atol=1e-10
        )
        physical.update(previous, current, following, previous_action, action)
        normalized.update(
            previous * state_scale,
            current * state_scale,
            following * state_scale,
            previous_action,
            action,
        )
        previous, current, previous_action = current, following, action

    np.testing.assert_allclose(
        physical.B * state_scale[:, None], normalized.B, rtol=1e-7, atol=1e-10
    )
    assert not np.allclose(np.diag(physical.P), diagonal)
    theta = physical.theta.copy()
    physical.reset_covariance()
    np.testing.assert_array_equal(physical.P, np.diag(diagonal))
    np.testing.assert_array_equal(physical.theta, theta)
