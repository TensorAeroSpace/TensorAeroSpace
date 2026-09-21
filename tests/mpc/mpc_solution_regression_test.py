"""Check MPC against analytic optima and its advertised actuator constraints."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.mpc.mpc import MPC, MPCConstraints, MPCWeights


def controller(**kwargs):
    params = dict(
        dynamics=lambda x, u: x + u,
        state_dim=1,
        action_dim=1,
        horizon=1,
        weights=MPCWeights(Q_diag=[1.0], R_diag=[1.0], terminal_weight=0.0),
        optimizer="sgd",
        lr=0.25,
        iters=1,
        warm_start=False,
        dtype=torch.float64,
    )
    params.update(kwargs)
    return MPC(**params)


@pytest.mark.parametrize("bounded", [False, True])
def test_last_optimizer_step_reaches_analytic_quadratic_minimum(bounded):
    constraints = MPCConstraints(u_min=[-1.0], u_max=[1.0]) if bounded else None
    result = controller(constraints=constraints).solve(x0=[0.0], x_ref=[[0.0], [1.0]])
    # min (u - 1)^2 + u^2: u*=1/2, J*=1/2.
    np.testing.assert_allclose(result.u0, [0.5])
    assert result.final_cost == pytest.approx(0.5)


def test_best_iterate_keeps_the_matching_cost_before_inplace_optimizer_update():
    result = controller(lr=2.0).solve(x0=[0.0], x_ref=[[0.0], [1.0]])
    # SGD overshoots to u=4, J=25. Keep the initial u=0, J=1.
    np.testing.assert_allclose(result.u0, [0.0])
    assert result.final_cost == pytest.approx(1.0)


def test_horizon_targets_start_at_the_first_predicted_state():
    mpc = controller(horizon=3, dynamics=lambda x, u: u)
    targets = np.array([[1.0], [0.2], [-0.6]])
    result = mpc.solve(x0=[0.0], x_ref=targets)
    np.testing.assert_allclose(result.u_seq, 0.5 * targets)


@pytest.mark.parametrize("previous", [None, [0.0]])
def test_rate_limits_apply_between_predicted_actions_even_without_previous(previous):
    mpc = controller(
        horizon=3,
        iters=0,
        warm_start=True,
        constraints=MPCConstraints(du_min=[-0.1], du_max=[0.1]),
    )
    mpc._u_warm = torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float64)
    result = mpc.solve(x0=[0.0], u_prev=previous)
    seq = result.u_seq[:, 0]
    if previous is not None:
        seq = np.r_[previous, seq]
    assert np.max(np.abs(np.diff(seq))) <= 0.100001


def test_nonfinite_prediction_fails_without_overwriting_warm_start():
    mpc = controller(dynamics=lambda x, u: x + u * float("nan"))
    with pytest.raises(RuntimeError, match="finite"):
        mpc.solve(x0=[0.0], x_ref=[[0.0], [1.0]])
    assert mpc._u_warm is None


def test_nonfinite_gradient_fails_before_applying_an_action():
    mpc = controller(dynamics=lambda x, u: torch.sqrt(u))
    with pytest.raises(RuntimeError, match="finite"):
        mpc.solve(x0=[0.0], x_ref=[[0.0], [1.0]])
