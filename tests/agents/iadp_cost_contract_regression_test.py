"""Cost admissibility in Konatala (2024), Eqs. (3), (5), and (11)."""

import numpy as np
import pytest

from tensoraerospace.agent.iadp import IADPAgent, IADPConfig


@pytest.mark.parametrize("name", ["Q", "R"])
@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[-1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, 0.5], [0.0, 1.0]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_invalid_quadratic_cost_rejected(name, matrix):
    with pytest.raises(ValueError, match=name):
        IADPAgent(2, 2, IADPConfig(**{name: matrix}))


@pytest.mark.parametrize("gamma", [-0.1, 0.0, 1.0, 1.01, np.inf, np.nan])
def test_invalid_discount_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        IADPAgent(1, 1, IADPConfig(gamma=gamma))


def test_semidefinite_tracking_cost_and_zero_input_weight_are_allowed():
    q = np.diag([0.0, 1.0])
    r = np.zeros((1, 1))
    agent = IADPAgent(2, 1, IADPConfig(Q=q, R=r))
    np.testing.assert_array_equal(agent.Q, q)
    np.testing.assert_array_equal(agent.R, r)


def test_policy_increment_minimizes_admissible_quadratic_cost():
    rng = np.random.default_rng(341)
    f = rng.normal(size=(4, 4)) * 0.1
    g = rng.normal(size=(4, 2))
    z = rng.normal(size=(4, 4))
    p = z.T @ z + np.eye(4)
    r = np.array([[2.0, 0.5], [0.5, 1.0]])
    agent = IADPAgent(2, 2, IADPConfig(F_init=f, G_init=g, P_init=p, R=r))
    x, dx = rng.normal(size=(2, 4))
    agent._delta_prev = np.array([0.1, -0.2])
    du = agent._compute_policy_increment(x, dx)

    def cost(action):
        u = agent._delta_prev + action
        following = x + f @ dx + g @ action
        return u @ r @ u + agent.cfg.gamma * following @ p @ following

    # Independent finite-difference stationary-point and neighborhood check.
    eps = 1e-5
    for axis in np.eye(2):
        derivative = (cost(du + eps * axis) - cost(du - eps * axis)) / (2 * eps)
        assert derivative == pytest.approx(0, abs=1e-8)
    for offset in rng.normal(size=(30, 2)) * 0.1:
        assert cost(du + offset) >= cost(du) - 1e-12


def test_frozen_critic_skips_ill_scaled_fitting_problem():
    agent = IADPAgent(1, 1, IADPConfig(policy_eval_blend=0.0))
    previous = agent.P.copy()
    for _ in range(4):
        agent._window.append(dict(X=np.full(2, 1e160), Xnext=np.zeros(2), cost=1.0))
    with np.errstate(over="raise", invalid="raise"):
        agent._policy_evaluation()
    np.testing.assert_array_equal(agent.P, previous)
