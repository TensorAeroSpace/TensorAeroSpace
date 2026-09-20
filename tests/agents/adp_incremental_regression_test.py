"""Numerical contracts across the identifier, costs and policy in incremental ADP."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig


def test_iadp_state_increment_uses_previous_transition_origin():
    agent = IADPAgent(n_state=1, n_control=1)
    reference = np.array([[0.1, 0.3, 0.4]])
    agent.predict(np.array([1.0]), reference, 0)
    agent.learn(np.array([2.0]), reference, 0)
    agent.predict(np.array([2.0]), reference, 1)
    np.testing.assert_allclose(agent._last_dX, [1.0, 0.2])


def test_iadp_full_online_loop_identifies_state_and_control_dynamics():
    rng = np.random.default_rng(17)
    n_steps = 400
    agent = IADPAgent(
        n_state=1,
        n_control=1,
        config=IADPConfig(
            model_learning_only_steps=n_steps,
            excitation_signal=rng.normal(0, 0.3, (n_steps, 1)),
            gamma_rls=1.0,
            phi_init=1e4,
            policy_eval_warmup_updates=n_steps + 1,
        ),
    )
    state = np.zeros(1)
    reference = np.zeros((1, n_steps + 1))
    for step in range(n_steps):
        action = agent.predict(state, reference, step)
        next_state = 0.8 * state + 0.5 * action
        agent.learn(next_state, reference, step)
        state = next_state
    assert agent.F[0, 0] == pytest.approx(0.8, abs=1e-3)
    assert agent.G[0, 0] == pytest.approx(0.5, abs=1e-3)


def make_imgdhp(scale, **kwargs):
    options = dict(
        obs_scale=(scale,),
        actor_hidden=(4,),
        critic_hidden=(4,),
        track_Q=(1.0,),
        gamma=0.0,
        warmup_steps=0,
        action_rate_penalty=0.0,
        exploration_noise_std=0.0,
        u_max=1.0,
        seed=17,
        device="cpu",
    )
    options.update(kwargs)
    agent = IMGDHPAgent(1, 1, config=IMGDHPConfig(**options))
    with torch.no_grad():
        for network in (agent.actor, agent.critic, agent.target_critic):
            for parameter in network.parameters():
                parameter.zero_()
    return agent


@pytest.mark.parametrize("scale", [1.0, 10.0, 0.1])
def test_imgdhp_identifier_recovers_physical_matrices_despite_network_scaling(scale):
    agent = make_imgdhp(
        scale,
        warmup_steps=500,
        exploration_noise_std=0.3,
        forgetting=1.0,
        cov_init=1e4,
    )
    state = np.zeros(1)
    reference = np.zeros((1, 401))
    for step in range(400):
        action = agent.predict(state, reference, step)
        next_state = 0.8 * state + 0.5 * action
        agent.learn(next_state, reference, step)
        state = next_state
    np.testing.assert_allclose(agent.incremental_model.A, [[0.8]], atol=1e-3)
    np.testing.assert_allclose(agent.incremental_model.B, [[0.5]], atol=1e-3)


@pytest.mark.parametrize("scale", [1.0, 10.0, 0.1])
def test_imgdhp_cost_scales_the_physical_tracking_error_once(scale, monkeypatch):
    agent = make_imgdhp(scale)
    update = MagicMock(return_value=0.0)
    monkeypatch.setattr(agent, "_critic_update", update)
    reference = np.ones((1, 3))
    observation = np.array([2.0])
    agent.predict(observation, reference, 0)
    observation[:] = 99.0  # A caller may reuse its observation buffer.
    agent.learn(np.array([3.0]), reference, 0)
    assert update.call_args.kwargs["c_now_value"] == pytest.approx(scale**2)
    np.testing.assert_allclose(update.call_args.kwargs["err_now_np"], [scale])
    np.testing.assert_allclose(agent._y_tm1, [1.0])


@pytest.mark.parametrize("scale", [1.0, 10.0, 0.1])
def test_imgdhp_costate_target_is_derivative_wrt_physical_state(scale):
    agent = make_imgdhp(scale)
    # gamma=0: J target is c=(scale*(y-r))^2 and lambda target is dc/dy.
    # With y=2, r=1: c=scale^2, dc/dy=2*scale^2, predictions initially zero.
    loss = agent._critic_update(
        agent._augment(np.array([2.0]), np.array([1.0])),
        np.array([3.0]),
        np.array([1.0]),
        scale**2,
        np.array([scale]),
    )
    expected = (
        0.5
        / (1 + agent.cfg.beta_lambda)
        * (scale**4 + agent.cfg.beta_lambda * (2 * scale**2) ** 2)
    )
    assert loss == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("scale", [1.0, 10.0, 0.1])
def test_imgdhp_zero_critic_gives_zero_actor_update(scale):
    """Eq. (67) has no independent next-step tracking/MPC term."""
    agent = make_imgdhp(scale)
    agent.incremental_model.theta[:] = [[0.0], [1.0]]
    before = [p.detach().clone() for p in agent.actor.parameters()]
    loss = agent._actor_update(
        np.array([2.0]), np.array([1.0]), np.array([1.0]), np.zeros(1), np.array([1.0])
    )
    assert loss == 0
    for actual, expected in zip(agent.actor.parameters(), before):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("reference", [np.array([1.0]), np.array([1.0, 2.0])])
def test_imgdhp_each_tracking_channel_uses_its_own_units(reference):
    agent = IMGDHPAgent(
        2,
        1,
        reference_size=len(reference),
        tracking_indices=[0, 1],
        config=IMGDHPConfig(obs_scale=(10.0, 0.1), track_Q=(1.0, 1.0), device="cpu"),
    )
    observation = np.array([2.0, 3.0])
    expected_error = observation - reference
    augmented = agent._augment(observation, reference)
    np.testing.assert_allclose(augmented, expected_error)
    np.testing.assert_allclose(agent.actor.input_scale.cpu().numpy(), [10.0, 0.1])
    augmented_t = agent._augment_torch(
        torch.as_tensor(observation, dtype=torch.float32),
        torch.as_tensor(reference, dtype=torch.float32),
    )
    np.testing.assert_allclose(augmented_t.numpy(), augmented, rtol=1e-6)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_imgdhp_actor_uses_squared_value_gradient(scale):
    """Eq. (67): dE/du=J_next * B.T * dJ_next/de_next."""
    agent = make_imgdhp(scale, gamma=0.5, track_Q=(0.0,), actor_hidden=())
    agent.incremental_model.theta[:] = [[0.0], [1.0]]

    class QuadraticCritic(torch.nn.Module):
        def forward(self, error):
            return 0.5 * (error - 2).square(), error - 2

    agent.critic = QuadraticCritic()
    agent.actor_opt = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    agent._actor_update(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
    assert agent.actor.head.weight[0, -1].item() == pytest.approx(0.004, abs=1e-7)
