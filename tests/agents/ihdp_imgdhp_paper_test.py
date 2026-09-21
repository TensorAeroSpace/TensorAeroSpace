"""Independent equations from Zhou (2016) and Sun & van Kampen (2021)."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ihdp.Actor import Actor
from tensoraerospace.agent.ihdp.Critic import Critic
from tensoraerospace.agent.ihdp.Incremental_model import IncrementalModel
from tensoraerospace.agent.im_gdhp import (
    GDHPCritic,
    IMGDHPAgent,
    IMGDHPConfig,
    IncrementalModelRLS,
)


@pytest.mark.parametrize("activation,gain", [("tanh", 7.0), ("sigmoid", 14.0)])
def test_ihdp_physical_action_gradient_matches_autograd(activation, gain):
    actor = Actor(
        ["u"],
        ["alpha"],
        ["alpha"],
        [0],
        10,
        0,
        layers=(3, 1),
        activations=("tanh", activation),
        maximum_input=7.0,
        NN_initial=8,
    )
    actor.build_actor_model()
    actor.run_actor_online(np.array([[0.2]]), np.array([[0.1]]))
    # Independent loss: J(x_next)=2+3*x_next; x_next=.4*u.
    x = torch.tensor([[0.1]])
    output = actor.model(x)
    u = gain * output - (7 if activation == "sigmoid" else 0)
    j = 2 + 3 * 0.4 * u
    direct = torch.autograd.grad(
        0.5 * j.square().sum(), tuple(actor.model.parameters())
    )
    de_du = np.array([float(j.detach().item()) * 3 * 0.4])
    for expected, raw in zip(direct, actor.dut_dWb):
        np.testing.assert_allclose(
            actor._weighted_param_gradient(de_du, raw),
            expected.numpy(),
            rtol=2e-6,
            atol=1e-6,
        )


def test_ihdp_identification_retains_weak_observable_channel(monkeypatch):
    model = IncrementalModel(["x"], ["u"], 20)
    a = np.column_stack([np.ones(4), np.array([-2, -1, 1, 2]) * 1e-8])
    target = a @ np.array([[0.8], [0.5]])
    monkeypatch.setattr(model, "build_A_LS_matrix", lambda: a)
    monkeypatch.setattr(model, "build_x_LS_vector", lambda: target)
    model.identify_incremental_model_LS(np.zeros(1), np.zeros(1))
    np.testing.assert_allclose(model.F, [[0.8]], atol=1e-8)
    np.testing.assert_allclose(model.G, [[0.5]], atol=1e-8)


def test_ihdp_quadratic_cost_preserves_cross_terms():
    q = np.array([[2, 0.3], [0.3, 4]])
    critic = Critic(q, ["x", "y"], ["x", "y"], [0, 1], 10, 0)
    critic.xt = np.array([[2.0], [3.0]])
    critic.xt_ref = np.array([[0.5], [-1.0]])
    e = np.array([1.5, 4.0])
    assert critic.c_computation().item() == pytest.approx(e @ q @ e)


def test_analytic_critic_matches_finite_differences_and_mixed_gradient():
    torch.manual_seed(4)
    critic = GDHPCritic(2, 2, (3,), input_scale=[10.0, 0.3]).double()
    x = torch.tensor([0.1, -0.3], dtype=torch.float64, requires_grad=True)
    j, lam = critic(x)
    for i in range(2):
        delta = torch.zeros_like(x)
        delta[i] = 1e-5
        finite = (critic(x + delta)[0] - critic(x - delta)[0]) / (2e-5)
        assert lam[i].item() == pytest.approx(finite.item(), rel=1e-7, abs=1e-8)
    assert critic(torch.zeros_like(x))[0].item() == 0
    # Costate fitting must reach the same scalar-value parameters.
    grads = torch.autograd.grad(lam.square().sum(), tuple(critic.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)
    with torch.no_grad():
        _, inferred = critic(x.detach())
    torch.testing.assert_close(inferred, lam.detach())


def test_imgdhp_costate_target_contains_both_paths_and_identity():
    cfg = IMGDHPConfig(
        actor_hidden=(),
        critic_hidden=(),
        track_Q=(2.0,),
        control_R=(0.3,),
        gamma=0.8,
        beta_lambda=3.0,
        seed=0,
    )
    agent = IMGDHPAgent(1, 1, config=cfg)
    with torch.no_grad():
        agent.actor.head.weight.fill_(0.02)
        agent.actor.head.weight[:, -1].fill_(1.0)
        agent.critic.j_head.weight.fill_(0.4)
    agent.incremental_model.theta[:] = [[0.7], [0.2]]
    e, ep = 0.3, 0.5
    z = 0.02 * e + 0.01
    u = cfg.u_max * np.tanh(z)
    du = cfg.u_max * 0.02 / (np.cosh(z) ** 2)
    cost = 2 * e**2 + 0.3 * u**2
    dc = 4 * e + du * 0.6 * u
    target_lambda = dc + 0.8 * (1 + 0.7 + 0.2 * du) * 0.4
    residual_j = 0.4 * e - cost - 0.8 * 0.4 * ep
    expected = 0.5 * (0.25 * residual_j**2 + 0.75 * (0.4 - target_lambda) ** 2)
    loss = agent._critic_update(
        np.array([e]), np.array([ep]), None, cost, np.array([e])
    )
    assert loss == pytest.approx(expected, rel=1e-6)


def test_history_identifier_recovers_hidden_second_order_dynamics():
    model = IncrementalModelRLS(
        1, 1, history_length=2, forgetting=1, cov_init=1e7, theta_init_scale=0
    )
    rng = np.random.default_rng(3)
    y = [np.zeros(1), np.zeros(1)]
    u = [np.zeros(1)]
    for _ in range(500):
        action = rng.normal(size=1)
        following = 1.1 * y[-1] - 0.28 * y[-2] + 0.2 * action + 0.1 * u[-1]
        model.update(y[-2], y[-1], following, u[-1], action)
        y.append(following)
        u.append(action)
    np.testing.assert_allclose(model.A, [[1.1, -0.28]], atol=1e-5)
    np.testing.assert_allclose(model.B, [[0.2, 0.1]], atol=1e-5)
    assert model.num_updates == 499


@pytest.mark.parametrize("cov_init", [100.0, (10000.0,) * 3 + (100.0,) * 3])
def test_multistep_identifier_checkpoint_resumes_exactly(tmp_path, cov_init):
    agent = IMGDHPAgent(
        1,
        1,
        config=IMGDHPConfig(
            history_length=3,
            cov_init=cov_init,
            actor_bias_input=0.25,
            seed=2,
            exploration_noise_std=0.1,
            actor_lr_decay=0.99,
            critic_lr_decay=0.98,
        ),
    )
    ref = np.zeros((1, 30))
    state = np.array([0.1])
    for k in range(10):
        action = agent.predict(state, ref, k)
        state = 0.8 * state + 0.2 * action
        agent.learn(state, ref, k)
    restored = IMGDHPAgent.from_pretrained(
        agent.save(tmp_path, save_gradients=True), load_gradients=True
    )
    assert restored.actor.bias_input == 0.25
    for k in range(10, 15):
        action = agent.predict(state, ref, k)
        np.testing.assert_array_equal(action, restored.predict(state, ref, k))
        state = 0.8 * state + 0.2 * action
        first = agent.learn(state, ref, k)
        second = restored.learn(state, ref, k)
        assert first == second
        np.testing.assert_array_equal(
            agent.incremental_model.theta, restored.incremental_model.theta
        )


def test_legacy_imgdhp_checkpoint_requires_retraining(tmp_path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps({"policy": {"params": {}, "config": {}}})
    )
    with pytest.raises(ValueError, match="require retraining"):
        IMGDHPAgent.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actor_lr_decay": 1.1},
        {"critic_lr_min": 1.0},
        {"actor_lr": float("nan")},
        {"weight_limit": 0},
    ],
)
def test_imgdhp_rejects_invalid_learning_schedule(kwargs):
    with pytest.raises(ValueError):
        IMGDHPAgent(1, 1, config=IMGDHPConfig(**kwargs))


def test_input_cost_and_its_derivative_use_the_same_applied_command(monkeypatch):
    from unittest.mock import Mock

    agent = IMGDHPAgent(
        1,
        1,
        config=IMGDHPConfig(
            warmup_steps=0,
            critic_only_steps=100,
            control_R=(0.3,),
            seed=0,
            exploration_noise_std=0.2,
        ),
    )
    critic_step = Mock(return_value=0.0)
    monkeypatch.setattr(agent, "_critic_update", critic_step)
    reference = np.zeros((1, 2))
    agent.predict(np.array([0.1]), reference, 0)
    # The caller changes the requested command; cost and derivative must both
    # describe this measured input, not an unperturbed policy evaluation.
    agent.learn(np.array([0.09]), reference, 0, applied_action=np.array([0.7]))
    values = critic_step.call_args.kwargs
    assert values["c_now_value"] == pytest.approx(0.1**2 + 0.3 * 0.7**2)
    assert values["policy_action"].item() == pytest.approx(0.7)


@pytest.mark.parametrize("bias_input", [0.01, 0.4])
def test_actor_constant_input_controls_trim_value_and_gradient(bias_input):
    from tensoraerospace.agent.im_gdhp.networks import GDHPActor

    actor = GDHPActor(1, 1, hidden_sizes=(), u_max=2, bias_input=bias_input).double()
    with torch.no_grad():
        actor.head.weight[:] = torch.tensor([[0.7, 0.5]])
    result = actor(torch.zeros(1, dtype=torch.float64))
    expected = 2 * np.tanh(0.5 * bias_input)
    assert result.item() == pytest.approx(expected)
    gradient = torch.autograd.grad(result.sum(), actor.head.weight)[0]
    expected_gradient = 2 * bias_input / np.cosh(0.5 * bias_input) ** 2
    assert gradient[0, 1].item() == pytest.approx(expected_gradient)


@pytest.mark.parametrize("bias_input", [0, -1, float("nan"), float("inf")])
def test_actor_rejects_invalid_constant_input(bias_input):
    with pytest.raises(ValueError, match="bias_input"):
        IMGDHPAgent(1, 1, config=IMGDHPConfig(actor_bias_input=bias_input))
