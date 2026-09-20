"""Known setpoint changes must not be identified as aircraft dynamics."""

import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig


def test_identifier_mode_is_explicit_and_paper_default_is_retained():
    assert IMGDHPConfig().identifier_mode == "tracking_error"
    with pytest.raises(ValueError, match="identifier_mode"):
        IMGDHPAgent(1, 1, config=IMGDHPConfig(identifier_mode="unknown"))


@pytest.mark.parametrize("history", [1, 3])
def test_known_reference_jumps_do_not_change_identified_plant(history):
    agent = IMGDHPAgent(
        2,
        1,
        config=IMGDHPConfig(
            identifier_mode="output",
            history_length=history,
            warmup_steps=1000,
            seed=0,
            cov_init=1e5,
            forgetting=1.0,
        ),
    )
    shifted = copy.deepcopy(agent)
    ref = np.zeros((1, 101))
    jumps = np.tile([0.0, 4.0, -7.0, 0.5], 26)[None, :101]
    rng = np.random.default_rng(5)
    state = np.array([0.2, -0.4])
    a = np.array([[0.7, 0.1], [-0.3, 0.5]])
    b = np.array([0.2, 0.1])
    for k in range(100):
        action = rng.normal(size=1)
        following = a @ state + b * action[0]
        for trial, command in ((agent, ref), (shifted, jumps)):
            trial.predict(state, command, k)
            trial.learn(following, command, k, applied_action=action)
        np.testing.assert_array_equal(
            agent.incremental_model.theta, shifted.incremental_model.theta
        )
        np.testing.assert_array_equal(
            agent.incremental_model.P, shifted.incremental_model.P
        )
        state = following
    assert agent.incremental_model.num_updates > 90


def test_known_reference_critic_uses_prior_output_prediction_minus_next_command(
    monkeypatch,
):
    agent = IMGDHPAgent(
        2,
        1,
        config=IMGDHPConfig(
            identifier_mode="output", warmup_steps=0, critic_only_steps=100, seed=0
        ),
    )
    agent.incremental_model.theta[:] = [[0.4], [0.2]]
    agent._y_tm1 = np.array([0.3])
    agent._u_tm1 = np.array([0.1])
    ref = np.array([[0.5, 3.0]])
    agent.predict(np.array([0.7, 99.0]), ref, 0)
    spy = Mock(return_value=0.0)
    monkeypatch.setattr(agent, "_critic_update", spy)
    # The arbitrary next measurement must not leak into the TD target.
    agent.learn(np.array([-10.0, -99.0]), ref, 0, applied_action=np.array([0.6]))
    values = spy.call_args.kwargs
    np.testing.assert_allclose(values["aug_t_np"], [0.2])
    np.testing.assert_allclose(values["y_next_np"], [0.7 + 0.4 * 0.4 + 0.2 * 0.5 - 3.0])
    assert values["c_now_value"] == pytest.approx(0.2**2)


def test_known_reference_actor_gradient_uses_future_tracking_error():
    agent = IMGDHPAgent(
        1,
        1,
        config=IMGDHPConfig(
            identifier_mode="output",
            actor_hidden=(),
            critic_hidden=(),
            actor_lr=0.02,
            u_max=2,
            seed=0,
        ),
    )
    with torch.no_grad():
        agent.actor.head.weight[:] = torch.tensor([[0.1, 0.2]])
        agent.critic.j_head.weight.fill_(0.7)
    agent.incremental_model.theta[:] = [[0.4], [0.2]]
    previous = agent.actor.head.weight.detach().clone()
    # Independent differentiable expression for J(y_next - r_next)^2 / 2.
    u = 2 * torch.tanh(previous[0, 0] * 0.2 + previous[0, 1] * 0.01)
    prediction = 0.7 + 0.4 * (0.7 - 0.3) + 0.2 * (u - 0.1) - 3.0
    gradient = (
        0.7**2 * prediction * 0.2 * 2 * (1 - (u / 2) ** 2) * torch.tensor([[0.2, 0.01]])
    )
    loss = agent._actor_update(
        np.array([0.7]),
        np.array([0.5]),
        np.array([3.0]),
        np.array([0.1]),
        np.array([0.3]),
    )
    assert loss == pytest.approx(float(0.5 * (0.7 * prediction) ** 2), rel=1e-6)
    torch.testing.assert_close(agent.actor.head.weight, previous - 0.02 * gradient)


def test_known_reference_checkpoint_continues_across_a_jump(tmp_path):
    agent = IMGDHPAgent(
        1,
        1,
        config=IMGDHPConfig(
            identifier_mode="output",
            history_length=2,
            seed=3,
            warmup_steps=2,
            exploration_noise_std=0.01,
        ),
    )
    ref = np.zeros((1, 20))
    ref[:, 8:14] = 0.4
    ref[:, 14:] = -0.2
    state = np.array([0.1])
    for k in range(8):
        action = agent.predict(state, ref, k)
        state = 0.8 * state + 0.2 * action
        agent.learn(state, ref, k)
    # Save a pending transition too, not just a reset policy.
    action = agent.predict(state, ref, 8)
    restored = IMGDHPAgent.from_pretrained(
        agent.save(tmp_path, save_gradients=True), load_gradients=True
    )
    assert restored.cfg.identifier_mode == "output"
    for k in range(8, 19):
        if k > 8:
            action = agent.predict(state, ref, k)
            np.testing.assert_array_equal(action, restored.predict(state, ref, k))
        state = 0.8 * state + 0.2 * action
        assert agent.learn(state, ref, k) == restored.learn(state, ref, k)
        np.testing.assert_array_equal(
            agent.incremental_model.theta, restored.incremental_model.theta
        )


@pytest.mark.parametrize("hidden", [(), (4,), (4, 3)])
def test_actor_input_retuning_preserves_trim_and_has_explicit_feedback_gain(
    hidden, tmp_path
):
    agent = IMGDHPAgent(1, 1, config=IMGDHPConfig(actor_hidden=hidden, seed=2))
    error = torch.tensor([[-0.4], [0.0], [0.3]])
    expected = agent.actor(error * 0.6).detach()
    critic_before = copy.deepcopy(agent.critic.state_dict())
    model_before = agent.incremental_model.theta.copy()
    agent.retune_actor_inputs(feedback_gain=0.6, bias_input=3.2)
    torch.testing.assert_close(agent.actor(error), expected)
    for name, value in critic_before.items():
        torch.testing.assert_close(agent.critic.state_dict()[name], value)
    np.testing.assert_array_equal(agent.incremental_model.theta, model_before)
    restored = IMGDHPAgent.from_pretrained(agent.save(tmp_path))
    assert restored.actor.bias_input == 3.2
    torch.testing.assert_close(restored.actor(error), agent.actor(error))
    before = agent.actor(error).detach()
    agent.retune_actor_inputs(bias_input=1.0)
    torch.testing.assert_close(agent.actor(error), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feedback_gain": 0},
        {"feedback_gain": float("nan")},
        {"bias_input": -1},
        {"bias_input": 1e-15},
    ],
)
def test_invalid_actor_retuning_does_not_partially_change_policy(kwargs):
    agent = IMGDHPAgent(1, 1, config=IMGDHPConfig(seed=0))
    state = copy.deepcopy(agent.actor.state_dict())
    bias = agent.cfg.actor_bias_input
    with pytest.raises(ValueError):
        agent.retune_actor_inputs(**kwargs)
    assert agent.cfg.actor_bias_input == bias
    for key, value in state.items():
        torch.testing.assert_close(value, agent.actor.state_dict()[key])


def test_actor_retuning_rejects_an_unconsumed_transition():
    agent = IMGDHPAgent(1, 1)
    agent.predict([0.0], [[0.0, 0.0]], 0)
    with pytest.raises(RuntimeError, match="pending"):
        agent.retune_actor_inputs(bias_input=2.0)
