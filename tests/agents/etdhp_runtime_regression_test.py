"""ET-DHP must hold control between events and remain finite at saturation."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.et_dhp.model import (
    ETDHPAgent,
    ETDHPConfig,
    _bounded_integral_cost,
    _jacobians_per_output,
)


def make_agent():
    agent = ETDHPAgent(
        1,
        1,
        config=ETDHPConfig(
            actor_hidden=(4,),
            critic_hidden=(4,),
            model_hidden=(4,),
            Q=(1.0,),
            R=(1.0,),
            num_epochs_per_trigger=1,
            trigger_floor=0.0,
            device="cpu",
            seed=17,
        ),
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.fill_(0.5)
    return agent


def test_control_is_held_exactly_between_events():
    agent = make_agent()
    try:
        agent.predict(np.array([0.1]), None, 0)
        assert agent.learn(np.array([0.1]), None, 0)["triggered"] == 1.0
        held = agent.last_action()
        calls = []
        hook = agent.actor.register_forward_hook(
            lambda module, args, output: calls.append(1)
        )
        try:
            actual = agent.predict(np.array([0.101]), None, 1)
            metrics = agent.learn(np.array([0.101]), None, 1)
            assert metrics["triggered"] == 0.0
            np.testing.assert_array_equal(actual, held)
            np.testing.assert_array_equal(agent.last_action(), held)
            assert calls == []
        finally:
            hook.remove()
    finally:
        if agent.writer is not None:
            agent.writer.close()


def test_trigger_updates_from_current_measurement_and_reset_releases_hold(monkeypatch):
    agent = make_agent()
    updated_states = []

    def update(state):
        updated_states.append(state.copy())
        return 0.0, 0.0

    monkeypatch.setattr(agent, "_run_inner_updates", update)
    try:
        agent.predict(np.array([0.1]), None, 0)
        assert agent.learn(np.array([0.25]), None, 0)["triggered"] == 1.0
        np.testing.assert_array_equal(updated_states, [[0.25]])
        with torch.no_grad():
            expected, _ = agent.actor(torch.tensor([0.25]))
        np.testing.assert_allclose(agent.last_action(), expected.numpy())
        held = agent.predict(np.array([0.251]), None, 1)
        np.testing.assert_allclose(held, expected.numpy())
        agent.reset()
        reset_action = agent.predict(np.array([0.5]), None, 0)
        assert not np.allclose(reset_action, held)
    finally:
        if agent.writer is not None:
            agent.writer.close()


@pytest.mark.parametrize("value", [-100.0, 100.0])
def test_saturated_integral_cost_and_its_gradient_are_finite(value):
    d = torch.tensor([value], requires_grad=True)
    cost = _bounded_integral_cost(d, u_bound=2.0, R=torch.tensor([3.0]))
    # Integral from 0 to a saturated action tends to 2 * R * u_bound^2 * log(2).
    assert cost.item() == pytest.approx(24.0 * np.log(2.0), rel=1e-5)
    cost.backward()
    assert torch.isfinite(d.grad).all()
    torch.testing.assert_close(d.grad, torch.zeros_like(d.grad), atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_plant_jacobians_stay_on_the_model_device(device):
    x = torch.ones(2, device=device, requires_grad=True)
    u = torch.ones(1, device=device, requires_grad=True)
    y = torch.stack([2 * x[0] + 3 * u[0], -x[1] + 4 * u[0]])
    f, g = _jacobians_per_output(y, [x, u])
    assert f.device == g.device == y.device
    assert f.shape == (2, 2)
    assert g.shape == (2, 1)
    if device == "cpu":
        torch.testing.assert_close(f, torch.tensor([[2.0, 0.0], [0.0, -1.0]]))
        torch.testing.assert_close(g, torch.tensor([[3.0], [4.0]]))
