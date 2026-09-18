"""Independent scalar oracle for Sun et al. (2022), equations 14 and 17–19."""

import numpy as np
import pytest
import torch
from torch import nn

from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig


class ScalarActor(nn.Module):
    def __init__(self, gain, bound):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.bound = bound

    def forward(self, x):
        d = self.gain * x
        return self.bound * torch.tanh(d), d


class ScalarCritic(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope)))

    def forward(self, x):
        return self.slope * x


class ScalarPlant(nn.Module):
    def forward(self, xu):
        return (0.8 * xu[0] + 0.3 * xu[1]).reshape(1)


@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("gamma", [1.0, 0.95])
def test_sgd_matches_paper_with_policy_derivative_and_simultaneous_targets(
    epochs, gamma
):
    cfg = ETDHPConfig(
        Q=[0.7],
        R=[0.6],
        u_bound=1.3,
        gamma=gamma,
        num_epochs_per_trigger=epochs,
        actor_lr=0.003,
        critic_lr=0.005,
        actor_hidden=(2,),
        critic_hidden=(2,),
        model_hidden=(2,),
    )
    agent = ETDHPAgent(1, 1, config=cfg)
    if agent.writer is not None:
        agent.writer.close()
        agent.writer = None
    agent.actor = ScalarActor(0.4, cfg.u_bound)
    agent.critic = ScalarCritic(1.1)
    agent.plant_model = ScalarPlant()
    agent.actor_opt = torch.optim.SGD(agent.actor.parameters(), lr=cfg.actor_lr)
    agent.critic_opt = torch.optim.SGD(agent.critic.parameters(), lr=cfg.critic_lr)
    a, c, x = 0.4, 1.1, 0.9
    for _ in range(epochs):
        d = a * x
        u = cfg.u_bound * np.tanh(d)
        du_dx = cfg.u_bound * a * (1 - np.tanh(d) ** 2)
        next_x = 0.8 * x + 0.3 * u
        next_lambda = c * next_x
        optimal = cfg.u_bound * np.tanh(
            -gamma * 0.3 * next_lambda / (2 * cfg.u_bound * 0.6)
        )
        # Eq. 18 includes G * d(policy)/dx, not merely the open-loop A.
        closed_derivative = 0.8 + 0.3 * du_dx
        # Eq. 17 includes the derivative of the bounded integral action cost.
        cost_derivative = 2 * 0.7 * x + 2 * cfg.u_bound * 0.6 * d * du_dx
        target = gamma * closed_derivative * next_lambda + cost_derivative
        actor_error, critic_error = u - optimal, c * x - target
        actor_loss, critic_loss = actor_error**2, critic_error**2
        # Author code uses MSELoss; both targets use the old pair of weights.
        a -= cfg.actor_lr * 2 * actor_error * cfg.u_bound * (1 - np.tanh(d) ** 2) * x
        c -= cfg.critic_lr * 2 * critic_error * x
    actual_losses = agent._run_inner_updates(np.array([x]))
    assert agent.actor.gain.item() == pytest.approx(a, abs=1e-7)
    assert agent.critic.slope.item() == pytest.approx(c, abs=1e-7)
    np.testing.assert_allclose(
        actual_losses, [actor_loss, critic_loss], rtol=2e-5, atol=1e-8
    )
