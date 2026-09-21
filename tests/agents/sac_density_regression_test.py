"""Density and entropy gradients must survive actuator saturation."""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Normal

from tensoraerospace.agent.sac.model import GaussianPolicy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-7, 1.0, 25.0])
def test_density_and_gradients_match_change_of_variables(monkeypatch, dtype, scale):
    policy = GaussianPolicy(1, 2, 4).to(dtype=dtype)
    mean = torch.tensor([[-20.0, 20.0], [-2.0, 0.5]], dtype=dtype, requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    policy.action_scale.fill_(scale)
    policy.action_bias.fill_(0.25)
    monkeypatch.setattr(policy, "forward", lambda state: (mean, log_std))
    # A fixed reparameterized draw permits an independent analytic derivative.
    monkeypatch.setattr(Normal, "rsample", lambda self: self.loc + 0 * self.scale)
    action, logp, deterministic = policy.sample(torch.zeros(2, 1, dtype=dtype))
    x = mean.detach().double().numpy()
    log_cosh = np.logaddexp(x, -x) - math.log(2)
    expected = (-0.5 * math.log(2 * math.pi) - math.log(scale) + 2 * log_cosh).sum(1)
    np.testing.assert_allclose(
        logp.detach().numpy().ravel(), expected, rtol=2e-6, atol=2e-6
    )
    grad_mean, grad_std = torch.autograd.grad(logp.sum(), (mean, log_std))
    torch.testing.assert_close(grad_mean, 2 * mean.detach().tanh())
    torch.testing.assert_close(grad_std, -torch.ones_like(mean))
    torch.testing.assert_close(action, deterministic)
    assert torch.isfinite(action).all()


@pytest.mark.parametrize("sign", [-1, 1])
def test_entropy_optimizer_can_leave_saturated_mean(sign):
    torch.manual_seed(17)
    policy = GaussianPolicy(1, 1, 4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.mean_linear.bias.fill_(sign * 12)
        policy.log_std_linear.bias.fill_(-5)
    optimizer = torch.optim.Adam([policy.mean_linear.bias], lr=0.05)
    for _ in range(240):
        _, logp, _ = policy.sample(torch.zeros(16, 1))
        optimizer.zero_grad()
        logp.mean().backward()
        optimizer.step()
        assert torch.isfinite(logp).all()
    assert abs(policy.mean_linear.bias.item()) < 2


@pytest.mark.parametrize("low,high", [(0.0, 0.0), (-np.inf, 1.0), (-1.0, np.inf)])
def test_gaussian_requires_finite_nonzero_action_intervals(low, high):
    space = gym.spaces.Box(low, high, (1,), np.float32)
    with pytest.raises(ValueError, match="finite.*positive"):
        GaussianPolicy(1, 1, 4, space)


def test_log_density_has_correct_physical_units(monkeypatch):
    unit = GaussianPolicy(1, 2, 4).double()
    scaled = GaussianPolicy(1, 2, 4).double()
    scaled.load_state_dict(unit.state_dict())
    scaled.action_scale.fill_(1e-7)
    # Keep exactly the same latent samples while retaining their gradients.
    monkeypatch.setattr(Normal, "rsample", lambda self: self.loc + 0.4 * self.scale)
    obs = torch.ones(3, 1, dtype=torch.float64)
    a_unit, log_unit, _ = unit.sample(obs)
    a_scaled, log_scaled, _ = scaled.sample(obs)
    torch.testing.assert_close(a_scaled, a_unit * 1e-7)
    torch.testing.assert_close(
        log_scaled - log_unit, torch.full_like(log_unit, -2 * math.log(1e-7))
    )
