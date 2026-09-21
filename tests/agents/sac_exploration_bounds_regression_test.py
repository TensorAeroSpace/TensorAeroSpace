"""Exploration must respect actuator units, bounds, and independent samples."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.sac.model import DeterministicPolicy


def make_policy(low, high, bias=0):
    space = gym.spaces.Box(np.array(low, np.float32), np.array(high, np.float32))
    policy = DeterministicPolicy(3, len(low), 8, space)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.mean.bias.fill_(bias)
    return policy, space


@pytest.mark.parametrize("bias", [-20, 20])
@pytest.mark.parametrize("bounds", [([-1], [1]), ([2, 0], [2.01, 130000])])
def test_exploration_never_leaves_physical_action_space(bias, bounds):
    policy, space = make_policy(*bounds, bias=bias)
    torch.manual_seed(29)
    action, _, mean = policy.sample(torch.zeros(4096, 3))
    assert torch.all(mean >= torch.as_tensor(space.low))
    assert torch.all(mean <= torch.as_tensor(space.high))
    assert torch.all(action >= torch.as_tensor(space.low))
    assert torch.all(action <= torch.as_tensor(space.high))
    torch.testing.assert_close(mean, policy.deterministic(torch.zeros(4096, 3)))


def test_identical_vector_observations_get_independent_exploration():
    policy, _ = make_policy([-1, -1], [1, 1])
    torch.manual_seed(11)
    action, _, _ = policy.sample(torch.zeros(128, 3))
    assert torch.unique(action, dim=0).shape[0] > 120


def test_exploration_is_invariant_to_physical_units():
    normal, _ = make_policy([-1, -1], [1, 1])
    physical, _ = make_policy([-0.01, 0], [0.01, 130000])
    torch.manual_seed(47)
    expected, _, _ = normal.sample(torch.zeros(128, 3))
    torch.manual_seed(47)
    action, _, _ = physical.sample(torch.zeros(128, 3))
    normalized = (action - physical.action_bias) / physical.action_scale
    torch.testing.assert_close(normalized, expected, rtol=1e-5, atol=1e-7)


def test_legacy_noise_buffer_remains_loadable_and_eval_is_unchanged():
    policy, _ = make_policy([-2], [3], bias=0.2)
    legacy = policy.state_dict()
    legacy["noise"].fill_(0.15)
    restored, _ = make_policy([-2], [3])
    restored.load_state_dict(legacy, strict=True)
    obs = torch.ones(4, 3)
    torch.testing.assert_close(restored.deterministic(obs), policy.forward(obs))
