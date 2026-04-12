import torch

from tensoraerospace.agent.sac.model import DeterministicPolicy


def test_deterministic_policy_forward_and_sample():
    pol = DeterministicPolicy(
        num_inputs=3, num_actions=2, hidden_dim=8, action_space=None
    )
    s = torch.zeros((4, 3))
    mean = pol.forward(s)
    assert mean.shape == (4, 2)
    action, logp, mean2 = pol.sample(s)
    assert action.shape == (4, 2)
    assert mean2.shape == (4, 2)
    assert isinstance(logp.item(), float)


def test_deterministic_policy_to_device_with_action_space_none():
    """Bug #14: DeterministicPolicy.to() should not crash when action_space=None."""
    pol = DeterministicPolicy(
        num_inputs=3, num_actions=2, hidden_dim=8, action_space=None
    )
    # This used to crash because action_scale/action_bias were plain floats
    # and .to(device) was called on them.
    pol = pol.to("cpu")
    assert isinstance(pol.action_scale, torch.Tensor)
    assert isinstance(pol.action_bias, torch.Tensor)
    # Verify forward/sample still work after .to()
    s = torch.zeros((2, 3))
    action, logp, mean = pol.sample(s)
    assert action.shape == (2, 2)
    assert logp.device == mean.device


def test_deterministic_policy_sample_returns_tensor_on_correct_device():
    """Bug #30: sample() log_prob should be a tensor on the same device as mean."""
    pol = DeterministicPolicy(
        num_inputs=3, num_actions=2, hidden_dim=8, action_space=None
    )
    s = torch.zeros((2, 3))
    action, logp, mean = pol.sample(s)
    assert isinstance(logp, torch.Tensor)
    assert logp.device == mean.device
