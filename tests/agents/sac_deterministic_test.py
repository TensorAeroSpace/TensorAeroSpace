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
