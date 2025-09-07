import torch

from tensoraerospace.agent.sac.model import GaussianPolicy, QNetwork, ValueNetwork


def test_sac_networks_forward_and_policy_sample():
    vn = ValueNetwork(num_inputs=3, hidden_dim=8)
    qn = QNetwork(num_inputs=3, num_actions=2, hidden_dim=8)
    pol = GaussianPolicy(num_inputs=3, num_actions=2, hidden_dim=8, action_space=None)

    s = torch.zeros((4, 3))
    a = torch.zeros((4, 2))

    v = vn(s)
    assert v.shape == (4, 1)

    q1, q2 = qn(s, a)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)

    act, logp, mean = pol.sample(s)
    assert act.shape == (4, 2)
    assert logp.shape == (4, 1)
    assert mean.shape == (4, 2)
