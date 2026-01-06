import pytest
import torch

from tensoraerospace.agent.dsac.flight_actor import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    NormalPolicyNet,
)
from tensoraerospace.agent.dsac.flight_mlp import make_mlp
from tensoraerospace.agent.dsac.flight_critic import ZNet
from tensoraerospace.agent.dsac.model import IQNCritic, QuantileTwin
from tensoraerospace.agent.dsac.risk_distortions import (
    distortion_functions,
    cpw,
    normal_cdf,
    normal_inverse_cdf,
    wang,
)


def test_iqncritic_forward_shapes_and_validation():
    critic = IQNCritic(obs_dim=3, act_dim=2, hidden_layers=[16, 8])
    state = torch.zeros((4, 3))
    action = torch.zeros((4, 2))
    taus = torch.rand((4, 5, 1))

    out = critic(state, action, taus)
    assert out.shape == (4, 5)

    with pytest.raises(ValueError):
        IQNCritic(obs_dim=3, act_dim=2, hidden_layers=[8])


def test_quantile_twin_forward_returns_two_heads():
    twin = QuantileTwin(obs_dim=2, act_dim=1, hidden_layers=[8, 8])
    state = torch.zeros((2, 2))
    action = torch.zeros((2, 1))
    taus = torch.rand((2, 3, 1))

    q1, q2 = twin(state, action, taus)
    assert q1.shape == (2, 3)
    assert q2.shape == (2, 3)
    assert not torch.equal(q1, q2)


def test_normal_policy_net_distribution_shapes_and_std_bounds():
    net = NormalPolicyNet(
        obs_dim=4,
        action_dim=2,
        n_hidden_layers=1,
        n_hidden_units=8,
    )
    states = torch.zeros((5, 4))

    dist = net(states)
    sample = dist.sample()
    assert sample.shape == (5, 2)
    assert net.get_mean(states).shape == (5, 2)

    stds = net.get_std(states)
    assert stds.shape == (5, 2)
    lower = float(torch.exp(torch.tensor(LOG_STD_MIN)))
    upper = float(torch.exp(torch.tensor(LOG_STD_MAX)))
    assert torch.all(stds >= lower)
    assert torch.all(stds <= upper + 1e-6)


def test_normal_policy_net_clamps_log_std_extremes():
    net = NormalPolicyNet(
        obs_dim=3,
        action_dim=1,
        n_hidden_layers=0,
        n_hidden_units=4,
    )
    states = torch.zeros((2, 3))
    with torch.no_grad():
        net.log_std_layer.bias.fill_(100.0)
    stds_hi = net.get_std(states)
    assert torch.allclose(stds_hi, torch.full_like(stds_hi, float(torch.exp(torch.tensor(LOG_STD_MAX)))), atol=1e-4)

    with torch.no_grad():
        net.log_std_layer.bias.fill_(-100.0)
    stds_lo = net.get_std(states)
    assert torch.all(stds_lo >= float(torch.exp(torch.tensor(LOG_STD_MIN))) - 1e-5)


def test_make_mlp_structure_and_forward():
    mlp = make_mlp(
        num_in=3,
        num_out=2,
        n_hidden_layers=2,
        n_hidden_units=5,
        final_activation=torch.nn.Tanh(),
    )
    # Expected layers: input Linear+ReLU (2) + 2*(Linear+LayerNorm+ReLU)=6 + Linear + final_activation = 2
    assert len(list(mlp)) == 10
    x = torch.zeros((4, 3))
    out = mlp(x)
    assert out.shape == (4, 2)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_znet_forward_and_tau_generation_cpu():
    device = torch.device("cpu")
    net = ZNet(
        n_states=3,
        n_actions=2,
        n_hidden_layers=1,
        n_hidden_units=8,
        n_cos=4,
        device=device,
    )
    taus = ZNet.generate_taus(batch_size=2, n_taus=6, device=device)
    assert taus.shape == (2, 6)
    assert torch.all(taus > 0.0)
    assert torch.all(taus < 1.0)

    states = torch.zeros((2, 3))
    actions = torch.zeros((2, 2))
    z = net(states, actions, taus)
    assert z.shape == (2, 6)


def test_risk_distortions_preserve_shape_and_bounds():
    tau = torch.linspace(0, 1, 5)
    xi = 0.5
    for name, fn in distortion_functions.items():
        distorted = fn(tau, xi)
        assert distorted.shape == tau.shape
        assert torch.all(distorted >= 0.0)
        assert torch.all(distorted <= 1.0)

    neutral_tau = distortion_functions["neutral"](tau, 1.0)
    assert torch.allclose(neutral_tau, tau)

    cvar_clamped = distortion_functions["cvar"](tau, 2.0)
    assert torch.isclose(cvar_clamped.max(), torch.tensor(1.0))


def test_wang_inverse_consistency_small_range():
    tau = torch.linspace(0.1, 0.9, 5)
    shifted = wang(tau, xi=0.0)
    assert torch.allclose(shifted, tau, atol=1e-6)
    # round-trip with small xi ~0 should be close to original after inverse/forward
    xi = 0.1
    forward = wang(tau, xi)
    inv = normal_cdf(normal_inverse_cdf(forward) - xi)
    assert torch.allclose(inv, tau, atol=1e-4)


def test_cpw_monotonic_in_tau():
    tau = torch.linspace(0.0, 1.0, 11)
    out = cpw(tau, xi=0.8)
    diff = torch.diff(out)
    assert torch.all(diff >= -1e-6)

