"""Edge case tests for A3C.

This module tests edge cases and robustness including:
- Extreme values
- Edge case dimensions
- Numerical stability
"""

import numpy as np
import torch

from tensoraerospace.agent.a3c.pytorch import Net
from tensoraerospace.agent.a3c.shared_optim import SharedAdam
from tensoraerospace.agent.a3c.utils import push_and_pull, v_wrap


def test_net_with_large_state_values():
    """Test Net with large state values."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Very large state values
    states = torch.randn(5, s_dim) * 1000

    mu, sigma, values = net.forward(states)

    # Should still produce finite outputs
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(values).all()


def test_net_with_zero_states():
    """Test Net with all-zero states."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.zeros(5, s_dim)

    mu, sigma, values = net.forward(states)

    # Should produce valid outputs
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(values).all()
    assert (sigma > 0).all()  # Sigma should always be positive


def test_net_with_single_dimensional_state():
    """Test Net with 1D state space."""
    s_dim, a_dim = 1, 1
    net = Net(s_dim, a_dim)

    state = torch.randn(1, s_dim)

    mu, sigma, values = net.forward(state)

    assert mu.shape == (1, a_dim)
    assert sigma.shape == (1, a_dim)
    assert values.shape == (1, 1)


def test_net_with_high_dimensional_spaces():
    """Test Net with high-dimensional state and action spaces."""
    s_dim, a_dim = 100, 20
    net = Net(s_dim, a_dim)

    state = torch.randn(1, s_dim)

    mu, sigma, values = net.forward(state)

    assert mu.shape == (1, a_dim)
    assert sigma.shape == (1, a_dim)
    assert values.shape == (1, 1)


def test_net_loss_with_extreme_targets():
    """Test Net loss with extreme target values."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.randn(5, s_dim)
    actions = torch.randn(5, a_dim)
    # Extreme target values
    v_targets = torch.randn(5, 1) * 1000

    loss = net.loss_func(states, actions, v_targets)

    # Should still compute finite loss
    assert torch.isfinite(loss)


def test_net_loss_with_zero_targets():
    """Test Net loss with zero targets."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.randn(5, s_dim)
    actions = torch.randn(5, a_dim)
    v_targets = torch.zeros(5, 1)

    loss = net.loss_func(states, actions, v_targets)

    assert torch.isfinite(loss)


def test_v_wrap_with_nan():
    """Test v_wrap handling of NaN values."""
    arr = np.array([1.0, np.nan, 3.0])
    tensor = v_wrap(arr)

    # Should convert, even if values are NaN
    assert tensor.shape == (3,)
    assert torch.isnan(tensor[1])


def test_v_wrap_with_inf():
    """Test v_wrap with infinity values."""
    arr = np.array([1.0, np.inf, -np.inf])
    tensor = v_wrap(arr)

    assert tensor.shape == (3,)
    assert torch.isinf(tensor[1])
    assert torch.isinf(tensor[2])


def test_v_wrap_empty_array():
    """Test v_wrap with empty array."""
    arr = np.array([])
    tensor = v_wrap(arr)

    assert tensor.shape == (0,)


def test_push_and_pull_with_large_rewards():
    """Test push_and_pull with very large rewards."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [1000.0 for _ in range(5)]  # Very large rewards
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    # Should handle large rewards with gradient clipping
    assert np.isfinite(metrics["loss"])


def test_push_and_pull_with_negative_rewards():
    """Test push_and_pull with large negative rewards."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [-100.0 for _ in range(5)]  # Large negative rewards
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert np.isfinite(metrics["loss"])


def test_push_and_pull_with_mixed_rewards():
    """Test push_and_pull with highly variable rewards."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    # Highly variable rewards
    br = [-1000.0, 1000.0, -500.0, 500.0, 0.0]
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert np.isfinite(metrics["loss"])


def test_push_and_pull_with_very_long_buffer():
    """Test push_and_pull with very long buffer."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    # Very long trajectory
    n_steps = 1000
    bs = [np.random.randn(s_dim) for _ in range(n_steps)]
    ba = [np.random.randn(a_dim) for _ in range(n_steps)]
    br = [0.1 for _ in range(n_steps)]
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert np.isfinite(metrics["loss"])


def test_push_and_pull_with_gamma_zero():
    """Test push_and_pull with gamma=0 (no discounting)."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [1.0 for _ in range(5)]
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.0
    )

    assert np.isfinite(metrics["loss"])


def test_push_and_pull_with_gamma_one():
    """Test push_and_pull with gamma=1 (no discounting)."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [1.0 for _ in range(5)]
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=1.0
    )

    assert np.isfinite(metrics["loss"])


def test_net_choose_action_consistency():
    """Test that choose_action produces consistent shapes."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Test with different batch sizes in state
    for batch_size in [1, 2, 5]:
        state = torch.randn(batch_size, s_dim)
        # choose_action expects single state, so use first
        action = net.choose_action(state[:1])

        assert action.shape == (a_dim,)


def test_net_with_extreme_sigma():
    """Test Net behavior when sigma becomes extreme."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Manually set sigma layer weights to produce extreme values
    with torch.no_grad():
        net.sigma.weight.fill_(10.0)
        net.sigma.bias.fill_(10.0)

    states = torch.randn(5, s_dim)
    mu, sigma, values = net.forward(states)

    # Sigma should still be positive and finite due to softplus + 0.001
    assert (sigma > 0).all()
    assert torch.isfinite(sigma).all()


def test_shared_adam_with_zero_gradients():
    """Test SharedAdam with zero gradients."""
    model = torch.nn.Linear(4, 2)
    opt = SharedAdam(model.parameters(), lr=1e-3)

    # Create zero gradients
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    # Step should not error
    opt.step()
    opt.zero_grad()


def test_shared_adam_state_sharing():
    """Test that SharedAdam state is in shared memory."""
    model = torch.nn.Linear(4, 2)
    opt = SharedAdam(model.parameters(), lr=1e-3)

    # Check that state tensors are in shared memory
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            assert state["step"].is_shared()
            assert state["exp_avg"].is_shared()
            assert state["exp_avg_sq"].is_shared()


def test_net_gradient_explosion_protection():
    """Test that gradient clipping prevents explosion."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1.0)  # High LR

    lnet.load_state_dict(gnet.state_dict())

    # Create scenario that might cause gradient explosion
    bs = [np.random.randn(s_dim) * 100 for _ in range(10)]
    ba = [np.random.randn(a_dim) * 100 for _ in range(10)]
    br = [1000.0 for _ in range(10)]
    s_ = np.random.randn(s_dim) * 100

    # Should not explode due to gradient clipping
    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    # Check that all parameters remain finite
    for p in gnet.parameters():
        assert torch.isfinite(p).all()


if __name__ == "__main__":
    # Run all tests
    test_net_with_large_state_values()
    test_net_with_zero_states()
    test_net_with_single_dimensional_state()
    test_net_with_high_dimensional_spaces()
    test_net_loss_with_extreme_targets()
    test_net_loss_with_zero_targets()
    test_v_wrap_with_nan()
    test_v_wrap_with_inf()
    test_v_wrap_empty_array()
    test_push_and_pull_with_large_rewards()
    test_push_and_pull_with_negative_rewards()
    test_push_and_pull_with_mixed_rewards()
    test_push_and_pull_with_very_long_buffer()
    test_push_and_pull_with_gamma_zero()
    test_push_and_pull_with_gamma_one()
    test_net_choose_action_consistency()
    test_net_with_extreme_sigma()
    test_shared_adam_with_zero_gradients()
    test_shared_adam_state_sharing()
    test_net_gradient_explosion_protection()
    print("All edge case tests passed!")
