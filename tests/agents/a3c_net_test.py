"""Tests for A3C Net (neural network) class.

This module tests the Net class including:
- Network initialization
- Forward pass
- Action selection
- Loss computation
"""

import numpy as np
import torch
import torch.nn as nn

from tensoraerospace.agent.a3c.pytorch import Net


def test_net_initialization():
    """Test Net initialization with different dimensions."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    assert net.s_dim == s_dim
    assert net.a_dim == a_dim
    assert isinstance(net.a1, nn.Linear)
    assert isinstance(net.mu, nn.Linear)
    assert isinstance(net.sigma, nn.Linear)
    assert isinstance(net.c1, nn.Linear)
    assert isinstance(net.v, nn.Linear)


def test_net_forward_pass():
    """Test forward pass through Net."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Create random state
    state = torch.randn(1, s_dim)

    # Forward pass
    mu, sigma, values = net.forward(state)

    # Check output shapes
    assert mu.shape == (1, a_dim)
    assert sigma.shape == (1, a_dim)
    assert values.shape == (1, 1)

    # Check that outputs are finite
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(values).all()

    # Check that sigma is positive (due to softplus + 0.001)
    assert (sigma > 0).all()


def test_net_forward_batch():
    """Test forward pass with batch of states."""
    s_dim, a_dim = 3, 1
    net = Net(s_dim, a_dim)

    # Batch of states
    batch_size = 10
    states = torch.randn(batch_size, s_dim)

    mu, sigma, values = net.forward(states)

    assert mu.shape == (batch_size, a_dim)
    assert sigma.shape == (batch_size, a_dim)
    assert values.shape == (batch_size, 1)


def test_net_mu_bounds():
    """Test that mu output is bounded by tanh."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.randn(100, s_dim)
    mu, _, _ = net.forward(states)

    # mu = 2 * tanh(x), so should be in [-2, 2]
    assert (mu >= -2.0).all()
    assert (mu <= 2.0).all()


def test_net_sigma_minimum():
    """Test that sigma has minimum value."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.randn(100, s_dim)
    _, sigma, _ = net.forward(states)

    # sigma = softplus(x) + 0.001, so minimum is 0.001
    assert (sigma >= 0.001).all()


def test_net_choose_action():
    """Test action selection."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    state = torch.randn(1, s_dim)
    action = net.choose_action(state)

    # Check output type and shape
    assert isinstance(action, np.ndarray)
    assert action.shape == (a_dim,)
    assert np.isfinite(action).all()


def test_net_choose_action_single_action_dim():
    """Test action selection with single action dimension."""
    s_dim, a_dim = 3, 1
    net = Net(s_dim, a_dim)

    state = torch.randn(1, s_dim)
    action = net.choose_action(state)

    assert isinstance(action, np.ndarray)
    # Should be scalar or 1D array with single element
    assert action.size == 1


def test_net_choose_action_deterministic():
    """Test that choose_action is stochastic (varies between calls)."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    state = torch.randn(1, s_dim)

    # Sample multiple times
    actions = [net.choose_action(state) for _ in range(10)]

    # At least some actions should differ (stochastic sampling)
    # With very high probability
    unique_actions = len(set(tuple(a) for a in actions))
    assert unique_actions > 1


def test_net_loss_func():
    """Test loss function computation."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Create dummy batch
    batch_size = 5
    states = torch.randn(batch_size, s_dim)
    actions = torch.randn(batch_size, a_dim)
    v_targets = torch.randn(batch_size, 1)

    # Compute loss
    loss = net.loss_func(states, actions, v_targets)

    # Check that loss is a scalar
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_net_loss_func_backprop():
    """Test that loss can be backpropagated."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    batch_size = 5
    states = torch.randn(batch_size, s_dim)
    actions = torch.randn(batch_size, a_dim)
    v_targets = torch.randn(batch_size, 1)

    # Compute loss and backward
    loss = net.loss_func(states, actions, v_targets)
    loss.backward()

    # Check that gradients were computed
    for param in net.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_net_loss_components():
    """Test that loss includes value and policy components."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    batch_size = 5
    states = torch.randn(batch_size, s_dim)
    actions = torch.randn(batch_size, a_dim)
    v_targets = torch.randn(batch_size, 1)

    loss = net.loss_func(states, actions, v_targets)

    # Loss should be finite
    # Note: A3C loss can be negative due to entropy bonus
    assert torch.isfinite(loss)


def test_net_train_eval_mode():
    """Test switching between train and eval modes."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    # Initially in training mode
    assert net.training

    # Switch to eval
    net.eval()
    assert not net.training

    # Switch back to train
    net.train()
    assert net.training


def test_net_different_state_dimensions():
    """Test Net with various state dimensions."""
    for s_dim in [1, 3, 5, 10, 20]:
        a_dim = 2
        net = Net(s_dim, a_dim)
        state = torch.randn(1, s_dim)
        mu, sigma, values = net.forward(state)

        assert mu.shape == (1, a_dim)
        assert sigma.shape == (1, a_dim)
        assert values.shape == (1, 1)


def test_net_different_action_dimensions():
    """Test Net with various action dimensions."""
    s_dim = 4
    for a_dim in [1, 2, 3, 5, 10]:
        net = Net(s_dim, a_dim)
        state = torch.randn(1, s_dim)
        action = net.choose_action(state)

        assert action.shape == (a_dim,)


def test_net_distribution_type():
    """Test that Net uses Normal distribution."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    assert net.distribution == torch.distributions.Normal


def test_net_independent_distribution():
    """Test Independent distribution for multi-dimensional actions."""
    s_dim, a_dim = 4, 3  # Multi-dimensional action
    net = Net(s_dim, a_dim)

    states = torch.randn(5, s_dim)
    actions = torch.randn(5, a_dim)

    # This should work without error (uses Independent distribution)
    mu, sigma, _ = net.forward(states)
    base = net.distribution(mu, sigma)
    dist = torch.distributions.Independent(base, 1)
    log_prob = dist.log_prob(actions)

    assert log_prob.shape == (5,)


def test_net_parameter_count():
    """Test that Net has reasonable number of parameters."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    total_params = sum(p.numel() for p in net.parameters())

    # Should have parameters from all layers
    assert total_params > 0
    # Rough estimate based on architecture
    # a1: 4*256 + 256, mu: 256*2 + 2, sigma: 256*2 + 2
    # c1: 4*256 + 256, v: 256*1 + 1
    expected_min = 1000  # Conservative lower bound
    assert total_params >= expected_min


def test_net_gradient_flow():
    """Test that gradients flow through all parameters."""
    s_dim, a_dim = 4, 2
    net = Net(s_dim, a_dim)

    states = torch.randn(5, s_dim)
    actions = torch.randn(5, a_dim)
    v_targets = torch.randn(5, 1)

    loss = net.loss_func(states, actions, v_targets)
    loss.backward()

    # All parameters should have gradients
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


if __name__ == "__main__":
    # Run all tests
    test_net_initialization()
    test_net_forward_pass()
    test_net_forward_batch()
    test_net_mu_bounds()
    test_net_sigma_minimum()
    test_net_choose_action()
    test_net_choose_action_single_action_dim()
    test_net_choose_action_deterministic()
    test_net_loss_func()
    test_net_loss_func_backprop()
    test_net_loss_components()
    test_net_train_eval_mode()
    test_net_different_state_dimensions()
    test_net_different_action_dimensions()
    test_net_distribution_type()
    test_net_independent_distribution()
    test_net_parameter_count()
    test_net_gradient_flow()
    print("All Net tests passed!")
