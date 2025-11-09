"""Comprehensive tests for DDPG neural networks.

Tests for ValueNetwork and PolicyNetwork classes.
"""
import numpy as np
import pytest
import torch

from tensoraerospace.agent.ddpg.model import PolicyNetwork, ValueNetwork


class TestValueNetwork:
    """Tests for ValueNetwork (Critic)."""

    def test_init(self):
        """Test network initialization."""
        net = ValueNetwork(num_inputs=4, num_actions=2, hidden_size=32)

        assert net.linear1.in_features == 6  # 4 + 2
        assert net.linear1.out_features == 32
        assert net.linear3.out_features == 1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        net = ValueNetwork(num_inputs=4, num_actions=2, hidden_size=32)
        device = next(net.parameters()).device

        batch_size = 8
        state = torch.randn(batch_size, 4, device=device)
        action = torch.randn(batch_size, 2, device=device)

        q_value = net(state, action)

        assert q_value.shape == (batch_size, 1)

    def test_forward_single(self):
        """Test forward pass with single sample."""
        net = ValueNetwork(num_inputs=3, num_actions=1, hidden_size=16)
        device = next(net.parameters()).device

        state = torch.randn(1, 3, device=device)
        action = torch.randn(1, 1, device=device)

        q_value = net(state, action)

        assert q_value.shape == (1, 1)
        assert torch.isfinite(q_value).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        net = ValueNetwork(num_inputs=3, num_actions=1, hidden_size=16)
        device = next(net.parameters()).device

        state = torch.randn(4, 3, requires_grad=True, device=device)
        action = torch.randn(4, 1, requires_grad=True, device=device)

        q_value = net(state, action)
        loss = q_value.mean()
        loss.backward()

        # Check that gradients are computed
        assert state.grad is not None
        assert action.grad is not None
        assert net.linear1.weight.grad is not None

    def test_weight_initialization(self):
        """Test that final layer weights are properly initialized."""
        init_w = 3e-3
        net = ValueNetwork(num_inputs=3, num_actions=1, hidden_size=16, init_w=init_w)

        # Check final layer weights are in expected range
        weights = net.linear3.weight.data
        bias = net.linear3.bias.data

        assert torch.all(torch.abs(weights) <= init_w)
        assert torch.all(torch.abs(bias) <= init_w)

    def test_different_dimensions(self):
        """Test with various input/output dimensions."""
        configs = [
            (2, 1, 8),
            (10, 5, 64),
            (20, 10, 128),
        ]

        for num_inputs, num_actions, hidden_size in configs:
            net = ValueNetwork(num_inputs, num_actions, hidden_size)
            device = next(net.parameters()).device
            state = torch.randn(1, num_inputs, device=device)
            action = torch.randn(1, num_actions, device=device)

            q_value = net(state, action)
            assert q_value.shape == (1, 1)


class TestPolicyNetwork:
    """Tests for PolicyNetwork (Actor)."""

    def test_init_default(self):
        """Test network initialization with default action bounds."""
        net = PolicyNetwork(num_inputs=4, num_actions=2, hidden_size=32)

        assert net.linear1.in_features == 4
        assert net.linear3.out_features == 2
        assert hasattr(net, "action_scale")
        assert hasattr(net, "action_bias")

    def test_init_with_action_bounds(self):
        """Test initialization with custom action bounds."""
        action_low = np.array([-2.0, -3.0])
        action_high = np.array([2.0, 3.0])

        net = PolicyNetwork(
            num_inputs=4,
            num_actions=2,
            hidden_size=32,
            action_low=action_low,
            action_high=action_high,
        )
        device = next(net.parameters()).device

        expected_scale = (action_high - action_low) / 2.0
        expected_bias = (action_high + action_low) / 2.0

        assert torch.allclose(
            net.action_scale, torch.FloatTensor(expected_scale).to(device), atol=1e-5
        )
        assert torch.allclose(
            net.action_bias, torch.FloatTensor(expected_bias).to(device), atol=1e-5
        )

    def test_forward_shape(self):
        """Test forward pass output shape."""
        net = PolicyNetwork(num_inputs=4, num_actions=2, hidden_size=32)
        device = next(net.parameters()).device

        batch_size = 8
        state = torch.randn(batch_size, 4, device=device)

        action = net(state)

        assert action.shape == (batch_size, 2)

    def test_forward_output_range_default(self):
        """Test that output is in default range [-1, 1]."""
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16)
        device = next(net.parameters()).device

        state = torch.randn(100, 3, device=device)
        actions = net(state)

        # With default bounds, should be roughly in [-1, 1]
        assert torch.all(actions >= -1.1)  # Small tolerance
        assert torch.all(actions <= 1.1)

    def test_forward_output_range_custom(self):
        """Test that output respects custom action bounds."""
        action_low = np.array([-5.0])
        action_high = np.array([10.0])

        net = PolicyNetwork(
            num_inputs=3,
            num_actions=1,
            hidden_size=16,
            action_low=action_low,
            action_high=action_high,
        )
        device = next(net.parameters()).device

        state = torch.randn(100, 3, device=device)
        actions = net(state)

        # Should be roughly in [action_low, action_high]
        assert torch.all(actions >= -5.5)  # Small tolerance
        assert torch.all(actions <= 10.5)

    def test_get_action_numpy(self):
        """Test get_action returns numpy array."""
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16)

        state = np.array([1.0, 2.0, 3.0])
        action = net.get_action(state)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_get_action_no_gradient(self):
        """Test that get_action doesn't require gradients."""
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16)

        state = np.array([1.0, 2.0, 3.0])

        # Should not raise error even without gradient tracking
        action = net.get_action(state)
        assert action is not None

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16)
        device = next(net.parameters()).device

        state = torch.randn(4, 3, requires_grad=True, device=device)

        action = net(state)
        loss = action.mean()
        loss.backward()

        # Check that gradients are computed
        assert state.grad is not None
        assert net.linear1.weight.grad is not None

    def test_weight_initialization(self):
        """Test that final layer weights are properly initialized."""
        init_w = 3e-3
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16, init_w=init_w)

        # Check final layer weights are in expected range
        weights = net.linear3.weight.data
        bias = net.linear3.bias.data

        assert torch.all(torch.abs(weights) <= init_w)
        assert torch.all(torch.abs(bias) <= init_w)

    def test_action_scaling_symmetry(self):
        """Test that action scaling works correctly for symmetric bounds."""
        action_low = np.array([-10.0, -20.0])
        action_high = np.array([10.0, 20.0])

        net = PolicyNetwork(
            num_inputs=3,
            num_actions=2,
            hidden_size=16,
            action_low=action_low,
            action_high=action_high,
        )
        device = next(net.parameters()).device

        # For zero input to tanh (before activation), output should be near bias
        state = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            # Manually set weights to zero to get zero output from tanh
            net.linear3.weight.data.fill_(0)
            net.linear3.bias.data.fill_(0)

            action = net(state)

            # tanh(0) = 0, so action should be bias (midpoint)
            expected = torch.FloatTensor([0.0, 0.0]).to(device)
            assert torch.allclose(action[0], expected, atol=1e-5)

    def test_different_dimensions(self):
        """Test with various input/output dimensions."""
        configs = [
            (2, 1, 8),
            (10, 5, 64),
            (20, 10, 128),
        ]

        for num_inputs, num_actions, hidden_size in configs:
            net = PolicyNetwork(num_inputs, num_actions, hidden_size)
            device = next(net.parameters()).device
            state = torch.randn(1, num_inputs, device=device)

            action = net(state)
            assert action.shape == (1, num_actions)

    def test_deterministic_output(self):
        """Test that same input produces same output (deterministic policy)."""
        net = PolicyNetwork(num_inputs=3, num_actions=1, hidden_size=16)
        device = next(net.parameters()).device

        state = torch.randn(1, 3, device=device)

        action1 = net(state)
        action2 = net(state)

        assert torch.allclose(action1, action2)

    def test_batch_processing(self):
        """Test processing multiple states in batch."""
        net = PolicyNetwork(num_inputs=3, num_actions=2, hidden_size=16)

        batch_sizes = [1, 4, 16, 32]

        for batch_size in batch_sizes:
            device = next(net.parameters()).device
            state = torch.randn(batch_size, 3, device=device)
            actions = net(state)

            assert actions.shape == (batch_size, 2)
            assert torch.isfinite(actions).all()
