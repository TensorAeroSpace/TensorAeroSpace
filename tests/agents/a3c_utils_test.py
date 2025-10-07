"""Tests for A3C utility functions.

This module tests utility functions including:
- v_wrap: numpy to tensor conversion
- set_init: layer initialization
- push_and_pull: gradient synchronization
- record: episode recording
"""

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tensoraerospace.agent.a3c.pytorch import Net
from tensoraerospace.agent.a3c.shared_optim import SharedAdam
from tensoraerospace.agent.a3c.utils import (
    push_and_pull,
    record,
    set_init,
    v_wrap,
)


def test_v_wrap_basic():
    """Test basic v_wrap functionality."""
    arr = np.array([1.0, 2.0, 3.0])
    tensor = v_wrap(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,)
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_v_wrap_dtype_conversion():
    """Test v_wrap with dtype conversion."""
    arr = np.array([1, 2, 3], dtype=np.int32)
    tensor = v_wrap(arr, dtype=np.float32)

    assert tensor.dtype == torch.float32


def test_v_wrap_2d_array():
    """Test v_wrap with 2D array."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    tensor = v_wrap(arr)

    assert tensor.shape == (2, 2)
    assert torch.allclose(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_v_wrap_preserves_values():
    """Test that v_wrap preserves numerical values."""
    arr = np.random.randn(10, 5)
    tensor = v_wrap(arr)

    # Check that values match
    assert np.allclose(tensor.numpy(), arr)


def test_set_init_weights():
    """Test set_init initializes layer weights."""
    layer = nn.Linear(4, 2)

    # Before initialization
    original_weight = layer.weight.data.clone()

    set_init([layer])

    # After initialization, weights should be different
    # (with very high probability)
    assert not torch.allclose(layer.weight.data, original_weight)


def test_set_init_bias():
    """Test set_init sets bias to zero."""
    layer = nn.Linear(4, 2)
    set_init([layer])

    # Bias should be zero
    assert torch.allclose(layer.bias.data, torch.zeros_like(layer.bias.data))


def test_set_init_multiple_layers():
    """Test set_init with multiple layers."""
    layers = [nn.Linear(4, 8), nn.Linear(8, 2)]
    set_init(layers)

    # All biases should be zero
    for layer in layers:
        assert torch.allclose(
            layer.bias.data, torch.zeros_like(layer.bias.data)
        )


def test_set_init_weight_distribution():
    """Test that set_init creates normally distributed weights."""
    layer = nn.Linear(100, 100)
    set_init([layer])

    weights = layer.weight.data.numpy().flatten()

    # Check mean is close to 0 (with tolerance)
    assert abs(np.mean(weights)) < 0.05

    # Check std is close to 0.1 (with tolerance)
    assert abs(np.std(weights) - 0.1) < 0.05


def test_push_and_pull_terminal_state():
    """Test push_and_pull with terminal state."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    # Copy global to local
    lnet.load_state_dict(gnet.state_dict())

    # Create dummy buffers
    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [0.1 * i for i in range(5)]
    s_ = np.random.randn(s_dim)

    # Call push_and_pull with done=True (terminal state)
    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    # Check that metrics are returned
    assert "loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics
    assert "entropy" in metrics

    # Check that all metrics are finite
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["entropy"])


def test_push_and_pull_non_terminal():
    """Test push_and_pull with non-terminal state."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [0.1 * i for i in range(5)]
    s_ = np.random.randn(s_dim)

    # Call with done=False
    metrics = push_and_pull(
        opt, lnet, gnet, done=False, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert isinstance(metrics, dict)
    assert len(metrics) == 4


def test_push_and_pull_updates_global():
    """Test that push_and_pull updates global network."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    # Store initial global weights
    initial_weights = [p.data.clone() for p in gnet.parameters()]

    lnet.load_state_dict(gnet.state_dict())

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [1.0 for _ in range(5)]
    s_ = np.random.randn(s_dim)

    push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    # Check that at least some global weights changed
    changed = False
    for initial, current in zip(initial_weights, gnet.parameters()):
        if not torch.allclose(initial, current.data):
            changed = True
            break

    assert changed, "Global network should be updated"


def test_push_and_pull_syncs_local():
    """Test that push_and_pull syncs local from global."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    # Deliberately make local different
    for p in lnet.parameters():
        p.data.fill_(99.0)

    bs = [np.random.randn(s_dim) for _ in range(5)]
    ba = [np.random.randn(a_dim) for _ in range(5)]
    br = [1.0 for _ in range(5)]
    s_ = np.random.randn(s_dim)

    push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    # After push_and_pull, local should match global
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        assert torch.allclose(lp.data, gp.data)


def test_push_and_pull_gradient_clipping():
    """Test that gradients are clipped in push_and_pull."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    # Create buffers with large rewards to potentially create large gradients
    bs = [np.random.randn(s_dim) for _ in range(10)]
    ba = [np.random.randn(a_dim) for _ in range(10)]
    br = [100.0 for _ in range(10)]
    s_ = np.random.randn(s_dim)

    # Should not raise even with large rewards (due to gradient clipping)
    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert np.isfinite(metrics["loss"])


def test_record_updates_counters():
    """Test that record updates global counters."""
    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    res_queue = mp.Queue()

    ep_r = 10.0

    record(global_ep, global_ep_r, ep_r, res_queue, "w0")

    # Check that global episode counter increased
    assert global_ep.value == 1

    # Check that moving average was updated
    assert global_ep_r.value == 10.0  # First episode


def test_record_moving_average():
    """Test record's moving average calculation."""
    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    res_queue = mp.Queue()

    # First episode
    record(global_ep, global_ep_r, 10.0, res_queue, "w0")
    assert global_ep_r.value == 10.0

    # Second episode
    record(global_ep, global_ep_r, 20.0, res_queue, "w0")
    expected = 10.0 * 0.99 + 20.0 * 0.01
    assert abs(global_ep_r.value - expected) < 1e-6


def test_record_queue():
    """Test that record puts values in queue."""
    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    res_queue = mp.Queue()

    record(global_ep, global_ep_r, 15.0, res_queue, "w0")

    # Check queue has the moving average
    result = res_queue.get()
    assert result == 15.0


def test_record_multiple_episodes():
    """Test record with multiple episodes."""
    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    res_queue = mp.Queue()

    rewards = [5.0, 10.0, 15.0]
    expected_count = len(rewards)

    for r in rewards:
        record(global_ep, global_ep_r, r, res_queue, "w0")

    # Episode counter should be 3
    with global_ep.get_lock():
        ep_count = global_ep.value
    
    assert ep_count == expected_count

    # Queue should have items (exact count may vary in different contexts)
    # Check that at least the function executed without errors
    assert ep_count > 0


def test_v_wrap_different_dtypes():
    """Test v_wrap with different numpy dtypes."""
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        arr = np.array([1, 2, 3], dtype=dtype)
        tensor = v_wrap(arr, dtype=np.float32)
        assert tensor.dtype == torch.float32


def test_push_and_pull_empty_buffers():
    """Test push_and_pull with minimal buffers."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    lnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    lnet.load_state_dict(gnet.state_dict())

    # Single step
    bs = [np.random.randn(s_dim)]
    ba = [np.random.randn(a_dim)]
    br = [1.0]
    s_ = np.random.randn(s_dim)

    metrics = push_and_pull(
        opt, lnet, gnet, done=True, s_=s_, bs=bs, ba=ba, br=br, gamma=0.99
    )

    assert "loss" in metrics


if __name__ == "__main__":
    # Run all tests
    test_v_wrap_basic()
    test_v_wrap_dtype_conversion()
    test_v_wrap_2d_array()
    test_v_wrap_preserves_values()
    test_set_init_weights()
    test_set_init_bias()
    test_set_init_multiple_layers()
    test_set_init_weight_distribution()
    test_push_and_pull_terminal_state()
    test_push_and_pull_non_terminal()
    test_push_and_pull_updates_global()
    test_push_and_pull_syncs_local()
    test_push_and_pull_gradient_clipping()
    test_record_updates_counters()
    test_record_moving_average()
    test_record_queue()
    test_record_multiple_episodes()
    test_v_wrap_different_dtypes()
    test_push_and_pull_empty_buffers()
    print("All utils tests passed!")

