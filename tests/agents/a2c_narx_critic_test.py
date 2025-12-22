"""Tests for NARX critic module.

This module contains unit tests for the NARX (Nonlinear AutoRegressive with
exogenous inputs) critic network and feature building utilities.
"""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.a2c.narx_critic import (
    NARXCritic,
    build_narx_features,
)


def test_build_narx_features_shape():
    """Test that build_narx_features produces correct output shape."""
    T, obs_dim, act_dim, h = 10, 3, 2, 4
    states = np.random.randn(T, obs_dim).astype(np.float32)
    actions = np.random.randn(T, act_dim).astype(np.float32)

    features = build_narx_features(states, actions, history_length=h)

    expected_dim = h * obs_dim + h * act_dim
    assert features.shape == (T, expected_dim)
    assert features.dtype == torch.float32


def test_build_narx_features_with_tensors():
    """Test build_narx_features with torch.Tensor inputs."""
    T, obs_dim, act_dim, h = 5, 2, 1, 2
    states = torch.randn(T, obs_dim)
    actions = torch.randn(T, act_dim)

    features = build_narx_features(states, actions, history_length=h)

    expected_dim = h * obs_dim + h * act_dim
    assert features.shape == (T, expected_dim)


def test_build_narx_features_zero_padding():
    """Test that early timesteps are zero-padded correctly."""
    T, obs_dim, act_dim, h = 3, 2, 1, 2
    states = torch.ones(T, obs_dim)
    actions = torch.ones(T, act_dim)

    features = build_narx_features(states, actions, history_length=h)

    # At t=0, x_{t-1} and u_{t-1}, u_{t-2} should be zero
    # features[0] = [x_0, x_{-1}, u_{-1}, u_{-2}]
    # x_0 = [1,1], x_{-1} = [0,0], u_{-1} = [0], u_{-2} = [0]
    assert features[0, 0] == 1.0  # x_0[0]
    assert features[0, 1] == 1.0  # x_0[1]
    assert features[0, 2] == 0.0  # x_{-1}[0]
    assert features[0, 3] == 0.0  # x_{-1}[1]
    assert features[0, 4] == 0.0  # u_{-1}
    assert features[0, 5] == 0.0  # u_{-2}


def test_build_narx_features_invalid_history():
    """Test that history_length < 1 raises assertion."""
    states = np.zeros((5, 2))
    actions = np.zeros((5, 1))

    with pytest.raises(AssertionError):
        build_narx_features(states, actions, history_length=0)


def test_build_narx_features_dimension_mismatch():
    """Test that mismatched T raises ValueError."""
    states = np.zeros((5, 2))
    actions = np.zeros((3, 1))

    with pytest.raises(ValueError, match="same T length"):
        build_narx_features(states, actions, history_length=2)


def test_build_narx_features_wrong_dims():
    """Test that 1D inputs raise ValueError."""
    states = np.zeros(5)
    actions = np.zeros(5)

    with pytest.raises(ValueError, match="2D tensors"):
        build_narx_features(states, actions, history_length=2)


def test_narx_critic_initialization():
    """Test NARXCritic initialization."""
    obs_dim, act_dim, h = 3, 2, 4
    critic = NARXCritic(
        observation_dim=obs_dim,
        action_dim=act_dim,
        history_length=h,
        hidden_sizes=(64, 64),
    )

    assert critic.observation_dim == obs_dim
    assert critic.action_dim == act_dim
    assert critic.history_length == h
    assert isinstance(critic.model, torch.nn.Sequential)


def test_narx_critic_forward():
    """Test NARXCritic forward pass."""
    obs_dim, act_dim, h = 3, 2, 4
    batch_size = 8
    critic = NARXCritic(
        observation_dim=obs_dim,
        action_dim=act_dim,
        history_length=h,
        hidden_sizes=(32,),
    )
    device = next(critic.model.parameters()).device

    feature_dim = h * (obs_dim + act_dim)
    features = torch.randn(batch_size, feature_dim, device=device)

    values = critic(features)

    assert values.shape == (batch_size, 1)
    assert values.dtype == torch.float32


def test_narx_critic_with_built_features():
    """Test end-to-end: build features and pass through critic."""
    T, obs_dim, act_dim, h = 10, 3, 2, 3
    states = np.random.randn(T, obs_dim).astype(np.float32)
    actions = np.random.randn(T, act_dim).astype(np.float32)

    features = build_narx_features(states, actions, history_length=h)

    critic = NARXCritic(
        observation_dim=obs_dim,
        action_dim=act_dim,
        history_length=h,
        hidden_sizes=(16,),
    )

    values = critic(features)

    assert values.shape == (T, 1)


def test_narx_critic_custom_activation():
    """Test NARXCritic with custom activation function."""
    obs_dim, act_dim, h = 2, 1, 2
    critic = NARXCritic(
        observation_dim=obs_dim,
        action_dim=act_dim,
        history_length=h,
        hidden_sizes=(8,),
        activation=torch.nn.ReLU,
    )
    device = next(critic.model.parameters()).device

    feature_dim = h * (obs_dim + act_dim)
    features = torch.randn(4, feature_dim, device=device)

    values = critic(features)

    assert values.shape == (4, 1)


def test_narx_critic_gradient_flow():
    """Test that gradients flow through NARXCritic."""
    obs_dim, act_dim, h = 2, 1, 2
    critic = NARXCritic(
        observation_dim=obs_dim,
        action_dim=act_dim,
        history_length=h,
        hidden_sizes=(8,),
    )
    device = next(critic.model.parameters()).device

    feature_dim = h * (obs_dim + act_dim)
    features = torch.randn(4, feature_dim, requires_grad=True, device=device)

    values = critic(features)
    loss = values.sum()
    loss.backward()

    assert features.grad is not None
    assert features.grad.shape == features.shape


def test_build_narx_features_device_consistency():
    """Test that build_narx_features preserves device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    T, obs_dim, act_dim, h = 5, 2, 1, 2
    device = torch.device("cuda")
    states = torch.randn(T, obs_dim, device=device)
    actions = torch.randn(T, act_dim, device=device)

    features = build_narx_features(states, actions, history_length=h)

    # `torch.device("cuda")` has no explicit index (None), while tensors are
    # typically allocated on an indexed device (e.g. cuda:0). Compare to the
    # actual input tensor device to avoid false negatives.
    assert features.device == states.device
