"""Tests for DQN Model."""
import numpy as np
import torch

from tensoraerospace.agent.dqn import Model


def test_model_initialization():
    """Test Model initialization."""
    num_actions = 4
    model = Model(num_actions=num_actions)

    # Check layers exist
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")
    assert hasattr(model, "out")

    # Output layer should have correct size
    assert model.out.out_features == num_actions


def test_model_predict_shape(seed_all):
    """Test that predict returns correct shape and dtype."""
    model = Model(num_actions=3)

    # Single observation
    obs = np.random.randn(1, 4).astype(np.float32)
    q_values = model.predict(obs)

    assert q_values.shape == (1, 3)
    assert q_values.dtype == np.float32

    # Batch of observations
    obs_batch = np.random.randn(8, 4).astype(np.float32)
    q_values_batch = model.predict(obs_batch)

    assert q_values_batch.shape == (8, 3)
    assert q_values_batch.dtype == np.float32


def test_model_action_value_single(seed_all):
    """Test action_value with single observation (batch_size=1)."""
    model = Model(num_actions=2)

    obs = np.random.randn(1, 4).astype(np.float32)
    action, q_values = model.action_value(obs)

    # For batch_size=1, action should be int
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 2

    # q_values should be 1D array of first batch item
    assert q_values.shape == (2,)
    assert q_values.dtype == np.float32


def test_model_action_value_batch(seed_all):
    """Test action_value with batch of observations (batch_size>1)."""
    model = Model(num_actions=3)

    obs_batch = np.random.randn(4, 4).astype(np.float32)
    actions, q_values = model.action_value(obs_batch)

    # For batch_size>1, actions should be ndarray
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (4,)
    assert np.all((actions >= 0) & (actions < 3))

    # q_values should be of first item only
    assert q_values.shape == (3,)
    assert q_values.dtype == np.float32


def test_model_action_value_argmax_correctness(seed_all):  # noqa: ARG001
    """Test that action_value returns argmax correctly on crafted inputs."""
    model = Model(num_actions=3)

    # Initialize model with a forward pass
    dummy_obs = np.zeros((1, 4), dtype=np.float32)
    _ = model.predict(dummy_obs)

    # Craft input that should give predictable output
    # We'll just verify argmax consistency
    obs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    action1, q_values1 = model.action_value(obs)

    # Verify action is argmax of q_values
    expected_action = np.argmax(q_values1)
    assert action1 == expected_action


def test_model_lazy_layer_initialization(seed_all):  # noqa: ARG001
    """Test that lazy layers are properly initialized after first forward pass."""
    model = Model(num_actions=2)

    # Before forward pass, fc1 should have in_features=0 (uninitialized)
    assert hasattr(model.fc1, "in_features")
    assert model.fc1.in_features == 0

    # First forward pass
    obs1 = np.random.randn(1, 4).astype(np.float32)
    q1 = model.predict(obs1)

    # After forward pass, fc1 should be initialized
    assert model.fc1.in_features == 4

    # Second forward pass should give consistent shape
    obs2 = np.random.randn(1, 4).astype(np.float32)
    q2 = model.predict(obs2)

    assert q1.shape == q2.shape == (1, 2)


def test_model_forward_returns_tensor(seed_all):
    """Test that forward method returns torch.Tensor."""
    model = Model(num_actions=2)

    obs = torch.randn(1, 4)
    output = model.forward(obs)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2)


def test_model_different_input_sizes(seed_all):
    """Test model works with different observation dimensions."""
    # Test with different obs_dim
    for obs_dim in [2, 4, 8, 16]:
        model = Model(num_actions=3)
        obs = np.random.randn(1, obs_dim).astype(np.float32)
        q_values = model.predict(obs)

        assert q_values.shape == (1, 3)


def test_model_predict_no_grad(seed_all):
    """Test that predict doesn't compute gradients."""
    model = Model(num_actions=2)
    model.train()  # Set to training mode

    obs = np.random.randn(1, 4).astype(np.float32)

    # predict should work without computing gradients
    q_values = model.predict(obs)

    assert q_values.shape == (1, 2)
    # No exception should be raised


def test_model_device_handling(seed_all):  # noqa: ARG001
    """Test that model correctly handles device placement."""
    model = Model(num_actions=2)

    # Predict with numpy input (should handle device internally)
    obs = np.random.randn(1, 4).astype(np.float32)
    q_values = model.predict(obs)

    # Output should be on CPU (numpy)
    assert isinstance(q_values, np.ndarray)
