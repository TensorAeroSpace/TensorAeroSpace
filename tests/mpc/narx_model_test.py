import pytest
import torch

from tensoraerospace.agent.mpc.narx import NARX, NARXDynamicsModel


def test_narx_forward_shapes():
    model = NARX(
        input_size=6,
        hidden_size=8,
        output_size=2,
        num_layers=2,
        state_lags=2,
        control_lags=1,
    )
    state = torch.randn(4, 4)  # batch=4, concatenated state lags
    control = torch.randn(4, 2)  # batch=4, concatenated control lags
    y = model(state, control)
    assert y.shape == (4, 2)


def test_narx_supports_batch_dim():
    model = NARX(
        input_size=6,
        hidden_size=4,
        output_size=2,
        num_layers=1,
        state_lags=1,
        control_lags=1,
    )
    # Contract: inputs are concatenated features (batch, features)
    state = torch.randn(8, 4)  # batch=8, features=state_dim*state_lags
    control = torch.randn(8, 2)  # batch=8, features=action_dim*control_lags
    y = model(state, control)
    assert y.shape == (8, 2)


def test_narx_dynamics_model_splits_and_reshapes():
    dyn = NARXDynamicsModel(
        state_dim=3,
        action_dim=2,
        hidden_size=10,
        num_layers=1,
        state_lags=2,
        control_lags=2,
    )
    # xu shape (B, state_dim*lags + action_dim*lags)
    xu = torch.randn(5, 3 * 2 + 2 * 2)
    y = dyn(xu)
    assert y.shape == (5, 3)


def test_narx_dynamics_model_validates_dims():
    with pytest.raises(ValueError):
        NARXDynamicsModel(state_dim=0, action_dim=1)
    with pytest.raises(ValueError):
        NARXDynamicsModel(state_dim=1, action_dim=0)
    with pytest.raises(ValueError):
        NARXDynamicsModel(state_dim=1, action_dim=1, state_lags=0)
