import pytest
import torch

from tensoraerospace.agent.mpc.transformers import TransformerDynamicsModel


def test_transformer_forward_shapes():
    model = TransformerDynamicsModel(
        input_dim=5,
        output_dim=4,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=32,
        dropout=0.1,
        seq_len=1,
    )
    # Contract: input is (batch, input_dim); model unsqueezes seq_len=1 internally
    xu = torch.randn(3, 5)
    y = model(xu)
    assert y.shape == (3, 4)


def test_transformer_handles_batch():
    model = TransformerDynamicsModel(
        input_dim=5,
        output_dim=4,
        d_model=8,
        nhead=1,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        seq_len=1,
    )
    xu = torch.randn(6, 5)  # batch=6, features=5
    y = model(xu)
    assert y.shape == (6, 4)


def test_transformer_rejects_wrong_dim():
    model = TransformerDynamicsModel(
        input_dim=5,
        output_dim=4,
        d_model=8,
        nhead=1,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        seq_len=1,
    )
    # Wrong feature size triggers matmul shape error in embedding
    bad = torch.randn(2, 3)  # last dim != input_dim
    with pytest.raises(RuntimeError):
        _ = model(bad)
