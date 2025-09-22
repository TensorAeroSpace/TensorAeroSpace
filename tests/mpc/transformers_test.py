import torch

from tensoraerospace.agent.mpc.transformers import (
    PositionalEncoding,
    TransformerDynamicsModel,
)


def test_positional_encoding_shapes():
    pe = PositionalEncoding(d_model=6, dropout=0.0, max_len=10)
    x = torch.zeros(5, 2, 6)  # (seq_len, batch, d_model)
    y = pe(x)
    assert y.shape == x.shape


def test_transformer_dynamics_forward():
    model = TransformerDynamicsModel(
        input_dim=4,
        output_dim=2,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)
