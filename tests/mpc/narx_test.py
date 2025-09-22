import torch

from tensoraerospace.agent.mpc.narx import NARX


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
