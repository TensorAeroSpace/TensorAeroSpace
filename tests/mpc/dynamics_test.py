import numpy as np
import torch

from tensoraerospace.agent.mpc.dynamics import DynamicsNN


class _TinyModel(torch.nn.Module):
    def __init__(self, in_dim=5, out_dim=4):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def test_generate_training_data_and_predict():
    model = _TinyModel(in_dim=5, out_dim=4)
    dyn = DynamicsNN(model)

    state_dim = 4
    control_dim = 1
    A = torch.eye(state_dim)
    B = torch.ones((state_dim, control_dim)) * 0.1
    state_ranges = [(-1, 1)] * state_dim

    states, controls, next_states = dyn.generate_training_data(
        num_samples=64,
        state_dim=state_dim,
        control_dim=control_dim,
        state_ranges=state_ranges,
        control_ranges=[(-1, 1)],
        A=A,
        B=B,
        control_signals=["sine"],
    )

    # shapes
    assert states.shape == (64, state_dim)
    assert controls.shape == (64, control_dim)
    assert next_states.shape == (64, state_dim)

    # predict uses torch tensors concatenated
    pred = dyn.predict(states[:2], controls[:2])
    assert pred.shape == (2, state_dim)
