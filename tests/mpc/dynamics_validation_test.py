import pytest
import torch

from tensoraerospace.agent.mpc.dynamics import DynamicsNN


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 4)

    def forward(self, x):
        return self.fc(x)


def test_generate_training_data_validation_errors():
    dyn = DynamicsNN(_TinyModel())

    with pytest.raises(ValueError):
        dyn.generate_training_data(
            state_ranges=None, A=torch.eye(4), B=torch.ones(4, 1)
        )

    with pytest.raises(ValueError):
        dyn.generate_training_data(
            state_ranges=[(-1, 1)] * 4, A=None, B=torch.ones(4, 1)
        )

    with pytest.raises(ValueError):
        dyn.generate_training_data(state_ranges=[(-1, 1)] * 4, A=torch.eye(4), B=None)

    with pytest.raises(ValueError):
        dyn.generate_training_data(
            state_ranges=[(-1, 1)] * 3, A=torch.eye(4), B=torch.ones(4, 1)
        )
