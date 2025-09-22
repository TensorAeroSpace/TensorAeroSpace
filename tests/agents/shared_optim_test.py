import torch
import torch.nn as nn

from tensoraerospace.agent.a3c.shared_optim import SharedAdam


def test_shared_adam_initializes_and_steps():
    model = nn.Linear(4, 2)
    opt = SharedAdam(model.parameters(), lr=1e-2)

    # Simple forward/backward
    x = torch.randn(3, 4)
    y = model(x).sum()
    y.backward()

    # Optimizer step should not error with shared state
    opt.step()
    opt.zero_grad()
