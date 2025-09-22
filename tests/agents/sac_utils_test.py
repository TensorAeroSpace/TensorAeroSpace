import numpy as np
import torch
import torch.nn as nn

from tensoraerospace.agent.sac.utils import (
    create_log_gaussian,
    hard_update,
    logsumexp,
    soft_update,
)


def test_create_log_gaussian_zero_inputs_matches_analytical():
    mean = torch.zeros((1, 2))
    log_std = torch.zeros((1, 2))  # std = 1
    t = torch.zeros((1, 2))

    out = create_log_gaussian(mean, log_std, t)
    # Expected: -0.5 * D * log(2*pi) for D=2
    expected = -0.5 * 2 * torch.log(torch.tensor(2 * np.pi))
    assert torch.allclose(out, expected.expand_as(out))


def test_logsumexp_matches_torch_reference():
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ref = torch.logsumexp(x, dim=1)
    got = logsumexp(x, dim=1)
    assert torch.allclose(got, ref)


def test_soft_update_blends_parameters():
    source = nn.Linear(2, 2, bias=True)
    target = nn.Linear(2, 2, bias=True)

    with torch.no_grad():
        for p in source.parameters():
            p.copy_(torch.ones_like(p))
        for p in target.parameters():
            p.copy_(torch.zeros_like(p))

    tau = 0.5
    soft_update(target, source, tau)

    with torch.no_grad():
        for tp, _ in zip(target.parameters(), source.parameters()):
            assert torch.allclose(tp, torch.full_like(tp, 0.5))


def test_hard_update_copies_parameters():
    source = nn.Linear(2, 2, bias=True)
    target = nn.Linear(2, 2, bias=True)

    # Initialize differently
    with torch.no_grad():
        for p in source.parameters():
            p.copy_(torch.randn_like(p))
        for p in target.parameters():
            p.copy_(torch.zeros_like(p))

    hard_update(target, source)

    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            assert torch.allclose(tp, sp)
