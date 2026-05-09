"""Squashed-Gaussian actor: shapes, log_prob with tanh Jacobian, range."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.actor import ActorConfig, GaussianActor


def test_rsample_shapes_and_action_range() -> None:
    cfg = ActorConfig(n_state=3, n_action=2, hidden_sizes=(16, 16))
    actor = GaussianActor(cfg)
    s = torch.randn(8, 3)
    a, logp = actor.rsample(s)
    assert a.shape == (8, 2)
    assert logp.shape == (8,)
    assert (a > -1.0 - 1e-6).all() and (a < 1.0 + 1e-6).all()


def test_log_prob_matches_torch_distribution_reference() -> None:
    cfg = ActorConfig(
        n_state=2, n_action=1, hidden_sizes=(8,), log_std_min=-2.0, log_std_max=2.0
    )
    actor = GaussianActor(cfg)
    s = torch.randn(4, 2)
    a, logp = actor.rsample(s)
    # Re-derive log-prob: log N(mean, std) - log(1 - tanh(z)^2)
    mean, log_std = actor(s)
    std = log_std.exp()
    z = torch.atanh(a.clamp(-0.999_999, 0.999_999))
    logp_ref = (
        -0.5 * ((z - mean) / std) ** 2
        - log_std
        - 0.5 * torch.log(torch.tensor(2 * 3.14159265))
    ).sum(dim=-1)
    logp_ref -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
    assert torch.allclose(logp, logp_ref, atol=1e-3)
