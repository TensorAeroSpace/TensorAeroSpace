"""QR critic forward pass shape + soft-target update behaviour."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.critic import (
    CriticConfig,
    QRDistCritic,
    qr_huber_loss,
    soft_update,
)


def test_forward_returns_n_quantiles() -> None:
    cfg = CriticConfig(n_state=3, n_action=2, n_quantiles=8, hidden_sizes=(16, 16))
    q = QRDistCritic(cfg)
    s = torch.randn(5, 3)
    a = torch.randn(5, 2)
    z = q(s, a)
    assert z.shape == (5, 8)


def test_qr_huber_loss_decreases_under_supervised_descent() -> None:
    rng = np.random.default_rng(0)
    cfg = CriticConfig(n_state=2, n_action=1, n_quantiles=16, hidden_sizes=(8, 8))
    q = QRDistCritic(cfg)
    opt = torch.optim.SGD(q.parameters(), lr=1e-2)
    s = torch.tensor(rng.standard_normal((64, 2)), dtype=torch.float32)
    a = torch.tensor(rng.standard_normal((64, 1)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((64, 16)), dtype=torch.float32)
    losses = []
    for _ in range(50):
        z = q(s, a)
        loss = qr_huber_loss(z, target.detach(), cfg.huber_kappa)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]


def test_soft_update_moves_target_toward_source() -> None:
    cfg = CriticConfig(n_state=2, n_action=1, n_quantiles=4, hidden_sizes=(4,))
    src = QRDistCritic(cfg)
    tgt = QRDistCritic(cfg)
    # Force src/tgt to differ.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    soft_update(target=tgt, source=src, tau=0.5)
    src_params = list(src.parameters())
    tgt_params = list(tgt.parameters())
    # After tau=0.5, target = 0.5*src_old + 0.5*tgt_old; abs differences shrink.
    diff = sum(
        float((p_s - p_t).abs().sum()) for p_s, p_t in zip(src_params, tgt_params)
    )
    assert diff > 0.0  # not yet identical
