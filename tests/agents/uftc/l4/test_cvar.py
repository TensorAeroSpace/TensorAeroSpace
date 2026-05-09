"""cvar_alpha_fn: tail-mean correctness; risk_gate: monotonicity."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.cvar import cvar_alpha_fn, risk_gate


def test_cvar_matches_numpy_tail_mean() -> None:
    rng = np.random.default_rng(0)
    z_np = rng.standard_normal((4, 32))
    z = torch.tensor(z_np, dtype=torch.float64)
    alpha = 0.25
    out = cvar_alpha_fn(z, alpha).numpy()
    z_sorted = np.sort(z_np, axis=-1)
    k = int(np.floor(alpha * 32))
    expected = z_sorted[:, :k].mean(axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_cvar_grad_flows_back() -> None:
    z = torch.randn(2, 16, requires_grad=True)
    out = cvar_alpha_fn(z, 0.25)
    out.sum().backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_risk_gate_monotone_in_each_input() -> None:
    z_low = torch.randn(2, 16) * 0.1
    z_hi = torch.randn(2, 16) * 5.0  # high variance
    g_low = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="OK")
    g_hi = risk_gate(z_hi, fdd_severity=0.0, monitor_alarm="OK")
    assert g_hi >= g_low

    g_fdd = risk_gate(z_low, fdd_severity=1.0, monitor_alarm="OK")
    assert g_fdd > g_low

    g_alarm_warn = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="WARN")
    g_alarm_crit = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="CRITICAL")
    assert g_alarm_warn >= 0.5
    assert g_alarm_crit >= 1.0 - 1e-9
