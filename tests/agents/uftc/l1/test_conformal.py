"""ConformalMargin growth law and monotonicity properties."""

from __future__ import annotations

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)


def _zero_output() -> FDDOutput:
    return FDDOutput(
        fault_present=False,
        severity=0.0,
        confidence=0.0,
        innovation_norm=0.0,
        time_since_event=0.0,
    )


def test_baseline_eps_when_fdd_clean() -> None:
    cfg = ConformalMarginConfig()
    cm = ConformalMargin(cfg, lipschitz_const=1.0)
    eps = cm.compute(_zero_output(), monitor_alarm="OK")
    assert abs(eps - cfg.eps_0) < 1e-12


def test_eps_grows_with_severity_and_alarm() -> None:
    cfg = ConformalMarginConfig()
    cm = ConformalMargin(cfg, lipschitz_const=1.0)
    base = cm.compute(_zero_output(), monitor_alarm="OK")

    sev = FDDOutput(
        fault_present=True,
        severity=2.0,
        confidence=0.8,
        innovation_norm=1.5,
        time_since_event=0.0,
    )
    e_sev = cm.compute(sev, monitor_alarm="OK")
    assert e_sev > base

    e_warn = cm.compute(sev, monitor_alarm="WARN")
    e_crit = cm.compute(sev, monitor_alarm="CRITICAL")
    assert e_warn > e_sev
    assert e_crit > e_warn


def test_lipschitz_scales_eps_linearly() -> None:
    cfg = ConformalMarginConfig()
    e1 = ConformalMargin(cfg, lipschitz_const=1.0).compute(_zero_output())
    e3 = ConformalMargin(cfg, lipschitz_const=3.0).compute(_zero_output())
    assert abs(e3 - 3.0 * e1) < 1e-12
