"""Conformal margin εₜ from FDD severity and monitor alarm.

The shield uses ``εₜ = L · ε_raw(fdd, alarm)`` where ``L`` is an upper
bound on the Lipschitz constant of ``∇V``. ``ε_raw`` aggregates abrupt
and gradual severities, the innovation norm and the monitor alarm
level. Phase 1 ``FDDOutput`` exposes only ``severity`` (used as
``severity_abrupt`` here); ``severity_gradual`` defaults to 0.0 until
Task 6 enriches the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput


@dataclass
class ConformalMarginConfig:
    eps_0: float = 0.05
    k_grad: float = 0.10
    k_abrupt: float = 0.20
    k_innov: float = 0.05
    k_alarm: float = 0.30


_ALARM_GAIN = {"OK": 0.0, "WARN": 0.5, "CRITICAL": 1.0}


class ConformalMargin:
    """Compute εₜ from FDDOutput + monitor alarm."""

    def __init__(self, cfg: ConformalMarginConfig, *, lipschitz_const: float) -> None:
        self.cfg = cfg
        self.lipschitz_const = float(lipschitz_const)

    def compute(self, fdd: FDDOutput, monitor_alarm: str = "OK") -> float:
        sev_abrupt = float(getattr(fdd, "severity_abrupt", fdd.severity))
        sev_grad = float(getattr(fdd, "severity_gradual", 0.0))
        innov = float(getattr(fdd, "innovation_norm", 0.0))
        gain_alarm = _ALARM_GAIN.get(str(monitor_alarm), 0.0)
        eps_raw = (
            self.cfg.eps_0
            + self.cfg.k_grad * sev_grad
            + self.cfg.k_abrupt * sev_abrupt
            + self.cfg.k_innov * innov
            + self.cfg.k_alarm * gain_alarm
        )
        return float(eps_raw * self.lipschitz_const)
