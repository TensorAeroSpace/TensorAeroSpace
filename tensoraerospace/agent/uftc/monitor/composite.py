"""CompositeLyapunovMonitor placeholder + dataclasses (filled in Task 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


AlarmLevel = Literal["OK", "WARN", "CRITICAL"]


@dataclass
class VState:
    V_hj: float = 0.0
    V_indi: float = 0.0
    V_iadp: float = 0.0
    V_dsac: float = 0.0
    V_fdd: float = 0.0
    timestamp: float = 0.0


@dataclass
class MonitorConfig:
    c_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    eps_matrix: tuple[tuple[float, ...], ...] = (
        (0.0, 0.05, 0.05, 0.05, 0.05),
        (0.05, 0.0, 0.05, 0.05, 0.05),
        (0.05, 0.05, 0.0, 0.05, 0.05),
        (0.05, 0.05, 0.05, 0.0, 0.05),
        (0.05, 0.05, 0.05, 0.05, 0.0),
    )
    a_diag: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5)
    d_disturbance: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05, 0.05)
    alarm_warn_frac: float = 0.7
    alarm_critical_frac: float = 0.95
    cooldown_steps: int = 200
    burst_factor: float = 1.0


@dataclass
class MonitorOutput:
    V_total: float = 0.0
    components: VState = field(default_factory=VState)
    alarm: AlarmLevel = "OK"
    mu_uub_pred: float = 0.0
    margin: float = 0.0
    interventions: list = field(default_factory=list)

    @classmethod
    def zero(cls) -> "MonitorOutput":
        return cls()
