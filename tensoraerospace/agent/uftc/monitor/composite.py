"""Composite Lyapunov monitor — Variant B (advisory + macro-actions)."""

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


class CompositeLyapunovMonitor:
    def __init__(self, cfg: MonitorConfig) -> None:
        from .alarm import AlarmStateMachine

        self.cfg = cfg
        c = np.asarray(cfg.c_weights, dtype=np.float64)
        a = np.asarray(cfg.a_diag, dtype=np.float64)
        eps = np.asarray(cfg.eps_matrix, dtype=np.float64)
        d = np.asarray(cfg.d_disturbance, dtype=np.float64)
        if c.shape != (5,) or a.shape != (5,) or eps.shape != (5, 5) or d.shape != (5,):
            raise ValueError("MonitorConfig must describe a 5-component system")
        self._c, self._a, self._eps, self._d = c, a, eps, d
        M = np.diag(a) - eps
        # Closed-form mu_uub = ‖M^{-1} d‖_c
        try:
            sol = np.linalg.solve(M, d)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(M) @ d
        self.mu_uub_pred = float(np.dot(c, np.abs(sol)))
        self._alarm = AlarmStateMachine(cooldown_steps=cfg.cooldown_steps)

    def step(self, vstate: VState) -> MonitorOutput:
        v_vec = np.array(
            [vstate.V_hj, vstate.V_indi, vstate.V_iadp, vstate.V_dsac, vstate.V_fdd],
            dtype=np.float64,
        )
        V_total = float(self._c @ v_vec)
        level = self._alarm.update(
            V_total=V_total,
            mu_uub=self.mu_uub_pred,
            warn_frac=self.cfg.alarm_warn_frac,
            crit_frac=self.cfg.alarm_critical_frac,
        )
        margin = float(self.mu_uub_pred - V_total)
        interventions = self._build_interventions(level, V_total)
        return MonitorOutput(
            V_total=V_total,
            components=vstate,
            alarm=level,
            mu_uub_pred=self.mu_uub_pred,
            margin=margin,
            interventions=interventions,
        )

    def _build_interventions(self, level: AlarmLevel, V_total: float):
        from .intervention import MacroAction

        actions: list = []
        if level == "WARN":
            actions.append(
                MacroAction(
                    kind="freeze_l4_learning",
                    payload={"duration": int(self.cfg.cooldown_steps)},
                )
            )
        elif level == "CRITICAL":
            actions.append(
                MacroAction(kind="force_rls_reset", payload={"severity": 1.0})
            )
            actions.append(
                MacroAction(
                    kind="freeze_l4_learning",
                    payload={"duration": int(2 * self.cfg.cooldown_steps)},
                )
            )
            actions.append(MacroAction(kind="degrade_reference_to_hold"))
            if V_total > self.mu_uub_pred * self.cfg.burst_factor:
                actions.append(MacroAction(kind="request_actuator_hold"))
        return actions

    def reset(self) -> None:
        self._alarm.reset()
