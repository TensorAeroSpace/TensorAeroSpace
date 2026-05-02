"""Outer-loop reference models for the AIDI controller.

Each block produces a per-axis "desired rate" that the linear controller
combines into the virtual-control vector ``ν``. PCH hedge signals are
subtracted from the integrator-style blocks (`CStarController`,
`RollReferenceModel`, `SideslipCompensator`) before integration, in line
with §III.A of Ul Haq et al. 2026.
"""

from __future__ import annotations

import numpy as np


class CStarController:
    """C\\* longitudinal controller — PI on ``C*_cmd − C*``.

    ``C* = n_z + (V/V_co)·q``. The output is a desired pitch rate ``q_des``.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        V_co: Crossover speed (m/s); MIL-STD value is ≈ 122.6.
        dt: Integration step (s).
        i_clip: Symmetric anti-windup clamp on the integrator state
            (in error·s units).
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.5,
        V_co: float = 122.6,
        dt: float = 0.01,
        i_clip: float = 5.0,
    ) -> None:
        if V_co <= 0.0:
            raise ValueError("V_co must be positive")
        self.kp = float(kp)
        self.ki = float(ki)
        self.V_co = float(V_co)
        self.dt = float(dt)
        self.i_clip = float(i_clip)
        self._int_err: float = 0.0

    def reset(self) -> None:
        self._int_err = 0.0

    def step(
        self,
        c_star_cmd: float,
        n_z: float,
        q: float,
        V: float,
        hedge: float = 0.0,
    ) -> float:
        c_star = float(n_z) + (float(V) / self.V_co) * float(q)
        err = float(c_star_cmd) - c_star
        # PCH: subtract the hedge BEFORE integration to keep the integrator
        # inside the achievable envelope.
        self._int_err = float(
            np.clip(
                self._int_err + (err - float(hedge)) * self.dt,
                -self.i_clip,
                self.i_clip,
            )
        )
        return self.kp * err + self.ki * self._int_err


class RollReferenceModel:
    """Second-order roll attitude reference model.

    ``phi_ddot = -2ζω_n·phi_dot + ω_n²·(phi_cmd − phi)``; output is the
    integrated ``phi_dot``, used as ``p_des``.
    """

    def __init__(
        self,
        omega_n: float = 2.0,
        zeta: float = 0.7,
        dt: float = 0.01,
    ) -> None:
        if omega_n <= 0.0 or zeta <= 0.0:
            raise ValueError("omega_n > 0 and zeta > 0 required")
        self.omega_n = float(omega_n)
        self.zeta = float(zeta)
        self.dt = float(dt)
        self._phi_dot: float = 0.0
        self._phi: float = 0.0

    def reset(self) -> None:
        self._phi_dot = 0.0
        self._phi = 0.0

    def step(self, phi_cmd: float, phi: float, hedge: float = 0.0) -> float:
        phi_ddot = (
            -2.0 * self.zeta * self.omega_n * self._phi_dot
            + self.omega_n**2 * (float(phi_cmd) - float(phi))
            - float(hedge)
        )
        self._phi_dot = self._phi_dot + self.dt * phi_ddot
        self._phi = self._phi + self.dt * self._phi_dot
        return self._phi_dot


class SideslipCompensator:
    """PI compensator on sideslip ``β`` driving a yaw-rate demand ``r_des``.

    Body-axis sign convention: ``β̇ ≈ −r·cos α``, so a positive yaw rate
    drives positive sideslip toward zero. The compensator therefore
    outputs ``r_des = +kp·(β − β_cmd) + ki·∫(β − β_cmd)`` — i.e. the gain
    is applied to the *negated* tracking error so positive ``kp`` means
    "suppress sideslip" rather than "track β_cmd via yaw rate".
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        dt: float = 0.01,
        i_clip: float = 5.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.i_clip = float(i_clip)
        self._int_err: float = 0.0

    def reset(self) -> None:
        self._int_err = 0.0

    def step(self, beta_cmd: float, beta: float, hedge: float = 0.0) -> float:
        # neg_err = β − β_cmd : positive when sideslip exceeds command.
        neg_err = float(beta) - float(beta_cmd)
        self._int_err = float(
            np.clip(
                self._int_err + (neg_err - float(hedge)) * self.dt,
                -self.i_clip,
                self.i_clip,
            )
        )
        return self.kp * neg_err + self.ki * self._int_err


class SpeedController:
    """Auto-throttle PID. No-op when ``enabled=False`` (constant-airspeed envs)."""

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.01,
        enabled: bool = False,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.enabled = bool(enabled)
        self._int_err = 0.0
        self._prev_err = 0.0

    def reset(self) -> None:
        self._int_err = 0.0
        self._prev_err = 0.0

    def step(self, V_cmd: float, V: float) -> float:
        if not self.enabled:
            return 0.0
        err = float(V_cmd) - float(V)
        self._int_err += err * self.dt
        deriv = (err - self._prev_err) / max(self.dt, 1e-9)
        self._prev_err = err
        return self.kp * err + self.ki * self._int_err + self.kd * deriv


class LinearController:
    """Combine the outer-loop desired rates into a virtual-control vector.

    ``ν = ω_des + K_p ⊙ (ω_des − ω)``. With ``K_p = 0`` this is a passthrough.
    """

    def __init__(
        self,
        rate_kp: np.ndarray | None = None,
        n_y: int = 3,
    ) -> None:
        if rate_kp is None:
            rate_kp = np.zeros(n_y, dtype=np.float64)
        rate_kp = np.asarray(rate_kp, dtype=np.float64).reshape(-1)
        self.rate_kp = rate_kp

    def combine(
        self,
        omega_des: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega_des = np.asarray(omega_des, dtype=np.float64).reshape(-1)
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        return omega_des + self.rate_kp * (omega_des - omega)
