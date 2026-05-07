"""Inner-loop (L2) extensions for UFTC: SM observer, mode switch,
trust-region wrapper around aa_indi.AAINDIAgent.

Phase 1 MVP — see docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from tensoraerospace.agent.aa_indi.model import AAINDIAgent


class SuperTwistingObserver:
    """Higher-order sliding-mode observer (super-twisting algorithm).

    Estimates the unmodeled high-frequency disturbance δ̂ on each angular
    axis from the residual ``s = ω̇_meas − ν_des − δ̂``::

        ṡ = −k₁·|s|^{1/2}·sign(s) + z
        ż = −k₂·sign(s)

    Args:
        n_axes: Number of angular axes (typically 3 for an aircraft).
        k1: Outer super-twisting gain.
        k2: Inner super-twisting gain.
        dt: Sampling period [s].
    """

    def __init__(
        self,
        n_axes: int,
        *,
        k1: float = 3.0,
        k2: float = 1.5,
        dt: float = 0.01,
    ) -> None:
        self.n_axes = int(n_axes)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        """Clear observer state."""
        self._s = np.zeros(self.n_axes, dtype=np.float64)
        self._z = np.zeros(self.n_axes, dtype=np.float64)

    def update(
        self,
        omega_dot_meas: np.ndarray,
        nu_des: np.ndarray,
    ) -> np.ndarray:
        """Run one observer step. Returns δ̂ ≈ (ω̇_meas − ν_des)."""
        wd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        nd = np.asarray(nu_des, dtype=np.float64).reshape(-1)
        if wd.size != self.n_axes:
            raise ValueError(
                f"omega_dot_meas must have length {self.n_axes}, got {wd.size}"
            )
        if nd.size != self.n_axes:
            raise ValueError(
                f"nu_des must have length {self.n_axes}, got {nd.size}"
            )

        # Discrete-time Euler integration of the super-twisting law.
        # The sliding variable σ = s − e drives s toward e = ω̇_meas − ν_des.
        # At convergence σ → 0, so s → e and s IS the disturbance estimate.
        e = wd - nd
        sigma = self._s - e  # sliding variable
        sgn = np.sign(sigma)
        abs_term = np.sqrt(np.abs(sigma))
        ds = -self.k1 * abs_term * sgn + self._z
        dz = -self.k2 * sgn

        self._s = self._s + self.dt * ds
        self._z = self._z + self.dt * dz

        # δ̂ = s; the observer state converges to e in finite time.
        return self._s.copy()


class ModeSwitcher:
    """Hysteretic rate-INDI ↔ angle-INDI mode selector.

    Args:
        alpha_threshold_deg: AoA above which the mode switches to angle-INDI.
        hysteresis_deg: AoA must drop below ``alpha_threshold_deg − hysteresis_deg``
            to switch back to rate-INDI.
    """

    def __init__(
        self,
        alpha_threshold_deg: float = 25.0,
        hysteresis_deg: float = 5.0,
    ) -> None:
        if hysteresis_deg < 0.0:
            raise ValueError("hysteresis_deg must be non-negative")
        self.alpha_threshold = float(np.deg2rad(alpha_threshold_deg))
        self.alpha_clear = float(np.deg2rad(alpha_threshold_deg - hysteresis_deg))
        self.reset()

    def reset(self) -> None:
        """Return to the default rate-INDI mode."""
        self._mode: Literal["rate", "angle"] = "rate"

    def select(self, alpha_rad: float) -> Literal["rate", "angle"]:
        """Update mode given current α and return new mode label."""
        a = float(alpha_rad)
        if self._mode == "rate" and a > self.alpha_threshold:
            self._mode = "angle"
        elif self._mode == "angle" and a < self.alpha_clear:
            self._mode = "rate"
        return self._mode

    @property
    def mode(self) -> Literal["rate", "angle"]:
        return self._mode


class WrappedAAINDI:
    """Bounded trust-region wrapper around :class:`AAINDIAgent`.

    Adds three behaviours on top of the AAINDI base:

    1. **Sliding-mode disturbance compensation.** A super-twisting
       observer estimates δ̂ from the residual ``ω̇_meas − ν_des`` and
       feeds a small additive correction into ω_meas before AAINDI
       runs.
    2. **Rate-INDI / angle-INDI mode switching.** The mode-switch label
       is exposed via ``self.mode``; the underlying AAINDI law is the
       same in MVP (true angle-INDI law lands in Phase 2).
    3. **Bounded trust-region.** Final command is clipped to a ball of
       radius ``δ`` around ``u_blend_target`` (the L3 absolute-control
       output). Radius interpolates linearly between
       ``trust_radius_nominal`` and ``trust_radius_fault`` driven by
       ``fault_severity ∈ [0, 1]`` (clipped if outside).
    """

    def __init__(
        self,
        base: AAINDIAgent,
        *,
        sm_obs: SuperTwistingObserver,
        mode_switch: ModeSwitcher,
        trust_radius_nominal: float = 0.1,
        trust_radius_fault: float = 0.5,
        dt: float = 0.01,
    ) -> None:
        if trust_radius_nominal <= 0.0 or trust_radius_fault <= 0.0:
            raise ValueError("trust radii must be positive")
        if trust_radius_fault < trust_radius_nominal:
            raise ValueError(
                "trust_radius_fault must be ≥ trust_radius_nominal"
            )
        self.base = base
        self.sm_obs = sm_obs
        self.mode_switch = mode_switch
        self.trust_radius_nominal = float(trust_radius_nominal)
        self.trust_radius_fault = float(trust_radius_fault)
        self.dt = float(dt)
        self._n_state = base.n_state
        self._n_control = base.n_control
        self._prev_omega: np.ndarray | None = None

    def predict(
        self,
        omega_ref: np.ndarray,
        omega_meas: np.ndarray,
        *,
        alpha: float,
        u_blend_target: np.ndarray,
        fault_severity: float,
        time_step: int,
    ) -> np.ndarray:
        """One control-tick from a rate target → constrained INDI action."""
        omega_meas_v = np.asarray(omega_meas, dtype=np.float64).reshape(-1)
        omega_ref_v = np.asarray(omega_ref, dtype=np.float64).reshape(-1)
        u_target = np.asarray(u_blend_target, dtype=np.float64).reshape(-1)
        if omega_meas_v.size != self._n_state:
            raise ValueError(f"omega_meas size mismatch: {omega_meas_v.size}")
        if omega_ref_v.size != self._n_state:
            raise ValueError(f"omega_ref size mismatch: {omega_ref_v.size}")
        if u_target.size != self._n_control:
            raise ValueError(f"u_blend_target size mismatch: {u_target.size}")

        # SM observer with omega_dot ~ (ω_meas − ω_meas_prev)/dt as a
        # first-order proxy for the unmeasured angular acceleration; the
        # ν_des proxy is (ω_ref − ω_meas)/dt — the rate gap that the
        # inner loop has to close in one tick.
        if self._prev_omega is None:
            omega_dot_proxy = np.zeros_like(omega_meas_v)
        else:
            omega_dot_proxy = (omega_meas_v - self._prev_omega) / self.dt
        self._prev_omega = omega_meas_v.copy()
        nu_des_proxy = (omega_ref_v - omega_meas_v) / max(self.dt, 1e-9)
        delta_hat = self.sm_obs.update(omega_dot_proxy, nu_des_proxy)

        # Mode update (label only — MVP uses one allocator).
        self.mode_switch.select(alpha)

        # SM disturbance compensation in measurement space.
        omega_corrected = omega_meas_v - delta_hat * self.dt

        u_indi_raw = self.base.predict(omega_corrected, omega_ref_v,
                                       time_step=time_step)

        # Trust-region clip targeting L3 absolute control.
        sev = float(np.clip(fault_severity, 0.0, 1.0))
        radius = (self.trust_radius_nominal +
                  (self.trust_radius_fault - self.trust_radius_nominal) * sev)
        diff = u_indi_raw - u_target
        norm = float(np.linalg.norm(diff))
        if norm > radius and norm > 0.0:
            u_indi = u_target + diff * (radius / norm)
        else:
            u_indi = u_indi_raw

        # Re-clip against AAINDI magnitude limit (u_target may already exceed it).
        u_indi = np.clip(u_indi,
                         -self.base.cfg.u_magnitude_limit,
                         self.base.cfg.u_magnitude_limit)
        return u_indi

    def learn(
        self,
        next_omega: np.ndarray,
        omega_ref: np.ndarray,
        time_step: int,
    ) -> dict:
        """Forward to AAINDI's learn step."""
        return self.base.learn(next_omega, omega_ref, time_step=time_step)

    def reset(self) -> None:
        self.base.reset()
        self.sm_obs.reset()
        self.mode_switch.reset()
        self._prev_omega = None

    @property
    def mode(self) -> str:
        return self.mode_switch.mode
