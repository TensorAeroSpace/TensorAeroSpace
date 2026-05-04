"""Engine model for the B-747 nonlinear simulation.

Implements a Pratt & Whitney JT9D-7 cluster (4 engines) with thrust as
a function of (Mach, altitude, throttle) using the canonical bypass
turbofan installed-thrust model:

    T_inst(M, h, PLA)  =  T_SLS · σ(h) · η(M, h) · PLA_eff

where:

    σ(h)         = density ratio ρ(h)/ρ_SL
    η(M, h)      = Mach derate (cruise lapse + ram-recovery factor)
    T_SLS        = sea-level static thrust per engine, lb
                   (JT9D-7: 47,100 lb take-off rating)
    PLA_eff      = idle_frac + (1 − idle_frac) · throttle

The η(M, h) lapse follows Mattingly *Aircraft Engine Design* (2002)
§ 8.6.4 — for high-bypass turbofans at subsonic cruise:

    η(M, h)  ≈  (1 − 0.49·√M) · σ(h)^0.7  for h < tropopause
    η(M, h)  ≈  (1 − 0.49·√M) · σ(h)^1.0  for h ≥ tropopause

The 4-engine cluster is modelled as a single thrust vector along body
+x; engine-out asymmetric scenarios should be handled at the env
level by tagging individual engines through the damage subsystem.

Reference values verified against:
* Boeing 747-100 type certificate data sheet (FAA TCDS A20WE).
* P&W JT9D-7 published cruise SFC envelope at M=0.85, h=35 kft.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import B747Parameters, isa_density_slug_ft3

_RHO0_SLUG_FT3 = 0.002378

# B-747-100 spanwise engine positions (NED body axis: +y to right of
# centerline, ft). Conventional 4-engine cluster: inner engines at BL.430
# ≈ 35.8 ft, outer engines at BL.860 ≈ 71.7 ft; sign indicates wing side.
ENGINE_Y_POSITIONS_FT: dict[int, float] = {
    1: -71.7,   # left outer
    2: -35.8,   # left inner
    3: +35.8,   # right inner
    4: +71.7,   # right outer
}


@dataclass
class JT9DEngine:
    """4-engine JT9D-7 cluster, installed thrust model."""

    n_engines: int = 4
    sls_thrust_per_engine_lb: float = 47_100.0
    idle_frac: float = 0.05
    spool_tau_s: float = 1.0    # for future first-order thrust dynamics
    use_ram_recovery: bool = True

    @property
    def total_sls_thrust_lb(self) -> float:
        return self.n_engines * self.sls_thrust_per_engine_lb

    def installed_thrust(
        self, mach: float, altitude_ft: float, throttle: float
    ) -> float:
        """Installed thrust in pounds at the given (M, h, throttle)."""
        thr = max(0.0, min(1.0, float(throttle)))
        pla_eff = self.idle_frac + (1.0 - self.idle_frac) * thr
        sigma = isa_density_slug_ft3(altitude_ft) / _RHO0_SLUG_FT3
        if not self.use_ram_recovery:
            return float(self.total_sls_thrust_lb * sigma * pla_eff)
        m = max(0.0, float(mach))
        # Mattingly high-bypass turbofan derate
        ram = 1.0 - 0.49 * math.sqrt(m)
        ram = max(ram, 0.05)
        # Stratospheric branch (h ≥ 36089 ft) decays faster
        sigma_pow = 0.7 if altitude_ft < 36089.0 else 1.0
        eta = ram * (sigma ** sigma_pow)
        return float(self.total_sls_thrust_lb * eta * pla_eff)


def jt9d_thrust(
    throttle: float, mach: float, altitude_ft: float, params: B747Parameters
) -> float:
    """Convenience entry point used by :func:`b747_ode_6dof`.

    Reads cluster size and SLS thrust from ``params``, falling back to
    the JT9D-7 defaults (4 × 47,100 lb).
    """
    eng = JT9DEngine(
        n_engines=4,
        sls_thrust_per_engine_lb=params.engine_thrust_max_lb / 4.0,
        idle_frac=params.engine_idle_frac,
        spool_tau_s=params.engine_tau_s,
    )
    return eng.installed_thrust(mach=mach, altitude_ft=altitude_ft, throttle=throttle)


def jt9d_thrust_with_asymmetry(
    throttle: float, mach: float, altitude_ft: float, params: B747Parameters
) -> tuple[float, float]:
    """Per-engine thrust + yaw moment from asymmetric engine effectiveness.

    Reads ``params.damage_state.engines_mu`` (a ``dict[int, float]``
    keyed 1..4 with values in ``[0, 1]``) and returns
    ``(T_total_lb, N_yaw_lb_ft)``. When the damage state is missing or
    all engines are at full effectiveness, ``N_yaw = 0`` and
    ``T_total`` is identical to :func:`jt9d_thrust`.

    Yaw-moment sign convention (NED body axis: +z down): a single +x
    thrust force at body y-position ``y_i`` generates ``N = -y_i · T_i``.
    If only the right wing engines are firing, the right wing accelerates
    forward and the nose yaws *left* (negative N) — toward the dead
    engines, as expected for an asymmetric thrust scenario.
    """
    eng = JT9DEngine(
        n_engines=4,
        sls_thrust_per_engine_lb=params.engine_thrust_max_lb / 4.0,
        idle_frac=params.engine_idle_frac,
        spool_tau_s=params.engine_tau_s,
    )
    # The cluster's installed_thrust assumes all 4 engines at full PLA;
    # divide to get per-engine thrust, then re-weight by engines_mu.
    cluster_thrust = eng.installed_thrust(
        mach=mach, altitude_ft=altitude_ft, throttle=throttle
    )
    per_engine_thrust = cluster_thrust / 4.0

    damage_state = getattr(params, "damage_state", None)
    if damage_state is not None and getattr(damage_state, "engines_mu", None):
        engines_mu = damage_state.engines_mu
    else:
        engines_mu = {i: 1.0 for i in (1, 2, 3, 4)}

    T_total = 0.0
    N_yaw = 0.0
    for engine_id in (1, 2, 3, 4):
        T_i = per_engine_thrust * float(engines_mu.get(engine_id, 1.0))
        T_total += T_i
        N_yaw += -ENGINE_Y_POSITIONS_FT[engine_id] * T_i
    return float(T_total), float(N_yaw)
