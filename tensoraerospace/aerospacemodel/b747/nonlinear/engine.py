"""Simplified JT9D-7 cluster thrust for the nonlinear B747 model.

Installed thrust is ``T_SLS * density_lapse * ram(M) * PLA_eff``.
The density lapse is sigma**0.7 below 36089 ft. Above that altitude,
it follows density relative to the boundary, anchored to the value
of the lower branch. This avoids a spurious thrust step at the tropopause.

The Mach correction and sea-level rating are retained from the original
model. Continuity is an internal consistency condition; this curve is not
a validation against a measured JT9D engine deck. Engine spool dynamics
remain unimplemented.
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
    1: -71.7,  # left outer
    2: -35.8,  # left inner
    3: +35.8,  # right inner
    4: +71.7,  # right outer
}


@dataclass
class JT9DEngine:
    """4-engine JT9D-7 cluster, installed thrust model."""

    n_engines: int = 4
    sls_thrust_per_engine_lb: float = 47_100.0
    idle_frac: float = 0.05
    spool_tau_s: float = 1.0  # for future first-order thrust dynamics
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
        # Existing empirical Mach correction
        ram = 1.0 - 0.49 * math.sqrt(m)
        ram = max(ram, 0.05)
        # Match the two density exponents at the layer boundary
        if altitude_ft < 36_089.0:
            density_lapse = sigma**0.7
        else:
            # Match the left branch at the tropopause. Merely switching
            # sigma**0.7 to sigma creates an artificial ~30% thrust drop.
            # Separate one-sided ISA references account for rounded constants.
            rho_left = isa_density_slug_ft3(math.nextafter(36_089.0, -math.inf))
            rho_right = isa_density_slug_ft3(36_089.0)
            density_lapse = (rho_left / _RHO0_SLUG_FT3) ** 0.7 * (
                sigma * _RHO0_SLUG_FT3 / rho_right
            )
        eta = ram * density_lapse
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
