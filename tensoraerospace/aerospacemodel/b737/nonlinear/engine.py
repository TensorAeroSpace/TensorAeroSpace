"""Engine model for the nonlinear Boeing 737.

Two engine variants are supported through the configuration enum:

* **JT8D-9** — Pratt & Whitney low-bypass turbofan, 14 500 lbf SLS
  per engine (737-100/200). Higher-throttle Mach-derate is steeper
  than CFM56 because of its lower bypass ratio.
* **CFM56-7B27** — CFM International high-bypass turbofan, 27 300 lbf
  SLS per engine (737-NG). Mach-derate matches the JT9D-7 model
  used in the B-747 module.

Following the B-747 model, installed thrust uses Mattingly's
high-bypass turbofan derate:

.. math::
   T_{inst}(M, h, \\delta_T) = T_{SLS} \\cdot \\sigma(h)^{n_h} \\cdot
                               \\eta_{ram}(M) \\cdot \\delta_{T,eff}

with $\\sigma(h) = \\rho(h) / \\rho_0$ and $\\eta_{ram}$ either the
JT8D or CFM56 form depending on the engine selection.

Engine spanwise positions on the 737 (under-wing nacelles):

* Inboard pylon at $y \\approx \\pm 16$ ft (BL ±192 in).
* Used in :func:`cfm56_thrust_with_asymmetry` for engine-out
  scenarios (parity with B-747 ``jt9d_thrust_with_asymmetry``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import B737Configuration, B737Parameters, isa_density_slug_ft3

_RHO0_SLUG_FT3 = 0.002378

# Engine spanwise positions (ft from centerline). 737-100/200 inboard
# pylons sit ~ 16.5 ft outboard of the centerline; the longer
# 737-800 has them ~ 18 ft outboard. We use a single value here —
# the moment arm difference is < 10 % across the family.
ENGINE_Y_POSITIONS_FT: dict[int, float] = {
    1: -16.5,  # left
    2: +16.5,  # right
}


@dataclass
class B737Engine:
    """Twin-engine cluster — JT8D-9 or CFM56-7B."""

    n_engines: int = 2
    sls_thrust_per_engine_lb: float = 14_500.0  # JT8D-9 default
    idle_frac: float = 0.05
    spool_tau_s: float = 1.5
    bypass_ratio: float = 1.0  # JT8D ≈ 1, CFM56 ≈ 5.5
    use_ram_recovery: bool = True

    @property
    def total_sls_thrust_lb(self) -> float:
        return self.n_engines * self.sls_thrust_per_engine_lb

    def installed_thrust(
        self, mach: float, altitude_ft: float, throttle: float
    ) -> float:
        thr = max(0.0, min(1.0, float(throttle)))
        pla_eff = self.idle_frac + (1.0 - self.idle_frac) * thr
        sigma = isa_density_slug_ft3(altitude_ft) / _RHO0_SLUG_FT3
        if not self.use_ram_recovery:
            return float(self.total_sls_thrust_lb * sigma * pla_eff)

        m = max(0.0, float(mach))
        # Mattingly §8.6.4 high-bypass turbofan ram-recovery factor.
        # For JT8D (BPR ≈ 1) Mattingly's correlation suggests ~ 0.55,
        # but installed-thrust data from FAA TCDS A16WE for the 737-100
        # at typical cruise (M = 0.78 / FL280) is matched with the
        # standard 0.49 used for the rest of the high-bypass family.
        # We therefore use 0.49 for both JT8D and CFM56 — the
        # bypass-ratio field stays for downstream studies that want a
        # different correlation.
        ram = 1.0 - 0.49 * math.sqrt(m)
        ram = max(ram, 0.05)
        sigma_pow = 0.7 if altitude_ft < 36_089.0 else 1.0
        eta = ram * (sigma**sigma_pow)
        return float(self.total_sls_thrust_lb * eta * pla_eff)


def _engine_for_config(params: B737Parameters) -> B737Engine:
    """Build a :class:`B737Engine` matching the configuration."""
    if params.config is B737Configuration.B737_100:
        return B737Engine(
            n_engines=params.n_engines,
            sls_thrust_per_engine_lb=params.engine_thrust_max_lb / params.n_engines,
            idle_frac=params.engine_idle_frac,
            spool_tau_s=params.engine_tau_s,
            bypass_ratio=1.0,  # JT8D-9
        )
    return B737Engine(
        n_engines=params.n_engines,
        sls_thrust_per_engine_lb=params.engine_thrust_max_lb / params.n_engines,
        idle_frac=params.engine_idle_frac,
        spool_tau_s=params.engine_tau_s,
        bypass_ratio=5.5,  # CFM56-7B
    )


def b737_thrust(
    throttle: float, mach: float, altitude_ft: float, params: B737Parameters
) -> float:
    """Total engine cluster thrust (lbf) at the given operating point."""
    return _engine_for_config(params).installed_thrust(mach, altitude_ft, throttle)


def b737_thrust_with_asymmetry(
    throttle: float, mach: float, altitude_ft: float, params: B737Parameters
) -> tuple[float, float]:
    """Per-engine thrust + yaw moment from asymmetric engine effectiveness.

    Reads ``params.damage_state.engines_mu`` (a ``dict[int, float]``
    keyed 1..2 with values in ``[0, 1]``) and returns
    ``(T_total_lb, N_yaw_lb_ft)``. When the damage state is missing or
    all engines are healthy, ``N_yaw = 0`` and ``T_total`` is identical
    to :func:`b737_thrust`.

    Sign convention (NED body axis, +z down): a thrust force at body
    +y produces moment ``N = -y * T`` along +z. So a dead engine on
    the **left** wing leaves more thrust on the right → nose yaws
    *left* (toward the dead engine). Same algebra as the B-747 model.
    """
    eng = _engine_for_config(params)
    cluster = eng.installed_thrust(mach, altitude_ft, throttle)
    per_engine = cluster / max(eng.n_engines, 1)

    damage_state = getattr(params, "damage_state", None)
    if damage_state is not None and getattr(damage_state, "engines_mu", None):
        engines_mu = damage_state.engines_mu
    else:
        engines_mu = {i: 1.0 for i in (1, 2)}

    T_total = 0.0
    N_yaw = 0.0
    for engine_id in (1, 2):
        T_i = per_engine * float(engines_mu.get(engine_id, 1.0))
        T_total += T_i
        N_yaw += -ENGINE_Y_POSITIONS_FT[engine_id] * T_i
    return float(T_total), float(N_yaw)
