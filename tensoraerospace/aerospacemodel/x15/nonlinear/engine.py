"""XLR99 rocket engine model for the X-15.

The Reaction Motors XLR99 is the production rocket engine of the
X-15. Key facts (Thompson 2000 / NASA SP-2000-4222):

* Single-chamber, throttleable from 30 % to 100 % thrust.
* Propellant: anhydrous ammonia (fuel) + LOX (oxidizer).
* Sea-level rated thrust ``T_SLS = 57 000 lbf``.
* Specific impulse ``Isp ≈ 254 s`` (sea level), rising slightly to
  ~ 290 s in vacuum due to nozzle expansion. We use the constant
  ``Isp = 254 s`` value for the powered envelope; the X-15 never
  reached pressures low enough for the vacuum correction to dominate.
* Burn time at full throttle: ~ 80 s for the basic X-15 (13 000 lb
  propellant), ~ 140 s for X-15A-2 (18 000 lb).

Unlike an air-breathing engine, **the rocket thrust is essentially
independent of Mach and altitude** — there is no inlet recovery, no
ram effect. The only altitude effect is the small back-pressure
correction $T(h) = T_{SLS} + p_{0} A_e - p(h) A_e$, which we ignore
in the initial release (it is ≤ 5 % below 100 kft).

Mass flow at any throttle setting:

.. math::
   \\dot m = -\\frac{T(\\delta_T)}{I_{sp}\\, g_0}, \\qquad
   T(\\delta_T) = T_{SLS} \\cdot \\delta_T

where ``δ_T ∈ [0.30, 1.0]`` (the XLR99 cannot run below 30 %
throttle). Below 30 % the engine is treated as off (zero thrust,
zero mass flow), matching the real lockout behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import X15Parameters


@dataclass
class XLR99Engine:
    """Reaction Motors XLR99 model — constant thrust / mass flow with throttle."""

    sls_thrust_lb: float = 57_000.0
    isp_s: float = 254.0
    throttle_min: float = 0.30
    throttle_max: float = 1.0
    g0_ft_s2: float = 32.174

    def thrust_and_mdot(
        self, throttle: float, propellant_lb: float
    ) -> tuple[float, float]:
        """Return ``(thrust_lbf, mass_flow_lb_per_s)`` at the requested throttle.

        Below ``throttle_min`` the engine is *off*: returns ``(0, 0)``.
        With ``propellant_lb <= 0`` the engine has flamed out (no fuel),
        also returns ``(0, 0)``.

        Mass flow is reported as **positive lb/s** (i.e. the magnitude
        of the propellant decrement); the dynamics integrator subtracts
        it from the propellant state.
        """
        if propellant_lb <= 0.0:
            return 0.0, 0.0
        thr = float(throttle)
        if thr < self.throttle_min:
            return 0.0, 0.0
        thr = min(thr, self.throttle_max)
        T_lb = self.sls_thrust_lb * thr
        mdot_lb_s = T_lb / (self.isp_s)  # weight-flow form: mdot_lb = T / Isp
        return T_lb, mdot_lb_s


def xlr99_thrust(
    throttle: float, propellant_lb: float, params: X15Parameters
) -> tuple[float, float]:
    """Convenience entry point used by :func:`x15_ode_6dof`.

    Delegates to :class:`XLR99Engine`, sourcing the engine constants
    from ``params`` (which is how damage-time perturbations will plug
    in later — e.g. an engine flameout event simply sets ``thrust =
    0`` regardless of throttle).
    """
    eng = XLR99Engine(
        sls_thrust_lb=params.engine_thrust_max_lb,
        isp_s=params.engine_isp_s,
        throttle_min=params.engine_throttle_min,
    )
    return eng.thrust_and_mdot(throttle, propellant_lb)
