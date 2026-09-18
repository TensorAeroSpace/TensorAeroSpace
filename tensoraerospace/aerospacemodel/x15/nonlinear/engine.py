"""XLR99 rocket engine model for the X-15.

The model uses constant maximum thrust (57,000 lbf) and specific impulse
(254 s), with throttle from 0.30 to 1.0. Below the cutoff or with no
remaining propellant it returns zero thrust and zero mass flow.

The propellant state is in pounds, so its positive depletion rate is
``T_lbf / Isp_s`` in lb/s. The equivalent mass rate in slug/s is
``T_lbf / (Isp_s * g0_ft_s2)``. At full throttle the configured BASIC
load of 17,900 lb lasts about 79.8 s; the A2 load of 30,900 lb lasts
about 137.7 s.

No nozzle back-pressure correction is modeled. Constant thrust and Isp
are approximations, not an altitude-calibrated engine performance map.
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
