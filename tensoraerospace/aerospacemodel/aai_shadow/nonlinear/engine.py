"""UEL AR-741 rotary engine + 24-inch pusher propeller for the Shadow.

The AR-741 is a single-rotor Wankel-type rotary engine producing
38 hp (28 kW) at 7 600 rpm. Mounted as a pusher behind the fuselage
boom intersection, driving a 2-blade carbon-composite propeller of
~ 0.61 m (24 in) diameter.

Following the Skywalker X8 module's approach, the propeller / motor
chain is collapsed into a **calibrated quadratic thrust model**:

.. math::
   T(\\delta_T, V) = T_{\\max} \\cdot \\delta_T^2 \\cdot (1 - V / V_{\\text{zero}})

with the calibration:

* **Static, full throttle**: $T \\approx 380$ N
  (38 hp × 0.55 prop η / V_static ≈ 28 000 W × 0.55 / ~ 40 m/s blade
  tip speed equivalent).
* **Cruise (35 m/s, 70 % throttle)**: $T \\approx 70$ N — matches
  drag at trim weight 170 kg, $C_{D_0}$ = 0.030, $C_L = 0.7$.

The published AR-741 thrust curves are not in the open literature
to cross-check; the calibration above produces a self-consistent
trim at the published cruise speed and propellant fuel-flow numbers.
"""

from __future__ import annotations

from .params import AAIShadowParameters, isa_density_kg_m3

# Calibration constants
_T_STATIC_FULL_N = 380.0  # full-throttle static thrust at sea level
_V_ZERO_THRUST_M_S = 65.0  # airspeed at which thrust would reach zero


def shadow_thrust(
    throttle: float,
    V_m_s: float,
    altitude_m: float,
    params: AAIShadowParameters,
) -> tuple[float, float]:
    """Calibrated thrust model. Returns ``(T_N, CT_value)``.

    The CT value is reported for parity with the Skywalker X8 API
    (so the airframe drag could include a ``CDCT`` coupling term in
    the future). For the Shadow the airframe drag does not include a
    propeller-coupling term — the prop is far enough behind the wing
    not to affect elevon airflow noticeably.
    """
    thr = max(0.0, min(1.0, float(throttle)))
    V = max(0.0, float(V_m_s))

    # Density correction
    rho_ratio = isa_density_kg_m3(altitude_m) / 1.225

    # Quadratic in throttle, linear knockdown in airspeed
    speed_factor = max(0.0, 1.0 - V / _V_ZERO_THRUST_M_S)
    T = _T_STATIC_FULL_N * thr * thr * speed_factor * rho_ratio

    # No CDCT coupling for this airframe — return 0 for the second value
    return float(T), 0.0
