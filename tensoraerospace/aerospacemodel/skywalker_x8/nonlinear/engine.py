"""Skywalker X8 propulsion: Hacker A40 motor + 14"x8 Aeronaut propeller.

The full model in Løw-Hansen et al. CEAS 2025 §2.3 couples the motor
electrical state (current, voltage) with the propeller cubic CT(J)
polynomial via the steady-state torque-balance equation. Solving that
fixed point at every ODE evaluation is feasible but heavy.

For the MVP we use a **calibrated quadratic thrust model**:

.. math::
   T(\\delta_T, V) = T_{\\max} \\cdot \\delta_T^2 \\cdot (1 - V / V_{\\text{zero}})

This is the canonical "two-point fit" used in most academic UAV
controllers (Carrillo et al., Beard & McLain Sec. 2.4). The two
calibration points come from the paper:

* **Static, full throttle**: ``T = T_max ≈ 40 N`` (Hacker A40 + 14"
  prop typical published value, cross-checked against motor specs).
* **Cruise, 44 % throttle, 18 m/s** (paper Eq. 38): ``T ≈ 3.5 N``.

The published cubic CT(J) polynomial — which is exact at the trim
point but extrapolates poorly outside the experimentally covered
J range — is exposed via :class:`X8Propeller` for downstream
high-fidelity studies that want to plug in the full motor electrical
model.

The thrust coefficient ``CT`` returned alongside the thrust feeds back
into the airframe-drag model (CDCT coupling, paper Sec. 4.1.3). Since
we use the simplified thrust model, we report a CT computed from the
ratio ``T / (ρ D⁴ n²)`` at the equivalent steady-state J, with
n = throttle · Ωp_max_loaded / (2π).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import SkywalkerX8Parameters, isa_density_kg_m3

# Calibrated thrust model constants — match Løw-Hansen 2025 trim point.
_T_STATIC_FULL_N = 40.0  # static thrust at full throttle (sea level)
_V_ZERO_THRUST_M_S = 35.0  # airspeed at which thrust would reach zero


@dataclass
class X8Propeller:
    """14"x8 Aeronaut CAM folding propeller — paper Tables 6, 7.

    Provided for high-fidelity studies that want to couple the
    motor electrical model with the published cubic CT(J).
    """

    diameter_m: float = 0.3556
    CT0: float = 0.1400
    CT1: float = -0.0300
    CT2: float = -0.2370
    CT3: float = 0.0847
    CQ0: float = 0.0082
    CQ1: float = 0.0112
    CQ2: float = -0.0211

    def CT(self, J: float) -> float:
        """Thrust coefficient as a cubic of advance ratio J."""
        return self.CT0 + self.CT1 * J + self.CT2 * J * J + self.CT3 * J * J * J

    def CQ(self, J: float) -> float:
        """Torque coefficient as a quadratic of advance ratio J."""
        return self.CQ0 + self.CQ1 * J + self.CQ2 * J * J


def x8_thrust(
    throttle: float,
    V_m_s: float,
    altitude_m: float,
    params: SkywalkerX8Parameters,
) -> tuple[float, float]:
    """Calibrated thrust model. Returns ``(T_N, CT_value)``.

    The CT value feeds back into the aero drag (CDCT coupling).
    """
    thr = max(0.0, min(1.0, float(throttle)))
    V = max(0.0, float(V_m_s))

    # Density correction relative to sea level
    rho_ratio = isa_density_kg_m3(altitude_m) / 1.225

    # Quadratic in throttle, linear knockdown in airspeed
    speed_factor = max(0.0, 1.0 - V / _V_ZERO_THRUST_M_S)
    T = _T_STATIC_FULL_N * thr * thr * speed_factor * rho_ratio

    # Reverse-engineer an equivalent CT for the airframe-drag coupling.
    # Use the loaded propeller speed Ωp = throttle · Ω_max with Ω_max
    # tuned so that cruise CT matches the published cubic at the trim
    # advance ratio J ≈ 0.7. This is purely a normalisation for the
    # CDCT term — the airframe drag responds to thrust quasi-linearly,
    # so any consistent positive CT value works for the controller.
    omega_p = max(thr, 0.05) * params.omega_max_rad_s
    n = omega_p / (2.0 * math.pi)
    rho = isa_density_kg_m3(altitude_m)
    denom = rho * (params.prop_diameter_m**4) * (n * n)
    CT_val = T / denom if denom > 1e-9 else 0.0
    return float(T), float(max(CT_val, 0.0))
