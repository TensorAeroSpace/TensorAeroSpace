"""Skywalker X8 small fixed-wing UAV — mass / inertia / geometry / engine.

This is the **peer-reviewed open-data UAV** in the tensoraerospace
roster. All numerical values transcribed from:

* **Løw-Hansen B., Hann R., Gryte K., Johansen T. A., Deiler C.**
  *"Modeling and identification of a small fixed-wing UAV using
  estimated aerodynamic angles"*, **CEAS Aeronautical Journal**,
  Springer (2025).
  https://link.springer.com/article/10.1007/s13272-025-00816-3

The paper reports a flight-test-identified 6-DoF nonlinear
aerodynamic model of the Skywalker X8 — a 2.1 m-span flying-wing
UAV with two elevons and a rear-mounted pusher propeller, used by
NTNU and DLR for icing-research and FTC studies.

The Skywalker X8 is the canonical "small fixed-wing UAV"
representative in the tensoraerospace roster — small (~ 2 m span),
electric, hobby-grade construction, the typical research platform.
The model parameters and identified coefficients are taken verbatim
from the open-access paper for reproducibility.

Units: **SI** (kg, m, N, rad, s) throughout, matching the source paper.
The other tensoraerospace nonlinear models (B-747, B-737, X-15) use
FPS — the X8 is the exception because its source publishes SI.

Mass / geometry (paper Table 1):

    m   = 3.364 kg
    Ix  = 0.325 kg·m²
    Iy  = 0.140 kg·m²
    Iz  = 0.400 kg·m²
    Ixz = 0.029 kg·m²
    c̄  = 0.36 m       (mean aerodynamic chord)
    b   = 2.10 m       (wingspan)
    S   = 0.75 m²      (planform reference area)

Propulsion (paper Tables 6, 7):

    Propeller:    14" Aeronaut CAM folding, D = 0.3556 m, Ip = 3.46e-4 kg·m²
    Motor:        Hacker A40-12 S V2 14-pin KV610, KE = 0.0157 V/(rad/s),
                  R = 0.017 Ω
    ESC:          Jeti SPIN Pro 66

Trim point used for system identification (paper Eq. 38):

    Va = 17.9 m/s,  α = 7.9°,  β = 1.2°,  γ = 0°
    δe = -2.35°,  δa = -2.16°,  δt = 0.44

Control surface notation. The X8 is a flying wing controlled by two
elevons (left and right). The published model uses the standard
mixing:

    δe = (δer + δel) / 2     (collective elevator pitch input)
    δa = (δer - δel) / 2     (differential aileron roll input)

There is **no rudder**; lateral-directional control is via differential
elevon only. The model treats the two control channels (δe, δa) and the
throttle δt as the agent inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---- ISA atmosphere (SI units) ------------------------------------------

_T0_K = 288.15
_RHO0_KG_M3 = 1.225
_LAPSE_K_M = 0.0065
_R_AIR = 287.05      # specific gas constant for air, J/(kg·K)
_TROPOPAUSE_M = 11_000.0
_T_TROPO_K = 216.65
_GAMMA_AIR = 1.4
_FT_TO_M = 0.3048


def isa_density_kg_m3(altitude_m: float) -> float:
    """ISA density in kg/m³ at the given altitude (m)."""
    if altitude_m < _TROPOPAUSE_M:
        T = _T0_K - _LAPSE_K_M * altitude_m
        return float(_RHO0_KG_M3 * (T / _T0_K) ** 4.2561)
    rho_tropo = _RHO0_KG_M3 * (_T_TROPO_K / _T0_K) ** 4.2561
    return float(rho_tropo * math.exp(-(altitude_m - _TROPOPAUSE_M) / 6341.6))


def isa_speed_of_sound_m_s(altitude_m: float) -> float:
    if altitude_m < _TROPOPAUSE_M:
        T = _T0_K - _LAPSE_K_M * altitude_m
    else:
        T = _T_TROPO_K
    return float(math.sqrt(_GAMMA_AIR * _R_AIR * T))


# ---- Skywalker X8 parameters --------------------------------------------


@dataclass
class SkywalkerX8Parameters:
    """Skywalker X8 inertial / geometric / propulsion parameters (SI).

    Defaults match the published values in
    Løw-Hansen et al. CEAS Aeronautical Journal (2025), Tables 1, 6, 7.
    """

    # Mass / inertia (paper Table 1)
    mass_kg: float = 3.364
    Ix: float = 0.325            # kg·m²
    Iy: float = 0.140
    Iz: float = 0.400
    Ixz: float = 0.029

    # Geometry (paper Table 1)
    S_m2: float = 0.75
    b_m: float = 2.10
    cbar_m: float = 0.36
    cg_frac_cbar: float = 0.25

    g_m_s2: float = 9.80665

    # Actuator dynamics — second-order elevon model from paper Sec. 4.1.4.
    # We collapse the second-order to a first-order lag for the simple
    # MVP integration; full second-order can be plugged in via the env
    # wrapper if needed.
    elevon_tau_s: float = 0.05
    elevon_max_rad: float = math.radians(20.0)
    elevon_rate_max_rad_s: float = math.radians(120.0)
    throttle_tau_s: float = 0.20

    # Propeller (paper Tables 6, 7)
    prop_diameter_m: float = 0.3556        # 14" Aeronaut CAM
    prop_inertia_kg_m2: float = 3.46e-4

    # CT(J) cubic polynomial — Table 6
    CT0: float = 0.1400
    CT1: float = -0.0300
    CT2: float = -0.2370
    CT3: float = 0.0847
    # CQ(J) quadratic — Table 6
    CQ0: float = 0.0082
    CQ1: float = 0.0112
    CQ2: float = -0.0211

    # Motor — Hacker A40-12 (paper Table 7)
    motor_KE_V_per_rad_s: float = 0.0157
    motor_resistance_ohm: float = 0.017
    motor_zero_load_current_A: float = 0.5    # paper-typical Im0
    battery_voltage_V: float = 16.0           # 4S nominal, paper Fig. 7

    # No-load motor speed at full throttle — used as the normalising
    # factor for the airframe-drag CDCT coupling in the simplified
    # thrust model (see :mod:`.engine`).
    @property
    def omega_max_rad_s(self) -> float:
        return self.battery_voltage_V / self.motor_KE_V_per_rad_s

    # Damage subsystem hooks
    damage_state: Optional[object] = None
    damage_geometry: Optional[object] = None


def default_parameters() -> SkywalkerX8Parameters:
    """Return the published Skywalker X8 parameters."""
    return SkywalkerX8Parameters()
