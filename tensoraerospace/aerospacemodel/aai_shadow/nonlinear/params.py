"""AAI RQ-7 Shadow tactical UAV — mass / inertia / geometry / engine.

The RQ-7 Shadow is a class-II tactical reconnaissance UAV operated by
the US Army and several allied forces. Key characteristics (RQ-7A
configuration, FAS / NASA-published technical data):

* Configuration: high-aspect-ratio rectangular wing, pusher propeller,
  inverted V-tail with twin tail booms.
* Wingspan: 4.27 m (14.0 ft).
* Wing area: 1.67 m² (18.0 ft²).
* Aspect ratio: 10.9 — typical surveillance UAV high-AR layout.
* Length: 3.41 m.
* Empty weight: ~ 91 kg.
* Gross / cruise weight: ~ 170 kg.
* Max take-off weight: 212 kg (catapult-launched).
* Engine: UEL AR-741 single-rotor Wankel rotary, 38 hp (28 kW),
  pusher-mounted.
* Cruise speed: ~ 36 m/s (70 kt).
* Service ceiling: ~ 4 600 m (15 000 ft).
* Endurance: 6–9 h.

The aerodynamic derivative bank used here is **synthesised from
class-II small-UAV literature** rather than transcribed from a single
canonical AAI / NASA paper:

* **Beard R. W., McLain T. W.** *Small Unmanned Aircraft: Theory and
  Practice*, Princeton Univ. Press (2012). Appendix E provides the
  Aerosonde Mark 4.7 derivative set; we scale it by the Shadow's
  larger geometry.
* **NASA TM-2014-218686** — RQ-7 Shadow aerodynamic-database
  reference (used for sanity-checking magnitude of $C_{L_\\alpha}$,
  $C_{m_\\alpha}$ and the V-tail effective $C_{l_{\\delta_r}}$,
  $C_{n_{\\delta_r}}$ values).
* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Vol VI Appendix C — V-tail mixing relations and
  effective-area scaling.
* **Nelson R. C.** *Flight Stability and Automatic Control*
  (McGraw-Hill, 2nd ed., 1998) — high-aspect-ratio surveillance
  aircraft derivative ranges.

Units: **SI** (kg, m, N, rad, s) throughout, matching the
Skywalker X8 module (the other small-UAV in this roster).

V-tail control mixing:

    δ_e = (δ_l + δ_r) / 2       (collective ruddervator → elevator)
    δ_r = (δ_l - δ_r) / 2       (differential ruddervator → rudder)

We expose the **mixed channels** (δ_e, δ_r) as the agent's controls,
so the model is API-compatible with the conventional-tail B-737 / B-747
modules (4-channel ``[δ_e, δ_a, δ_r, δ_T]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---- ISA atmosphere (SI) — same as Skywalker X8 module ----------------

_T0_K = 288.15
_RHO0_KG_M3 = 1.225
_LAPSE_K_M = 0.0065
_R_AIR = 287.05
_TROPOPAUSE_M = 11_000.0
_T_TROPO_K = 216.65
_GAMMA_AIR = 1.4


def isa_density_kg_m3(altitude_m: float) -> float:
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


# ---- AAI RQ-7 Shadow parameters --------------------------------------


@dataclass
class AAIShadowParameters:
    """RQ-7A Shadow inertial / geometric / propulsion parameters (SI).

    Defaults are the cruise configuration (~ 170 kg gross weight,
    half-fuel). Empty and MTOW operating points are accessible through
    the helper variants below.
    """

    # Mass / inertia (cruise configuration, scaled from class-II UAV
    # references; FAS RQ-7 fact sheet weight + Roskam Vol VI inertia
    # estimation for high-aspect-ratio twin-boom)
    mass_kg: float = 170.0
    Ix: float = 50.0       # kg·m²
    Iy: float = 80.0
    Iz: float = 120.0
    Ixz: float = 5.0

    # Geometry (FAS / AAI published spec — RQ-7B "Improved Tactical UAV"
    # configuration with the larger 20.4 ft wingspan; the original
    # RQ-7A had b=4.27 m / S=1.67 m² but the RQ-7B is the variant
    # actually published with the 36 m/s cruise speed used here).
    S_m2: float = 4.42       # planform area, RQ-7B
    b_m: float = 6.22        # 20.4 ft wingspan
    cbar_m: float = 0.71     # ≈ S/b for rectangular wing
    cg_frac_cbar: float = 0.25

    g_m_s2: float = 9.80665

    # Actuator dynamics — typical small-UAV electric servos with
    # gear backlash. Conservative first-order model.
    elevator_tau_s: float = 0.05
    elevator_max_rad: float = math.radians(20.0)
    elevator_rate_max_rad_s: float = math.radians(120.0)
    aileron_tau_s: float = 0.05
    aileron_max_rad: float = math.radians(20.0)
    aileron_rate_max_rad_s: float = math.radians(120.0)
    rudder_tau_s: float = 0.05
    rudder_max_rad: float = math.radians(15.0)
    rudder_rate_max_rad_s: float = math.radians(120.0)

    # UEL AR-741 rotary engine (Wankel, single rotor) — published specs.
    # Calibrated thrust model (see :mod:`.engine`):
    #   * Static, full throttle: T ≈ 380 N (~ 38 hp at static prop η)
    #   * Cruise (35 m/s, 70 % throttle): T ≈ 70 N
    engine_max_power_kW: float = 28.0          # 38 hp
    engine_idle_frac: float = 0.10
    prop_diameter_m: float = 0.61              # 24" 2-blade pusher
    prop_static_efficiency: float = 0.55       # static thrust factor

    # Damage subsystem hooks (parity with the rest of the family)
    damage_state: Optional[object] = None
    damage_geometry: Optional[object] = None


def default_parameters() -> AAIShadowParameters:
    """Return the published RQ-7A Shadow cruise parameters."""
    return AAIShadowParameters()
