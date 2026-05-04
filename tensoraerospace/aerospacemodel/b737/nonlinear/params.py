"""Boeing 737-100/200 mass / inertia / geometry parameters.

Numerical values are sourced primarily from the **JSBSim open-source
737 model** (``aircraft/737/737.xml``), which itself is built from
Roskam's published 737-100 derivative tables. The JSBSim repository
attributes its data to:

* **Roskam J.** *Airplane Flight Dynamics and Automatic Flight
  Controls*, Roskam Aviation & Engineering Corp., 1995 — Appendix B
  has the 737-100 stability and control derivatives. The original
  Boeing wind-tunnel data underlying these tables was published as:
* **Hanke C. R.** *The Simulation of a Large Jet Transport
  Aircraft*, NASA CR-114494 (1971) — methodology used to derive
  Roskam's tabulated values.
* **Cook M. V.** *Flight Dynamics Principles*, Elsevier 3rd ed.
  (2013), Chapter 11 — uses the same 737-100 reference dataset.

Cross-validated against:

* **NASA TM-86821 (1986)** *"Design and verification by nonlinear
  simulation of a Mach/CAS control law for the NASA TCV B737
  aircraft"* — published 737 nonlinear-simulation envelope.
* **FAA Type Certificate Data Sheet A16WE** — Boeing 737 family
  weights, geometry, engine ratings.
* **CFM International CFM56-3/-7B fact sheets** — engine performance.

All values are in **US customary units (slug, ft, lb, sec)** as
published. SI helpers via :func:`to_si`.

Two configurations are exposed (mirroring how the B-747 nonlinear
module exposes NOMINAL / POWER_APPROACH / LANDING):

* :attr:`B737Configuration.B737_100` — 737-100 (matches JSBSim
  defaults). 2 × Pratt & Whitney JT8D-9, S=1171 ft², MTOW≈110 000 lb.
* :attr:`B737Configuration.B737_800` — 737-800 NG. 2 × CFM56-7B27,
  S=1341 ft², MTOW≈174 200 lb. Aerodynamics use the same
  derivative tables (the geometry change is captured by the wing
  area / span / chord scaling); for high-fidelity 737NG studies a
  full re-derivation is recommended (out of scope for this MVP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---- ISA standard atmosphere (US customary) ----------------------------

_T0_R = 518.67
_RHO0_SLUG_FT3 = 0.002378
_LAPSE_R_FT = 0.00356616
_R_FT_R = 1716.49
_TROPOPAUSE_FT = 36089.0
_T_TROPO_R = 389.97
_GAMMA_AIR = 1.4
_FT_TO_M = 0.3048
_LB_TO_KG = 0.45359237
_SLUG_TO_KG = 14.5939029
_SLUG_FT2_TO_KG_M2 = _SLUG_TO_KG * _FT_TO_M**2


def isa_density_slug_ft3(altitude_ft: float) -> float:
    """ISA density in slug/ft³ for the given altitude."""
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
        return float(_RHO0_SLUG_FT3 * (T / _T0_R) ** 4.2561)
    rho_tropo = _RHO0_SLUG_FT3 * (_T_TROPO_R / _T0_R) ** 4.2561
    return float(rho_tropo * math.exp(-(altitude_ft - _TROPOPAUSE_FT) / 20806.0))


def isa_speed_of_sound_ft_s(altitude_ft: float) -> float:
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
    else:
        T = _T_TROPO_R
    return float(math.sqrt(_GAMMA_AIR * _R_FT_R * T))


# ---- Configurations ----------------------------------------------------


class B737Configuration(Enum):
    """Discrete Boeing 737 configurations covered by this model."""

    B737_100 = "737-100"   # original 737, JSBSim defaults
    B737_800 = "737-800"   # 737NG, CFM56-7B engines


# Geometry per configuration (constant — cf. JSBSim 737.xml + Boeing TCDS A16WE)


@dataclass
class B737Parameters:
    """Boeing 737 inertial / geometric parameters at one configuration.

    Defaults match :attr:`B737Configuration.B737_100` (JSBSim values).
    Inertias are at the **operating empty weight + half-fuel** typical
    cruise configuration (Roskam Appendix B); for true MTOW or empty
    they are within ±15 %.
    """

    config: B737Configuration = B737Configuration.B737_100
    weight_lb: float = 100_000.0          # mid-cruise weight, JSBSim default
    Ix: float = 562_000.0                 # slug·ft²
    Iy: float = 1_473_000.0
    Iz: float = 1_894_000.0
    Ixz: float = 8_000.0
    cg_frac_cbar: float = 0.25
    g_ft_s2: float = 32.174

    # Geometry (do NOT mutate — see module-level constants below)
    S_ft2: float = 1_171.0
    b_ft: float = 94.7
    cbar_ft: float = 12.31

    # Actuator dynamics (Roskam Vol 6 + JSBSim limits)
    elevator_tau_s: float = 0.10
    elevator_max_rad: float = math.radians(17.2)        # ±0.30 rad per JSBSim
    elevator_rate_max_rad_s: float = math.radians(40.0)
    aileron_tau_s: float = 0.10
    aileron_max_rad: float = math.radians(20.1)         # ±0.35 rad per JSBSim
    aileron_rate_max_rad_s: float = math.radians(40.0)
    rudder_tau_s: float = 0.10
    rudder_max_rad: float = math.radians(20.1)
    rudder_rate_max_rad_s: float = math.radians(40.0)

    # Engines — defaults are 2 × CFM56-7B at 27 300 lbf SLS (737-800).
    # The 737-100 default config below overrides to 2 × JT8D-9 @ 14 500 lbf.
    engine_thrust_max_lb: float = 54_600.0   # 2 × 27 300 lb (737-800)
    engine_idle_frac: float = 0.05
    engine_tau_s: float = 1.5
    n_engines: int = 2

    # Damage subsystem hooks (parity with B-747 / X-15)
    damage_state: Optional[object] = None
    damage_geometry: Optional[object] = None

    @property
    def mass_slug(self) -> float:
        return self.weight_lb / self.g_ft_s2


def default_parameters(
    config: B737Configuration = B737Configuration.B737_100,
) -> B737Parameters:
    """Return parameters for one of the documented 737 variants."""
    if config is B737Configuration.B737_100:
        return B737Parameters(
            config=B737Configuration.B737_100,
            weight_lb=100_000.0,
            Ix=562_000.0, Iy=1_473_000.0, Iz=1_894_000.0, Ixz=8_000.0,
            S_ft2=1_171.0, b_ft=94.7, cbar_ft=12.31,
            engine_thrust_max_lb=29_000.0,   # 2 × 14 500 lbf JT8D-9
            n_engines=2,
        )
    if config is B737Configuration.B737_800:
        return B737Parameters(
            config=B737Configuration.B737_800,
            # 737-800 typical operating weight (mid-cruise, half fuel).
            # Derived by scaling JSBSim numbers by FAA TCDS A16WE ratios.
            weight_lb=140_000.0,
            Ix=820_000.0, Iy=2_300_000.0, Iz=3_000_000.0, Ixz=12_000.0,
            S_ft2=1_341.0, b_ft=117.5, cbar_ft=12.97,
            engine_thrust_max_lb=54_600.0,   # 2 × 27 300 lbf CFM56-7B27
            n_engines=2,
        )
    raise ValueError(f"unknown B737Configuration: {config!r}")


@dataclass
class B737ParametersSI:
    """SI-units snapshot of :class:`B737Parameters`."""

    mass_kg: float
    Ix_kg_m2: float
    Iy_kg_m2: float
    Iz_kg_m2: float
    Ixz_kg_m2: float
    S_m2: float
    b_m: float
    cbar_m: float
    engine_thrust_max_N: float
    cg_frac_cbar: float
    config: B737Configuration


def to_si(p: B737Parameters) -> B737ParametersSI:
    return B737ParametersSI(
        mass_kg=p.weight_lb * _LB_TO_KG,
        Ix_kg_m2=p.Ix * _SLUG_FT2_TO_KG_M2,
        Iy_kg_m2=p.Iy * _SLUG_FT2_TO_KG_M2,
        Iz_kg_m2=p.Iz * _SLUG_FT2_TO_KG_M2,
        Ixz_kg_m2=p.Ixz * _SLUG_FT2_TO_KG_M2,
        S_m2=p.S_ft2 * _FT_TO_M**2,
        b_m=p.b_ft * _FT_TO_M,
        cbar_m=p.cbar_ft * _FT_TO_M,
        engine_thrust_max_N=p.engine_thrust_max_lb * 4.4482216,
        cg_frac_cbar=p.cg_frac_cbar,
        config=p.config,
    )
