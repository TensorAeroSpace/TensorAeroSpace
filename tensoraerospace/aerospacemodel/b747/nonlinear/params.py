"""Boeing 747-100 mass / inertia / geometry parameters from NASA CR-2144 §IX.

Reference: Heffley R. K., Jewell W. F., *Aircraft Handling Qualities Data*,
NASA CR-2144, December 1972, Systems Technology Inc.

All values are in **US customary units (slug, ft)** as published. SI
helpers are exposed via :meth:`B747Parameters.to_si`.

Three configurations are provided:

* ``Configuration.NOMINAL`` — TOGW less 40% fuel, c.g. at 0.25 c̄,
  cruise (clean) — pages 212, 217-228 of CR-2144.
* ``Configuration.POWER_APPROACH`` — Max landing weight, 20° flaps, gear
  up, 1.4·V_s — Table IX-2.
* ``Configuration.LANDING`` — Power approach configuration with 30°
  flaps and gear down — Table IX-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---- ISA standard atmosphere (US customary) ----------------------------

_T0_R = 518.67  # sea-level temperature, °R
_RHO0_SLUG_FT3 = 0.002378  # sea-level density, slug/ft³
_LAPSE_R_FT = 0.00356616  # ISA lapse rate, °R/ft (= 6.5e-3 K/m)
_R_FT_R = 1716.49  # specific gas constant for air, ft²/(s²·°R)
_TROPOPAUSE_FT = 36089.0
_T_TROPO_R = 389.97
_GAMMA_AIR = 1.4
_FT_TO_M = 0.3048
_LB_TO_KG = 0.45359237
_SLUG_TO_KG = 14.5939029
_SLUG_FT2_TO_KG_M2 = _SLUG_TO_KG * _FT_TO_M**2


def isa_density_slug_ft3(altitude_ft: float) -> float:
    """ISA density in slug/ft³ for the given altitude (ft)."""
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
        return float(_RHO0_SLUG_FT3 * (T / _T0_R) ** 4.2561)
    rho_tropo = _RHO0_SLUG_FT3 * (_T_TROPO_R / _T0_R) ** 4.2561
    return float(rho_tropo * math.exp(-(altitude_ft - _TROPOPAUSE_FT) / 20806.0))


def isa_speed_of_sound_ft_s(altitude_ft: float) -> float:
    """ISA speed of sound in ft/s."""
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
    else:
        T = _T_TROPO_R
    return float(math.sqrt(_GAMMA_AIR * _R_FT_R * T))


# ---- Configurations ----------------------------------------------------


class B747Configuration(Enum):
    """Discrete B-747 configurations covered in NASA CR-2144 §IX."""

    NOMINAL = "nominal"  # TOGW − 40% fuel, clean
    POWER_APPROACH = "power_approach"  # 20° flaps, gear up, 1.4 V_s
    LANDING = "landing"  # 30° flaps, gear down, 1.2 V_s


# Geometry — common to all three configurations (CR-2144 figure IX-2)
S_FT2 = 5500.0
B_FT = 195.68
CBAR_FT = 27.31


@dataclass
class B747Parameters:
    """B-747 inertial / geometric parameters at one configuration.

    Slugs and feet, body-axis inertia tensor, c.g. at 0.25 c̄.

    Defaults correspond to :attr:`B747Configuration.NOMINAL`.
    """

    config: B747Configuration = B747Configuration.NOMINAL
    weight_lb: float = 636_600.0  # CR-2144 p. 212 (TOGW − 40% fuel)
    Ix: float = 18.2e6  # slug·ft²
    Iy: float = 33.1e6
    Iz: float = 49.7e6
    Ixz: float = 0.97e6
    cg_frac_cbar: float = 0.25  # c.g. as fraction of mean aerodynamic chord
    g_ft_s2: float = 32.174

    # Geometry (do NOT mutate — see module-level constants above)
    S_ft2: float = S_FT2
    b_ft: float = B_FT
    cbar_ft: float = CBAR_FT

    # Actuator dynamics — 1st-order lag with rate limit. CR-2144 doesn't
    # publish actuator data; values here are conservative defaults
    # representative of late-1960s wide-body transports.
    elevator_tau_s: float = 0.10
    elevator_max_rad: float = math.radians(25.0)
    elevator_rate_max_rad_s: float = math.radians(40.0)
    aileron_tau_s: float = 0.10
    aileron_max_rad: float = math.radians(20.0)
    aileron_rate_max_rad_s: float = math.radians(40.0)
    rudder_tau_s: float = 0.10
    rudder_max_rad: float = math.radians(25.0)
    rudder_rate_max_rad_s: float = math.radians(40.0)

    # Engine — 4 × Pratt & Whitney JT9D-7. Numbers from Boeing 747-100
    # type certificate; idle thrust assumed 5% of max.
    engine_thrust_max_lb: float = 188_400.0  # 4 × 47,100 lb sea-level static
    engine_idle_frac: float = 0.05
    engine_tau_s: float = 1.0  # spool-up time constant

    # Damage subsystem hooks (parity with F-16). None = healthy.
    damage_state: Optional[object] = None
    damage_geometry: Optional[object] = None

    @property
    def mass_slug(self) -> float:
        """Mass in slugs (weight / g)."""
        return self.weight_lb / self.g_ft_s2

    def to_si(self) -> "B747ParametersSI":
        """Return a SI-units (kg, m, N) snapshot of these parameters."""
        return B747ParametersSI(
            mass_kg=self.weight_lb * _LB_TO_KG,
            Ix_kg_m2=self.Ix * _SLUG_FT2_TO_KG_M2,
            Iy_kg_m2=self.Iy * _SLUG_FT2_TO_KG_M2,
            Iz_kg_m2=self.Iz * _SLUG_FT2_TO_KG_M2,
            Ixz_kg_m2=self.Ixz * _SLUG_FT2_TO_KG_M2,
            S_m2=self.S_ft2 * _FT_TO_M**2,
            b_m=self.b_ft * _FT_TO_M,
            cbar_m=self.cbar_ft * _FT_TO_M,
            engine_thrust_max_N=self.engine_thrust_max_lb * 4.4482216,
            cg_frac_cbar=self.cg_frac_cbar,
            config=self.config,
        )


@dataclass
class B747ParametersSI:
    """SI-units snapshot of :class:`B747Parameters`."""

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
    config: B747Configuration


def default_parameters(
    config: B747Configuration = B747Configuration.NOMINAL,
) -> B747Parameters:
    """Return parameters for one of the three CR-2144 configurations."""
    if config is B747Configuration.NOMINAL:
        return B747Parameters(
            config=B747Configuration.NOMINAL,
            weight_lb=636_600.0,
            Ix=18.2e6,
            Iy=33.1e6,
            Iz=49.7e6,
            Ixz=0.97e6,
        )
    if config is B747Configuration.POWER_APPROACH:
        return B747Parameters(
            config=B747Configuration.POWER_APPROACH,
            weight_lb=564_000.0,
            Ix=13.7e6,
            Iy=30.5e6,
            Iz=43.1e6,
            Ixz=0.825e6,
        )
    if config is B747Configuration.LANDING:
        return B747Parameters(
            config=B747Configuration.LANDING,
            weight_lb=564_000.0,  # CR-2144 lists Landing under Max Landing Weight
            Ix=13.7e6,
            Iy=30.5e6,
            Iz=43.1e6,
            Ixz=0.825e6,
        )
    raise ValueError(f"unknown B747Configuration: {config!r}")
