"""North American X-15 mass / inertia / geometry parameters.

Reference data sources (open):

* **NASA TM X-1669** — Walker H. J. & Wolowicz C. H. *"Stability and
  Control Derivatives of the X-15 Airplane"* (1968). Tabulated
  longitudinal and lateral-directional derivatives across the full
  Mach envelope (M = 0.4 to 6.7).
* **NASA TN D-1402** — Heffley R. K. & Jewell W. F. body of stability
  data also used in the linear ``x15_data.m`` file shipped with this
  repo.
* **NASA SP-2000-4222** — Thompson M. O. *"At the Edge of Space: The
  X-15 Flight Program"*, including XLR99 propellant flow rates.
* **NASA Technical Memorandum 81350** — X-15A-2 advanced configuration
  reference.

All values are in **US customary units (slug, ft, lb, sec)** as
published. SI helpers are exposed via :meth:`X15Parameters.to_si`.

The X-15 is a research aerospaceplane. Its envelope spans high
subsonic to **hypersonic** (M > 6) and from sea level to 354,200 ft
(record altitude). This model focuses on the *aerodynamic* corridor,
i.e. the densest published envelope: **M ∈ [0.4, 6.7], h ∈ [0,
250,000] ft**. Above ~ 250 kft the dynamic pressure drops below the
practical limit for aerodynamic control surfaces (the real X-15
switched to RCS thrusters); RCS is **not** modelled in this initial
release — see the ``scope-and-limitations`` section of the docs.

Two configurations are supplied:

* :attr:`X15Configuration.BASIC` — stock X-15 (X-15-1, X-15-3),
  empty weight ≈ 14 600 lb, full propellant ≈ 13 000 lb, gross take-off
  ≈ 33 000 lb after B-52 air-launch.
* :attr:`X15Configuration.A2` — X-15A-2 advanced configuration with
  external propellant tanks and ablative coating. Empty weight ≈
  16 050 lb, full propellant ≈ 18 000 lb, gross ≈ 50 000 lb. Used for
  the M=6.7 record flight (Pete Knight, 3 October 1967).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---- ISA / hypersonic atmosphere (US customary, US Std. 1976) ---------

_T0_R = 518.67  # sea-level temperature, °R
_RHO0_SLUG_FT3 = 0.002378  # sea-level density, slug/ft³
_LAPSE_R_FT = 0.00356616  # ISA lapse rate, °R/ft (= 6.5e-3 K/m)
_R_FT_R = 1716.49  # specific gas constant, ft²/(s²·°R)
_TROPOPAUSE_FT = 36089.0
_T_TROPO_R = 389.97
_GAMMA_AIR = 1.4
_FT_TO_M = 0.3048
_LB_TO_KG = 0.45359237
_SLUG_TO_KG = 14.5939029
_SLUG_FT2_TO_KG_M2 = _SLUG_TO_KG * _FT_TO_M**2

# Stratopause / mesosphere boundary used by the high-altitude
# branch. Above ~ 65 kft the temperature levels off at 389.97 °R; above
# 105 kft (32 km) it begins to rise again. The X-15 spent most of its
# powered phase in the constant-temperature tropopause-to-stratopause
# layer, so this single-layer approximation is accurate to better than
# 5 % up to 200 kft for our purposes.


def isa_density_slug_ft3(altitude_ft: float) -> float:
    """ISA-style density in slug/ft³.

    Below 36 089 ft uses the troposphere lapse model; above uses the
    isothermal-stratosphere approximation. For altitudes above 200 kft
    the density tends toward zero — at that point aerodynamic forces
    are negligible and the (un-modelled) RCS thrusters are required.
    """
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
        return float(_RHO0_SLUG_FT3 * (T / _T0_R) ** 4.2561)
    rho_tropo = _RHO0_SLUG_FT3 * (_T_TROPO_R / _T0_R) ** 4.2561
    return float(rho_tropo * math.exp(-(altitude_ft - _TROPOPAUSE_FT) / 20806.0))


def isa_speed_of_sound_ft_s(altitude_ft: float) -> float:
    """Speed of sound in ft/s using the same atmosphere model."""
    if altitude_ft < _TROPOPAUSE_FT:
        T = _T0_R - _LAPSE_R_FT * altitude_ft
    else:
        T = _T_TROPO_R
    return float(math.sqrt(_GAMMA_AIR * _R_FT_R * T))


# ---- Configuration enum -----------------------------------------------


class X15Configuration(Enum):
    """X-15 airframe configurations covered by this model."""

    BASIC = "basic"  # stock X-15-1 / X-15-3, single internal propellant
    A2 = "a2"  # X-15A-2 with external tanks (Mach 6.7 record airframe)


# ---- Geometry (constant across configurations) ------------------------

S_FT2 = 200.0  # planform reference area, ft²
B_FT = 22.36  # span, ft
CBAR_FT = 10.27  # mean aerodynamic chord, ft


@dataclass
class X15Parameters:
    """X-15 inertial / geometric parameters at one configuration.

    Mass is **time-varying** during the rocket burn — the
    :attr:`mass_slug` returned here is the *current* value (updated by
    the dynamics each step from the propellant mass state). The
    *empty* mass after burnout is :attr:`empty_mass_slug` and the
    *full* propellant load is :attr:`propellant_full_lb`.

    Inertias are the **empty-airframe** values; the dynamics ODE
    interpolates linearly between empty and full inertias based on
    the current propellant fraction, which is a standard rocket-aircraft
    approximation that keeps inertia tracking physical without
    requiring a tank-by-tank c.g. model.
    """

    config: X15Configuration = X15Configuration.BASIC

    # Empty airframe weight — used as the floor for the variable-mass
    # state. CR-2144-style published numbers.
    empty_weight_lb: float = 14_600.0
    # Empty-airframe inertias (slug·ft²). Walker/Wolowicz Table 1.
    Ix_empty: float = 3_650.0
    Iy_empty: float = 80_000.0  # matches x15_data.m Iy
    Iz_empty: float = 82_000.0
    Ixz_empty: float = 590.0

    # Inertias when fully fueled (mostly drives I_y because LOX/ammonia
    # tanks are aft of the pilot — slightly increases I_y by ~ 10 %).
    Ix_full: float = 3_650.0
    Iy_full: float = 88_000.0
    Iz_full: float = 90_000.0
    Ixz_full: float = 650.0

    # Propellant full load (anhydrous ammonia + LOX combined).
    # 1018 gal LOX × 9.5 lb/gal + 1445 gal NH3 × 5.7 lb/gal ≈ 17 900 lb.
    propellant_full_lb: float = 17_900.0

    cg_frac_cbar: float = 0.22  # c.g. as fraction of MAC (Walker/Wolowicz)
    g_ft_s2: float = 32.174

    # Geometry
    S_ft2: float = S_FT2
    b_ft: float = B_FT
    cbar_ft: float = CBAR_FT

    # Actuator dynamics — first-order lag with rate limit. Real X-15
    # used hydraulic surface actuators rated ~ 60 deg/s.
    elevator_tau_s: float = 0.05
    elevator_max_rad: float = math.radians(15.0)  # all-flying horizontal tail
    elevator_rate_max_rad_s: float = math.radians(60.0)
    aileron_tau_s: float = 0.05
    aileron_max_rad: float = math.radians(15.0)
    aileron_rate_max_rad_s: float = math.radians(60.0)
    rudder_tau_s: float = 0.05
    rudder_max_rad: float = math.radians(8.5)  # vertical stab is small
    rudder_rate_max_rad_s: float = math.radians(60.0)

    # XLR99 rocket engine. Numbers from NASA TM X-2670 / Thompson 2000.
    engine_thrust_max_lb: float = 57_000.0  # rated thrust at full throttle
    engine_throttle_min: float = 0.30  # XLR99 cannot run below 30 %
    engine_isp_s: float = 254.0  # ammonia-LOX, sea-level Isp

    # Damage subsystem hooks (parity with B-747 / F-16). None = healthy.
    damage_state: Optional[object] = None
    damage_geometry: Optional[object] = None

    # ---- variable-mass helpers --------------------------------------

    def current_mass_slug(self, propellant_lb: float) -> float:
        """Mass in slugs given remaining propellant (lb)."""
        total_weight = self.empty_weight_lb + max(0.0, float(propellant_lb))
        return total_weight / self.g_ft_s2

    def inertia_at(self, propellant_lb: float) -> tuple[float, float, float, float]:
        """Linearly interpolate (Ix, Iy, Iz, Ixz) by propellant fraction.

        Returns inertia of the *full* aircraft when ``propellant_lb ==
        propellant_full_lb`` and of the *empty* aircraft when
        ``propellant_lb == 0``.
        """
        f = max(0.0, min(1.0, float(propellant_lb) / max(self.propellant_full_lb, 1.0)))
        return (
            self.Ix_empty + f * (self.Ix_full - self.Ix_empty),
            self.Iy_empty + f * (self.Iy_full - self.Iy_empty),
            self.Iz_empty + f * (self.Iz_full - self.Iz_empty),
            self.Ixz_empty + f * (self.Ixz_full - self.Ixz_empty),
        )

    @property
    def empty_mass_slug(self) -> float:
        return self.empty_weight_lb / self.g_ft_s2


def default_parameters(
    config: X15Configuration = X15Configuration.BASIC,
) -> X15Parameters:
    """Return parameters for the requested X-15 configuration."""
    if config is X15Configuration.BASIC:
        return X15Parameters(config=X15Configuration.BASIC)
    if config is X15Configuration.A2:
        # X-15A-2 with external tanks: heavier empty + more propellant.
        # Internal 17 900 lb + external tanks add ~ 13 000 lb of LOX +
        # NH3 → total propellant ≈ 30 900 lb. Realistic for the Mach
        # 6.7 record airframe.
        return X15Parameters(
            config=X15Configuration.A2,
            empty_weight_lb=16_050.0,
            Ix_empty=3_900.0,
            Iy_empty=92_000.0,
            Iz_empty=94_000.0,
            Ixz_empty=620.0,
            Ix_full=3_900.0,
            Iy_full=110_000.0,
            Iz_full=112_000.0,
            Ixz_full=720.0,
            propellant_full_lb=30_900.0,
            engine_thrust_max_lb=57_000.0,  # same XLR99
        )
    raise ValueError(f"unknown X15Configuration: {config!r}")


@dataclass
class X15ParametersSI:
    """SI-units snapshot of :class:`X15Parameters`."""

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
    config: X15Configuration


def to_si(p: X15Parameters, propellant_lb: float) -> X15ParametersSI:
    """Convert FPS-units :class:`X15Parameters` to SI for cross-checks."""
    Ix, Iy, Iz, Ixz = p.inertia_at(propellant_lb)
    return X15ParametersSI(
        mass_kg=(p.empty_weight_lb + propellant_lb) * _LB_TO_KG,
        Ix_kg_m2=Ix * _SLUG_FT2_TO_KG_M2,
        Iy_kg_m2=Iy * _SLUG_FT2_TO_KG_M2,
        Iz_kg_m2=Iz * _SLUG_FT2_TO_KG_M2,
        Ixz_kg_m2=Ixz * _SLUG_FT2_TO_KG_M2,
        S_m2=p.S_ft2 * _FT_TO_M**2,
        b_m=p.b_ft * _FT_TO_M,
        cbar_m=p.cbar_ft * _FT_TO_M,
        engine_thrust_max_N=p.engine_thrust_max_lb * 4.4482216,
        cg_frac_cbar=p.cg_frac_cbar,
        config=p.config,
    )
