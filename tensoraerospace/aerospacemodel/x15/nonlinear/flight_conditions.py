"""Anchor flight conditions for the X-15 aerodynamic envelope.

Five published trim points spanning the X-15's powered-flight corridor,
distilled from NASA TM X-1669 (Walker & Wolowicz, 1968) Table 2 and
NASA TN D-1402 stability-derivative tabulations.

The model interpolates aerodynamic coefficients linearly in (M, h)
between these anchors. Outside the envelope, the nearest anchor is
clamped, with a published-data warning printed by ``aero.b747_aero``
when the operating point lies more than 0.5 Mach or 50 kft outside
the bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class X15FlightCondition:
    """One published anchor in the X-15 envelope."""

    fc_id: int
    label: str
    altitude_ft: float
    mach: float
    V_ft_s: float
    alpha0_deg: float           # trim α at the anchor
    elevator0_deg: float        # trim elevator deflection
    propellant_lb: float        # remaining propellant at the anchor


# Walker/Wolowicz Table 2 + Thompson 2000 mission profiles.
#
# FC1 — drop @ 45 kft, M=0.83, just before XLR99 ignition (boost start).
# FC2 — boost climb @ 70 kft, M=2.5, full thrust, propellant ~80 % of full.
# FC3 — peak velocity in atmospheric flight @ 100 kft, M=4, propellant 50 %.
# FC4 — high-altitude push-over @ 200 kft, M=5, propellant exhausted (coast).
# FC5 — A2-style hypersonic record @ 102 kft, M=6.7, ablative, A2 mass.
X15_FLIGHT_CONDITIONS: list[X15FlightCondition] = [
    X15FlightCondition(
        fc_id=1, label="boost_start",
        altitude_ft=45_000.0, mach=0.83, V_ft_s=796.5,
        alpha0_deg=4.5, elevator0_deg=-2.5,
        propellant_lb=13_000.0,
    ),
    X15FlightCondition(
        fc_id=2, label="boost_climb",
        altitude_ft=70_000.0, mach=2.5, V_ft_s=2_412.0,
        alpha0_deg=5.0, elevator0_deg=-3.0,
        propellant_lb=10_500.0,
    ),
    X15FlightCondition(
        fc_id=3, label="cruise_M4",
        altitude_ft=100_000.0, mach=4.0, V_ft_s=3_865.0,
        alpha0_deg=4.0, elevator0_deg=-2.0,
        propellant_lb=6_500.0,
    ),
    X15FlightCondition(
        fc_id=4, label="coast_high",
        altitude_ft=200_000.0, mach=5.0, V_ft_s=4_876.0,
        alpha0_deg=10.0, elevator0_deg=-1.0,
        propellant_lb=0.0,
    ),
    X15FlightCondition(
        fc_id=5, label="hypersonic_record",
        altitude_ft=102_000.0, mach=6.7, V_ft_s=6_525.0,
        alpha0_deg=4.5, elevator0_deg=-2.0,
        propellant_lb=2_000.0,
    ),
]


def get_flight_condition(fc_id: int) -> X15FlightCondition:
    """Return the FC with the given 1-based id."""
    for fc in X15_FLIGHT_CONDITIONS:
        if fc.fc_id == fc_id:
            return fc
    raise ValueError(
        f"unknown flight_condition_id={fc_id}; available: "
        f"{[fc.fc_id for fc in X15_FLIGHT_CONDITIONS]}"
    )
