"""B-747 trim flight conditions from NASA CR-2144 Table IX-3.

The 10 flight conditions that anchor the published derivative tables.
Conditions 1-2 are sea-level approach configurations; 3-10 are the
clean cruise grid (SL/20K/40K × Mach 0.45/0.65/0.5/0.65/0.8/0.7/0.8/0.9).
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import B747Configuration


@dataclass(frozen=True)
class B747FlightCondition:
    """One trim point published in CR-2144 Table IX-3.

    ``alpha0_deg`` and ``stab_trim_deg`` are the trimmed angle of attack
    and stabilizer setting at this point. Use :meth:`q_psf` for the
    dynamic pressure exactly as reported in the table.
    """

    fc_id: int
    altitude_ft: float
    mach: float
    V_ft_s: float
    V_ktas: float
    weight_lb: float
    cg_frac_cbar: float
    Ix: float
    Iy: float
    Iz: float
    Ixz: float
    epsilon_ca_deg: float       # CR-2144 EPSILCNDEG: c.g. arm to ALPHA, ε_CA
    q_psf: float                 # dynamic pressure, lb/ft²
    qc_psf: float                # impact pressure, lb/ft²
    alpha0_deg: float
    gamma_deg: float
    stab_trim_deg: float
    config: B747Configuration


# CR-2144 Table IX-3, columns 1..10. Body axis system. Quantities not
# directly used by the dynamics module (LXP, LZP, ITH, XII, LTH) are
# omitted; they describe pilot station offsets used only for the
# acceleration transfer functions (a_z and a_y at pilot's seat).
B747_FLIGHT_CONDITIONS: list[B747FlightCondition] = [
    B747FlightCondition(  # FC1 — Landing
        fc_id=1, altitude_ft=0.0, mach=0.198,
        V_ft_s=221.0, V_ktas=131.0,
        weight_lb=564_032.0, cg_frac_cbar=0.250,
        Ix=0.142e8, Iy=0.323e8, Iz=0.454e8, Ixz=0.870e6,
        epsilon_ca_deg=-1.460,
        q_psf=58.1, qc_psf=58.7,
        alpha0_deg=8.50, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.LANDING,
    ),
    B747FlightCondition(  # FC2 — Power Approach
        fc_id=2, altitude_ft=0.0, mach=0.249,
        V_ft_s=278.0, V_ktas=165.0,
        weight_lb=564_032.0, cg_frac_cbar=0.250,
        Ix=0.142e8, Iy=0.323e8, Iz=0.454e8, Ixz=0.870e6,
        epsilon_ca_deg=-1.460,
        q_psf=92.2, qc_psf=93.6,
        alpha0_deg=5.70, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.POWER_APPROACH,
    ),
    B747FlightCondition(  # FC3 — SL × M=0.45
        fc_id=3, altitude_ft=0.0, mach=0.450,
        V_ft_s=502.0, V_ktas=298.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=300.0, qc_psf=315.0,
        alpha0_deg=3.10, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC4 — SL × M=0.65
        fc_id=4, altitude_ft=0.0, mach=0.650,
        V_ft_s=726.0, V_ktas=430.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=626.0, qc_psf=695.0,
        alpha0_deg=0.0, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC5 — 20K × M=0.50
        fc_id=5, altitude_ft=20_000.0, mach=0.500,
        V_ft_s=518.0, V_ktas=307.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=170.0, qc_psf=181.0,
        alpha0_deg=6.80, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC6 — 20K × M=0.65
        fc_id=6, altitude_ft=20_000.0, mach=0.650,
        V_ft_s=674.0, V_ktas=399.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=288.0, qc_psf=320.0,
        alpha0_deg=2.50, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC7 — 20K × M=0.80
        fc_id=7, altitude_ft=20_000.0, mach=0.800,
        V_ft_s=830.0, V_ktas=492.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=436.0, qc_psf=510.0,
        alpha0_deg=0.0, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC8 — 40K × M=0.70
        fc_id=8, altitude_ft=40_000.0, mach=0.700,
        V_ft_s=678.0, V_ktas=402.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=135.0, qc_psf=153.0,
        alpha0_deg=7.30, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC9 — 40K × M=0.80
        fc_id=9, altitude_ft=40_000.0, mach=0.800,
        V_ft_s=774.0, V_ktas=459.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=177.0, qc_psf=207.0,
        alpha0_deg=4.60, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
    B747FlightCondition(  # FC10 — 40K × M=0.90
        fc_id=10, altitude_ft=40_000.0, mach=0.900,
        V_ft_s=871.0, V_ktas=516.0,
        weight_lb=636_636.0, cg_frac_cbar=0.250,
        Ix=0.182e8, Iy=0.331e8, Iz=0.497e8, Ixz=0.970e6,
        epsilon_ca_deg=-1.760,
        q_psf=224.0, qc_psf=273.0,
        alpha0_deg=2.40, gamma_deg=0.0, stab_trim_deg=-10.0,
        config=B747Configuration.NOMINAL,
    ),
]


def get_flight_condition(fc_id: int) -> B747FlightCondition:
    """Return the trim flight condition with the given ID (1..10)."""
    if not 1 <= fc_id <= 10:
        raise ValueError(f"fc_id must be in [1, 10], got {fc_id}")
    return B747_FLIGHT_CONDITIONS[fc_id - 1]
