"""Quadrotor inertial / geometric parameters.

Defaults follow a typical 1.5 kg research quadrotor (e.g. DJI F450 frame
or Crazyflie-class scaled up). Inertia values match the AscTec Hummingbird
/ AscTec Pelican-class platforms widely used in academic FTC papers
(Lu 2019, Lanzon 2015), expressed in the body NED frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuadrotorParameters:
    """Inertial + geometric parameters of a single rigid-body quadrotor."""

    # Mass and gravity
    m: float = 1.5             # mass, kg
    g: float = 9.81            # gravitational acceleration, m/s²

    # Diagonal inertia tensor about body-frame principal axes (kg·m²)
    Jx: float = 0.0211
    Jy: float = 0.0211         # symmetric: Jx == Jy for an X / + frame
    Jz: float = 0.0366

    # Geometry
    arm_length: float = 0.225  # rotor radius from CG, m

    # Aerodynamic linear drag (body-frame translational damping, N·s/m).
    # Matches typical low-speed hover regime; the model is dominated by
    # rigid-body dynamics, drag is a small correction.
    kdx: float = 0.10
    kdy: float = 0.10
    kdz: float = 0.20

    # Actuator limits — used by the env layer for clipping; the bare ODE
    # is happy with any input.
    thrust_max: float = 30.0    # N (≈ 2·m·g, typical 2:1 thrust-to-weight)
    thrust_min: float = 0.0
    torque_max: float = 1.5     # N·m, per-axis


def default_parameters() -> QuadrotorParameters:
    """Return a fresh `QuadrotorParameters` with default values."""
    return QuadrotorParameters()
