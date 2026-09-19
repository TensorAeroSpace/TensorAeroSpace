"""Classical attitude and longitudinal loops for nonlinear aircraft.

B747 longitudinal gains reproduce the documented 20,000 ft / 674 ft/s
benchmark profile. Reassess gains and actuator authority at other trims.
"""

import numpy as np

from tensoraerospace.aerospacemodel.b747.nonlinear import trim

from . import PID

B747_LATERAL_PID_GAINS = (
    27.72535093846033,
    0.6523737554404134,
    4.729243701446712,
    1.34289301767933,
    0.026054165115317957,
    6.609208606368212,
)


class B747LongitudinalHold:
    """Same measured-state PI/PD hold for both arms; no event/time input."""

    def __init__(
        self, dt, *, trim_result=None, altitude_ft=20000.0, airspeed_ft_s=674.0
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.altitude_ft = float(altitude_ft)
        self.airspeed_ft_s = float(airspeed_ft_s)
        self.trim_result = (
            trim_result
            if trim_result is not None
            else trim(self.altitude_ft, self.airspeed_ft_s)
        )
        if not self.trim_result.converged:
            raise ValueError("Longitudinal hold requires a converged trim")
        self.dt = dt
        self.int_v = self.int_h = self.hdot = 0.0
        self.prev_h = self.altitude_ft
        tr = self.trim_result
        self.elevator = tr.elevator_rad
        self.throttle = tr.throttle

    def command(self, obs, *, speed_ref_ft_s=None, height_ref_ft=None):
        speed_ref_ft_s = (
            self.airspeed_ft_s if speed_ref_ft_s is None else speed_ref_ft_s
        )
        height_ref_ft = self.altitude_ft if height_ref_ft is None else height_ref_ft
        tr = self.trim_result
        dt = self.dt
        ev = float(np.linalg.norm(obs[:3]) - speed_ref_ft_s)
        h = float(-obs[11])
        eh = h - height_ref_ft
        self.hdot += (1 - np.exp(-dt / 0.4)) * ((h - self.prev_h) / dt - self.hdot)
        self.prev_h = h
        self.int_v = float(np.clip(self.int_v + ev * dt, -800, 800))
        self.int_h = float(np.clip(self.int_h + eh * dt, -2000, 2000))
        pitch = np.rad2deg(tr.theta_rad) + np.clip(
            -0.0012 * eh - 4e-5 * self.int_h - 0.025 * self.hdot, -4, 4
        )
        delta = float(
            np.clip(
                0.7 * (np.rad2deg(obs[7]) - pitch) + 0.7 * np.rad2deg(obs[4]), -6, 6
            )
        )
        target = tr.elevator_rad + np.deg2rad(delta)
        self.elevator += float(
            np.clip(target - self.elevator, -np.deg2rad(0.7) * dt, np.deg2rad(0.7) * dt)
        )
        target = float(np.clip(tr.throttle - 0.010 * ev - 0.0015 * self.int_v, 0.2, 1))
        self.throttle += float(np.clip(target - self.throttle, -0.12 * dt, 0.12 * dt))
        return self.elevator, self.throttle


class LateralAircraftPID:
    """Two SDK PID loops using measured Euler-angle rates, with degree outputs.

    Input: the 12-state Boeing layout (angles/rates in radians). Positive
    aileron and rudder signs follow the B747 model. ``gains`` contains roll
    Kp/Ki/Kd then heading Kp/Ki/Kd, fitted in degrees. No fault inputs exist.
    """

    def __init__(
        self,
        gains=B747_LATERAL_PID_GAINS,
        dt=0.02,
        *,
        magnitude_limit_deg=8.0,
        rate_limit_deg_s=None,
    ):
        self.gains = np.asarray(gains, dtype=float).reshape(2, 3).copy()
        if (
            not np.isfinite(self.gains).all()
            or not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(magnitude_limit_deg)
            or magnitude_limit_deg <= 0
        ):
            raise ValueError("Finite gains and positive dt/magnitude limit required")
        self.dt = dt
        self.controllers = [
            PID(
                kp=sign * g[0],
                ki=sign * g[1],
                kd=sign * g[2],
                dt=dt,
                output_limits=(-magnitude_limit_deg, magnitude_limit_deg),
                rate_limit=rate_limit_deg_s,
            )
            for sign, g in zip((1, -1), self.gains)
        ]

    @property
    def integrals(self):
        """Measured-minus-reference integrals, in degree-seconds."""
        return -np.array([pid.integral for pid in self.controllers])

    @integrals.setter
    def integrals(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("integrals must contain two finite values")
        for pid, value in zip(self.controllers, values):
            pid.integral = -float(value)

    def reset(self):
        """Reset both PID histories for a fresh episode."""
        for pid in self.controllers:
            pid.reset()

    def command(
        self, obs, *, roll_ref_deg=0.0, heading_ref_deg=0.0, applied_output=None
    ):
        """Return aileron and rudder commands in degrees."""
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (12,) or not np.isfinite(obs).all():
            raise ValueError("Expected a finite 12-state aircraft observation")
        phi, theta = obs[6:8]
        if abs(np.cos(theta)) < 1e-3:
            raise ValueError(
                "Euler rate conversion is singular near pitch +/-90 degrees"
            )
        p, q, r = obs[3:6]
        rates = np.rad2deg(
            [
                p + np.tan(theta) * (q * np.sin(phi) + r * np.cos(phi)),
                (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta),
            ]
        )
        errors = np.rad2deg(obs[[6, 8]]) - [roll_ref_deg, heading_ref_deg]
        errors[1] = (errors[1] + 180) % 360 - 180
        previous: list[float | None] | np.ndarray = [None, None]
        if applied_output is not None:
            previous = np.asarray(applied_output, dtype=float)
            if previous.shape != (2,) or not np.isfinite(previous).all():
                raise ValueError(
                    "applied_output must contain two finite degree-valued positions"
                )
        if not np.isfinite([roll_ref_deg, heading_ref_deg]).all():
            raise ValueError("Angle references must be finite")
        return np.array(
            [
                pid.select_action(
                    0.0, error, measurement_rate=rate, applied_output=actual
                )
                for pid, error, rate, actual in zip(
                    self.controllers, errors, rates, previous
                )
            ]
        )
