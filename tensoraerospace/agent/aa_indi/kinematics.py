"""Aircraft kinematics for AA-INDI (Atmaca 2026, Eqs. 19--30).

SI units; right-handed body axes forward/right/down; navigation axes NED;
Euler angles use the conventional yaw-pitch-roll (3-2-1) sequence.
Accelerometers measure specific force, not inertial acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def body_to_ned(attitude: np.ndarray) -> np.ndarray:
    phi, theta, psi = np.asarray(attitude, dtype=float)
    sp, st, sy = np.sin([phi, theta, psi])
    cp, ct, cy = np.cos([phi, theta, psi])
    return np.array(
        [
            [ct * cy, sp * st * cy - cp * sy, cp * st * cy + sp * sy],
            [ct * sy, sp * st * sy + cp * cy, cp * st * sy - sp * cy],
            [-st, sp * ct, cp * ct],
        ]
    )


def jacobian(function, x: np.ndarray) -> np.ndarray:
    """Central finite-difference linearization; relative perturbations in SI."""
    x = np.asarray(x, dtype=float)
    result = []
    for i, value in enumerate(x):
        dx = np.zeros_like(x)
        dx[i] = 1e-5 * max(1.0, abs(value))
        result.append((function(x + dx) - function(x - dx)) / (2 * dx[i]))
    return np.asarray(result).T


def aircraft_kinematics(
    state: np.ndarray,
    imu: np.ndarray,
    fault_estimate: np.ndarray,
    gravity: float = 9.80665,
) -> np.ndarray:
    """Non-exact model: retain each primary fault, compensate cross-couplings.

    This intentionally differs from subtracting all six estimated faults:
    the primary fault must still create observable drift for HOSM.
    """
    u, v, w, phi, theta, _ = state
    ax, ay, az, p, q, r = imu
    fp, fq, fr = fault_estimate[3:]
    cphi, ctheta = np.cos([phi, theta])
    sphi, stheta = np.sin([phi, theta])
    if abs(ctheta) < 1e-3:
        raise ValueError("Euler kinematics singular near pitch = +/-90 degrees")
    return np.array(
        [
            v * (r - fr) - w * (q - fq) + ax - gravity * stheta,
            -u * (r - fr) + w * (p - fp) + ay + gravity * ctheta * sphi,
            u * (q - fq) - v * (p - fp) + az + gravity * ctheta * cphi,
            p + (q - fq) * sphi * np.tan(theta) + (r - fr) * cphi * np.tan(theta),
            q * cphi - (r - fr) * sphi,
            (q - fq) * sphi / ctheta + r * cphi / ctheta,
        ]
    )


def observation_model(state: np.ndarray) -> np.ndarray:
    return np.concatenate([body_to_ned(state[3:]) @ state[:3], state[3:]])


def rk4(function, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = function(state)
    k2 = function(state + 0.5 * dt * k1)
    k3 = function(state + 0.5 * dt * k2)
    k4 = function(state + dt * k3)
    return np.asarray(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)


@dataclass
class FlightMeasurement:
    """One timestamp of independent navigation and IMU measurements.

    ``ground_velocity`` is NED ground velocity [m/s], independent of the
    possibly faulty IMU; ``attitude`` is independent 3-2-1 attitude [rad].
    Either can be None after initialization for a slower-rate sensor.
    ``surface_position`` [rad] describes the applied input over the preceding
    interval (zero-order-hold value or measured interval average).
    ``airspeed`` [m/s] is air-relative, not the magnitude of GPS ground speed.
    """

    time: float
    angular_rate: np.ndarray
    specific_force: np.ndarray
    ground_velocity: np.ndarray | None
    attitude: np.ndarray | None
    surface_position: np.ndarray
    airspeed: float
    density: float

    def __post_init__(self):
        for key in ("angular_rate", "specific_force", "ground_velocity", "attitude"):
            value = getattr(self, key)
            if value is None and key in ("ground_velocity", "attitude"):
                continue
            value = np.asarray(value, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite length-three vector")
            setattr(self, key, value.copy())
        self.surface_position = np.asarray(self.surface_position, dtype=float).copy()
        if (
            self.surface_position.ndim != 1
            or not np.isfinite(self.surface_position).all()
        ):
            raise ValueError("surface_position must be a finite vector")
        if (
            not np.isfinite(self.time)
            or not np.isfinite(self.airspeed)
            or self.airspeed <= 0
            or not np.isfinite(self.density)
            or self.density <= 0
        ):
            raise ValueError(
                "time must be finite; airspeed and density must be positive"
            )

    @classmethod
    def from_model(
        cls,
        model,
        *,
        applied_action=None,
        surface_indices=(0, 1, 2),
        state=None,
        time=None,
    ):
        """Generate ideal SI sensors from a nonlinear Boeing model.

        The model state is US-unit 12-state NED, its physical inputs are
        radians/radians/radians/normalized throttle. Uses the public
        ``dynamics`` method, so modeled faults affect the accelerometer.
        This ideal no-wind simulator adapter supplies independent navigation;
        hardware integrations should construct packets from actual sensors.
        Initial packets require an explicit trim ``applied_action``.
        """
        state = model.current_state if state is None else np.asarray(state, dtype=float)
        action = (
            model.applied_action
            if applied_action is None
            else np.asarray(applied_action, dtype=float)
        )
        time = model.current_time if time is None else float(time)
        indices = np.asarray(surface_indices)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices > 2)
            or len(set(indices.tolist())) != indices.size
        ):
            raise ValueError(
                "surface_indices must be unique control surface indices in 0..2"
            )
        derivative = model.dynamics(state, action, time=time)
        rotation = body_to_ned(state[6:9])
        force = (
            derivative[:3]
            + np.cross(state[3:6], state[:3])
            - rotation.T @ np.array([0.0, 0.0, model.param.g_ft_s2])
        ) * 0.3048
        return cls(
            time=time,
            angular_rate=state[3:6],
            specific_force=force,
            ground_velocity=rotation @ (state[:3] * 0.3048),
            attitude=state[6:9],
            surface_position=action[indices],
            airspeed=float(np.linalg.norm(state[:3]) * 0.3048),
            density=model.density_at(-state[11]),
        )

    @property
    def imu(self) -> np.ndarray:
        return np.concatenate([self.specific_force, self.angular_rate])
