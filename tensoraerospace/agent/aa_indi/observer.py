"""Independent-navigation OTSEKF/HOSM sensor-fault reconstruction.

Atmaca 2026 III.A; Atmaca 2025 Eqs. (9)--(51). The two-stage filter's
input-bias coordinate and accumulated physical state drift have different
units. HOSM differentiates the latter: the open kinematic trajectory minus
its navigation-corrected trajectory, not an already rate-valued bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .hosm import HOSMDifferentiator
from .kinematics import (
    FlightMeasurement,
    aircraft_kinematics,
    body_to_ned,
    jacobian,
    observation_model,
    rk4,
)
from .otse import OptimalTwoStageEKF


@dataclass
class ObserverConfig:
    """Sampling, noise and drift-reconstruction settings for OTSEKF–HOSM.

    Time is in seconds and gravity in m/s². IMU vectors order specific force then
    angular rate; navigation vectors order NED velocity then Euler angles. Standard
    deviations use the corresponding SI measurement units.
    """

    dt: float = 0.01
    gravity: float = 9.80665
    imu_std: np.ndarray = field(
        default_factory=lambda: np.array([6.9e-4] * 3 + [4.1e-6] * 3)
    )
    navigation_std: np.ndarray = field(
        default_factory=lambda: np.array([0.03] * 3 + [8.7e-5, 8.7e-5, 1.7e-4])
    )
    initial_bias_std: np.ndarray = field(
        default_factory=lambda: np.array([1.0] * 3 + [0.03] * 3)
    )
    bias_walk_std: np.ndarray = field(
        default_factory=lambda: np.array([0.1] * 3 + [0.005] * 3)
    )
    hosm_gains: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 100.0, 5.0, 0.1])
    )
    drift_scales: np.ndarray = field(
        default_factory=lambda: np.array([1.0] * 3 + [0.01] * 3)
    )

    def __post_init__(self):
        if (
            not np.isfinite(self.dt)
            or self.dt <= 0
            or not np.isfinite(self.gravity)
            or self.gravity <= 0
        ):
            raise ValueError("dt and gravity must be finite and positive")
        for name in (
            "imu_std",
            "navigation_std",
            "initial_bias_std",
            "bias_walk_std",
            "drift_scales",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if (
                value.shape != (6,)
                or not np.isfinite(value).all()
                or np.any(value <= 0)
            ):
                raise ValueError(f"{name} must contain six positive values")
            setattr(self, name, value.copy())
        self.hosm_gains = np.asarray(self.hosm_gains, dtype=float)
        HOSMDifferentiator(6, self.dt, self.hosm_gains)


class OTSEKFHOSMObserver:
    """Fuse independent NED velocity/attitude; reconstruct all six IMU faults.

    Covariances and HOSM tuning are explicit. The articles do not publish all
    filter tuning or source code, so these defaults are a reproducible setup,
    not a claim to reproduce the Flying-V experiment's hidden parameters.
    """

    def __init__(self, config: ObserverConfig):
        self.cfg = config
        self.hosm = HOSMDifferentiator(6, config.dt, config.hosm_gains)
        self.filter: OptimalTwoStageEKF | None = None
        self.faults = np.zeros(6)
        self.drift = np.zeros(6)
        self._free_state: np.ndarray | None = None
        self._previous: FlightMeasurement | None = None

    @property
    def state(self) -> np.ndarray:
        """Return estimated body velocity and Euler angles after navigation
        initialization.
        """
        if self.filter is None:
            raise RuntimeError(
                "observer has not received initial navigation measurements"
            )
        return self.filter.state

    def update(self, measurement: FlightMeasurement) -> np.ndarray:
        """Assimilate a sensor packet and return bias-corrected IMU measurements.

        The first packet requires attitude and independent ground velocity. Later
        timestamps must advance by exactly ``cfg.dt``. Updating also reconstructs sensor
        faults and advances the HOSM differentiator.
        """
        if self.filter is None:
            if measurement.attitude is None or measurement.ground_velocity is None:
                raise ValueError(
                    "initial attitude and independent ground velocity are required"
                )
            state = np.r_[
                body_to_ned(measurement.attitude).T @ measurement.ground_velocity,
                measurement.attitude,
            ]
            self.filter = OptimalTwoStageEKF(
                state,
                np.diag(self.cfg.navigation_std**2),
                np.diag(self.cfg.initial_bias_std**2),
            )
            self._free_state = state.copy()
            self._previous = measurement
            self.hosm.step(np.zeros(6))
            return measurement.imu.copy()
        previous = self._previous
        if previous is None:
            raise RuntimeError("observer is missing its previous measurement")
        dt = measurement.time - previous.time
        if not np.isclose(dt, self.cfg.dt, atol=1e-10, rtol=1e-7):
            raise ValueError("measurements must advance exactly one observer dt")
        prior = self.filter.state
        imu = previous.imu

        def rates(x):
            """Evaluate aircraft kinematics with the current IMU sample and
            reconstructed faults.
            """
            return aircraft_kinematics(x, imu, self.faults, self.cfg.gravity)

        nominal = rk4(rates, prior, dt)
        F = jacobian(rates, prior)
        G = -jacobian(
            lambda sample: aircraft_kinematics(
                prior, sample, self.faults, self.cfg.gravity
            ),
            imu,
        )
        # Exact zero-order-hold linearization, including singular F.
        augmented = np.zeros((12, 12))
        augmented[:6, :6] = F
        augmented[:6, 6:] = G
        transition = expm(augmented * dt)
        Phi, Gamma = transition[:6, :6], transition[:6, 6:]
        Q = Gamma @ np.diag(self.cfg.imu_std**2) @ Gamma.T
        self.filter.predict(
            nominal, Phi, Gamma, Q, np.diag(self.cfg.bias_walk_std**2) * dt
        )
        available: list[int] = []
        values: list[float] = []
        if measurement.ground_velocity is not None:
            available.extend(range(3))
            values.extend(measurement.ground_velocity)
        if measurement.attitude is not None:
            available.extend(range(3, 6))
            values.extend(measurement.attitude)
        if available:
            estimate = self.filter.state
            h = observation_model(estimate)[available]
            H = jacobian(observation_model, estimate)[available]
            R = np.diag(self.cfg.navigation_std[available] ** 2)
            self.filter.correct(
                np.asarray(values),
                h,
                H,
                R,
                tuple(i for i, axis in enumerate(available) if axis >= 3),
            )
        # Accumulated drift is in physical state units. Use corrected states
        # for the couplings but retain the primary faulty input in each rate,
        # as in the non-exact kinematic model (25)--(30).
        self._free_state += nominal - prior
        self.drift = self._free_state - self.filter.state
        drift_rate = (
            self.hosm.step(self.drift / self.cfg.drift_scales) * self.cfg.drift_scales
        )
        phi, theta = self.filter.state[3:5]
        primary_gain = np.array(
            [1.0, 1.0, 1.0, 1.0, np.cos(phi), np.cos(phi) / np.cos(theta)]
        )
        if np.min(np.abs(primary_gain)) < 1e-3:
            raise ValueError("fault reconstruction ill-conditioned at this attitude")
        self.faults = drift_rate / primary_gain
        self._previous = measurement
        return measurement.imu - self.faults
