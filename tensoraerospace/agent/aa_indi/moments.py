"""Physical moment reconstruction and per-axis VFF-RLS, Atmaca 2026.

Eqs. (7)--(16), (50)--(57): fit dimensionless moment coefficients to measured
surface positions. Inputs are absolute positions, not their increments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .otse import covariance
from .vff_rls import VFFRLSEstimator


@dataclass
class AircraftGeometry:
    inertia: np.ndarray
    area: float
    span: float
    chord: float

    def __post_init__(self):
        self.inertia = covariance(self.inertia, 3, "inertia")
        if np.linalg.eigvalsh(self.inertia).min() <= 0:
            raise ValueError("inertia must be positive definite")
        if any(
            not np.isfinite(v) or v <= 0 for v in (self.area, self.span, self.chord)
        ):
            raise ValueError("area, span and chord must be finite and positive")

    @classmethod
    def from_parameters(cls, params):
        """Convert B737/B747 US-unit geometry and full inertia tensor to SI."""
        scale = 14.5939029 * 0.3048**2
        inertia = np.array(
            [
                [params.Ix * scale, 0.0, -params.Ixz * scale],
                [0.0, params.Iy * scale, 0.0],
                [-params.Ixz * scale, 0.0, params.Iz * scale],
            ]
        )
        return cls(
            inertia,
            params.S_ft2 * 0.3048**2,
            params.b_ft * 0.3048,
            params.cbar_ft * 0.3048,
        )

    def moment_scale(self, density: float, airspeed: float) -> np.ndarray:
        if (
            not np.isfinite(density)
            or density <= 0
            or not np.isfinite(airspeed)
            or airspeed <= 0
        ):
            raise ValueError("density and airspeed must be finite and positive")
        return (
            0.5
            * density
            * airspeed**2
            * self.area
            * np.array([self.span, self.chord, self.span])
        )

    def coefficients(
        self,
        rate: np.ndarray,
        acceleration: np.ndarray,
        density: float,
        airspeed: float,
    ) -> np.ndarray:
        rate, acceleration = np.asarray(rate), np.asarray(acceleration)
        if (
            rate.shape != (3,)
            or acceleration.shape != (3,)
            or not np.isfinite(rate).all()
            or not np.isfinite(acceleration).all()
        ):
            raise ValueError(
                "rate and acceleration must be finite length-three vectors"
            )
        # Euler's rigid-body equation, including products of inertia.
        moment = self.inertia @ acceleration + np.cross(rate, self.inertia @ rate)
        return np.asarray(moment / self.moment_scale(density, airspeed))

    def effectiveness(
        self, derivatives: np.ndarray, density: float, airspeed: float
    ) -> np.ndarray:
        return np.linalg.solve(
            self.inertia, self.moment_scale(density, airspeed)[:, None] * derivatives
        )


class MomentIdentifier:
    """Three scalar-output recursions: independent forgetting per moment axis."""

    def __init__(
        self,
        nominal_derivatives: np.ndarray,
        *,
        sigma0: float = 15.0,
        forgetting_min: float = 0.25,
        covariance_init: float = 100.0,
    ) -> None:
        nominal = np.asarray(nominal_derivatives, dtype=float)
        if (
            nominal.ndim != 2
            or nominal.shape[0] != 3
            or nominal.shape[1] == 0
            or not np.isfinite(nominal).all()
        ):
            raise ValueError(
                "nominal_derivatives must be finite with shape (3, n_surface)"
            )
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        self.estimators = [
            VFFRLSEstimator(
                1,
                nominal.shape[1],
                forgetting_min=forgetting_min,
                forgetting_max=1.0,
                eps_sensitivity=np.sqrt(sigma0),
                cov_init=covariance_init,
                theta_init_scale=0.0,
            )
            for _ in range(3)
        ]
        for axis, estimator in enumerate(self.estimators):
            estimator.theta[:, 0] = nominal[axis]

    @property
    def derivatives(self) -> np.ndarray:
        return np.vstack([e.theta[:, 0] for e in self.estimators])

    def update(self, surfaces: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        surfaces, coefficients = np.asarray(surfaces), np.asarray(coefficients)
        if (
            surfaces.shape != (self.estimators[0].n_u,)
            or coefficients.shape != (3,)
            or not np.isfinite(surfaces).all()
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("finite surface and coefficient vectors required")
        return np.array(
            [
                e.update(surfaces, coefficients[i : i + 1])[0]
                for i, e in enumerate(self.estimators)
            ]
        )
