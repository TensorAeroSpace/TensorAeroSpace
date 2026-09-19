"""AA-INDI with physical moment identification and OTSEKF-HOSM measurements.

Atmaca et al., AIAA 2026-1743. ``predict_acceleration`` is the published inner
loop boundary (virtual angular acceleration supplied by an outer controller).
``predict`` adds a configurable proportional rate loop as an application
adapter; it is not a reproduction of Flying-V's C*/sideslip guidance gains.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kinematics import FlightMeasurement
from .moments import AircraftGeometry, MomentIdentifier
from .observer import ObserverConfig, OTSEKFHOSMObserver


@dataclass
class AAINDIConfig:
    geometry: AircraftGeometry
    nominal_derivatives: np.ndarray
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    sigma0: float = 15.0
    enable_sensor_correction: bool = True
    forgetting_min: float = 0.25
    covariance_init: float = 100.0
    acceleration_cutoff_hz: float = 10.0
    rate_feedback: np.ndarray = field(default_factory=lambda: np.full(3, 4.0))
    magnitude_limit: float = np.deg2rad(25.0)
    rate_limit: float = np.deg2rad(60.0)
    pinv_rcond: float = 1e-8

    def __post_init__(self):
        self.nominal_derivatives = np.asarray(
            self.nominal_derivatives, dtype=float
        ).copy()
        MomentIdentifier(
            self.nominal_derivatives,
            sigma0=self.sigma0,
            forgetting_min=self.forgetting_min,
            covariance_init=self.covariance_init,
        )
        self.rate_feedback = np.asarray(self.rate_feedback, dtype=float).copy()
        if (
            self.rate_feedback.shape != (3,)
            or not np.isfinite(self.rate_feedback).all()
            or np.any(self.rate_feedback < 0)
        ):
            raise ValueError(
                "rate_feedback must contain three finite nonnegative gains"
            )
        for name in (
            "acceleration_cutoff_hz",
            "magnitude_limit",
            "rate_limit",
            "pinv_rcond",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class AAINDIAgent:
    """Published inner-loop architecture with explicit physical sensor inputs.

    Online adaptation and fault reconstruction remain enabled throughout an
    episode; the controller receives no fault time or true effectiveness.
    Angles, rates, accelerations and surface positions use SI units throughout.
    """

    def __init__(self, config: AAINDIConfig):
        self.cfg = config
        self.n_state = 3
        self.n_control = config.nominal_derivatives.shape[1]
        self.identifier = MomentIdentifier(
            config.nominal_derivatives,
            sigma0=config.sigma0,
            forgetting_min=config.forgetting_min,
            covariance_init=config.covariance_init,
        )
        self.reset()

    def reset(self) -> None:
        """Clear sensor and command history, retaining identified derivatives."""
        self.observer = OTSEKFHOSMObserver(self.cfg.observer)
        self._measurement: FlightMeasurement | None = None
        self._corrected_rate: np.ndarray | None = None
        self._acceleration = np.zeros(3)
        self._surfaces = np.zeros(self.n_control)
        self._coefficients = np.zeros(3)
        self._moment_initialized = False
        self._last_command: np.ndarray | None = None
        self._metrics = {"moment_residual_norm": 0.0, "fault_norm": 0.0}

    @property
    def dt(self) -> float:
        return self.cfg.observer.dt

    @property
    def G(self) -> np.ndarray:
        if self._measurement is None:
            raise RuntimeError("airspeed and density measurements are required for G")
        return self.cfg.geometry.effectiveness(
            self.identifier.derivatives,
            self._measurement.density,
            self._measurement.airspeed,
        )

    def _observe(self, measurement: FlightMeasurement) -> None:
        if not isinstance(measurement, FlightMeasurement):
            raise TypeError(
                "paper AA-INDI requires FlightMeasurement, including independent navigation"
            )
        if measurement.surface_position.shape != (self.n_control,):
            raise ValueError("surface_position has the wrong number of channels")
        if self._measurement is not None and measurement.time == self._measurement.time:
            # predict after learn must reuse the identical timestamp sample.
            for key in (
                "angular_rate",
                "specific_force",
                "ground_velocity",
                "attitude",
                "surface_position",
                "airspeed",
                "density",
            ):
                old, new = getattr(self._measurement, key), getattr(measurement, key)
                if (old is None) != (new is None) or (
                    old is not None and not np.array_equal(old, new)
                ):
                    raise ValueError(
                        "a timestamp cannot carry two different measurements"
                    )
            return
        corrected = self.observer.update(measurement)
        rate = (
            corrected[3:]
            if self.cfg.enable_sensor_correction
            else measurement.angular_rate
        )
        if self._corrected_rate is None:
            self._surfaces = measurement.surface_position.copy()
        else:
            acceleration = (rate - self._corrected_rate) / self.dt
            midpoint_rate = 0.5 * (rate + self._corrected_rate)
            coefficients = self.cfg.geometry.coefficients(
                midpoint_rate, acceleration, measurement.density, measurement.airspeed
            )
            # Common linear low-pass on the identified input/output pair;
            # do not fit unfiltered surfaces to delayed moment measurements.
            alpha = min(1.0, 2 * np.pi * self.cfg.acceleration_cutoff_hz * self.dt)
            if not self._moment_initialized:
                # Prime both sides from the first measured interval. Assuming
                # zero previous moment with a nonzero trim surface would teach
                # an artificial loss of effectiveness on the first update.
                self._acceleration = acceleration.copy()
                self._surfaces = measurement.surface_position.copy()
                self._coefficients = coefficients.copy()
                self._moment_initialized = True
            else:
                self._acceleration += alpha * (acceleration - self._acceleration)
                self._surfaces += alpha * (
                    measurement.surface_position - self._surfaces
                )
                self._coefficients += alpha * (coefficients - self._coefficients)
            residual = self.identifier.update(self._surfaces, self._coefficients)
            self._metrics["moment_residual_norm"] = float(np.linalg.norm(residual))
        self._corrected_rate = rate.copy()
        self._measurement = measurement
        self._metrics["fault_norm"] = float(np.linalg.norm(self.observer.faults))

    def predict_acceleration(
        self, measurement: FlightMeasurement, desired_acceleration: np.ndarray
    ) -> np.ndarray:
        """Paper Eq. (9); the outer controller supplies virtual control nu."""
        desired = np.asarray(desired_acceleration, dtype=float)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError(
                "desired_acceleration must be a finite length-three vector"
            )
        self._observe(measurement)
        increment = np.linalg.pinv(self.G, rcond=self.cfg.pinv_rcond) @ (
            desired - self._acceleration
        )
        command = self._surfaces + increment
        previous = measurement.surface_position
        command = np.clip(
            command,
            previous - self.cfg.rate_limit * self.dt,
            previous + self.cfg.rate_limit * self.dt,
        )
        command = np.clip(command, -self.cfg.magnitude_limit, self.cfg.magnitude_limit)
        self._last_command = command.copy()
        return np.asarray(command)

    def predict(
        self,
        measurement: FlightMeasurement,
        reference: np.ndarray,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Rate-tracking adapter; reference is a vector or a (3,T) schedule."""
        del deterministic
        reference = np.asarray(reference, dtype=float)
        if reference.ndim == 2 and reference.shape[0] == 3 and reference.shape[1]:
            reference = reference[:, int(np.clip(time_step, 0, reference.shape[1] - 1))]
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError("reference must be finite with shape (3,) or (3,T)")
        self._observe(measurement)
        desired = self.cfg.rate_feedback * (reference - self._corrected_rate)
        return self.predict_acceleration(measurement, desired)

    def learn(
        self,
        next_measurement: FlightMeasurement,
        reference=None,
        time_step: int = 0,
        *,
        applied_action=None,
    ) -> dict:
        """Assimilate the next sensors and applied surfaces once per step."""
        del reference, time_step
        if self._last_command is None or self._measurement is None:
            raise RuntimeError("learn must follow predict and a plant transition")
        if applied_action is not None and not np.array_equal(
            np.asarray(applied_action), next_measurement.surface_position
        ):
            raise ValueError(
                "applied_action must match FlightMeasurement.surface_position"
            )
        if next_measurement.time <= self._measurement.time:
            raise ValueError("learn requires the next timestamp")
        self._observe(next_measurement)
        self._last_command = None
        return dict(
            self._metrics,
            G_norm=float(np.linalg.norm(self.G)),
            vff_min=float(min(e.last_lambda for e in self.identifier.estimators)),
        )

    def save(self, path: str | Path) -> str:
        """Save a numeric JSON checkpoint including the complete observer state."""
        folder = Path(path) / f"{datetime.datetime.now():%b%d_%H-%M-%S}_AAINDIAgent"
        folder.mkdir(parents=True, exist_ok=True)
        f = self.observer.filter
        state = dict(
            config=dataclasses.asdict(self.cfg),
            measurement=(
                None
                if self._measurement is None
                else dataclasses.asdict(self._measurement)
            ),
            corrected_rate=self._corrected_rate,
            acceleration=self._acceleration,
            surfaces=self._surfaces,
            coefficients=self._coefficients,
            moment_initialized=self._moment_initialized,
            last_command=self._last_command,
            metrics=self._metrics,
            faults=self.observer.faults,
            drift=self.observer.drift,
            free_state=self.observer._free_state,
            hosm=self.observer.hosm.z,
            hosm_initialized=self.observer.hosm.initialized,
            filter=(
                None
                if f is None
                else {
                    k: getattr(f, k) for k in ("x_bar", "P_bar", "P_bias", "bias", "V")
                }
            ),
            estimators=[
                dict(
                    theta=e.theta,
                    P=e.P,
                    last_lambda=e.last_lambda,
                    num_updates=e.num_updates,
                    last_prediction_error=e.last_prediction_error,
                )
                for e in self.identifier.estimators
            ],
        )

        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        (folder / "paper_aaindi.json").write_text(
            json.dumps(state, default=encode, allow_nan=False)
        )
        return str(folder)

    @classmethod
    def from_pretrained(cls, folder: str | Path) -> "AAINDIAgent":
        """Load a local checkpoint without executable/pickled objects."""
        from .otse import OptimalTwoStageEKF

        state = json.loads((Path(folder) / "paper_aaindi.json").read_text())
        config = state["config"]
        config["geometry"] = AircraftGeometry(**config["geometry"])
        config["observer"] = ObserverConfig(**config["observer"])
        agent = cls(AAINDIConfig(**config))
        for name in (
            "corrected_rate",
            "acceleration",
            "surfaces",
            "coefficients",
            "last_command",
        ):
            value = state[name]
            setattr(
                agent,
                "_" + name,
                None if value is None else np.asarray(value, dtype=float),
            )
        agent._metrics = state["metrics"]
        agent._moment_initialized = state["moment_initialized"]
        agent._measurement = (
            None
            if state["measurement"] is None
            else FlightMeasurement(**state["measurement"])
        )
        o = agent.observer
        o._previous = agent._measurement
        for name in ("faults", "drift", "free_state"):
            value = state[name]
            setattr(
                o,
                "_" + name if name == "free_state" else name,
                None if value is None else np.asarray(value, dtype=float),
            )
        o.hosm.z = np.asarray(state["hosm"], dtype=float)
        o.hosm.initialized = state["hosm_initialized"]
        if state["filter"] is not None:
            fstate = {k: np.asarray(v, dtype=float) for k, v in state["filter"].items()}
            o.filter = OptimalTwoStageEKF(
                fstate["x_bar"], fstate["P_bar"], fstate["P_bias"]
            )
            for key, value in fstate.items():
                setattr(o.filter, key, value)
        for e, saved in zip(agent.identifier.estimators, state["estimators"]):
            for key, value in saved.items():
                setattr(
                    e,
                    key,
                    (
                        np.asarray(value, dtype=float)
                        if key in ("theta", "P", "last_prediction_error")
                        and value is not None
                        else value
                    ),
                )
        return agent
