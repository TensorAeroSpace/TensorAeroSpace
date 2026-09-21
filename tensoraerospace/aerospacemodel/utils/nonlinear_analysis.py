"""Analysis interface shared by the US-unit nonlinear Boeing models."""

from typing import Any, Callable

import numpy as np


class NonlinearAircraftAnalysis:
    """Public instantaneous dynamics, input feedback and local linearization.

    Implementations supply ``_rhs``, ``param``, state/input history and the
    model clock. Analysis never advances history or changes the fault schedule.
    State/input units are those of the underlying aircraft model.
    """

    t0: float
    dt: float
    time_step: int
    u_history: list[Any]
    param: Any
    damage_state: Any
    damage_geometry: Any
    _density: Callable[[float], float]
    _rhs: Callable[..., np.ndarray]

    @property
    def current_state(self) -> np.ndarray:
        """Return the latest state; implemented by the concrete aircraft model."""
        raise NotImplementedError

    @property
    def current_time(self) -> float:
        """Time of the latest state, in seconds."""
        return self.t0 + self.dt * (self.time_step - 1)

    @property
    def air_density_kg_m3(self) -> float:
        """Atmospheric density at the current altitude, in SI units."""
        return self._density(-self.current_state[11]) * 14.5939029 / 0.3048**3

    def density_at(self, altitude_ft: float) -> float:
        """SI density at a supplied altitude in model-native feet."""
        if not np.isfinite(altitude_ft):
            raise ValueError("altitude must be finite")
        return self._density(altitude_ft) * 14.5939029 / 0.3048**3

    @property
    def applied_action(self) -> np.ndarray:
        """Copy of the actual input over the last interval; unavailable at reset."""
        if not self.u_history:
            raise RuntimeError("No applied action yet; supply the initial trim input")
        return np.asarray(self.u_history[-1], dtype=float).ravel().copy()

    def _aerodynamic_action(self, action, time):
        return np.asarray(action, dtype=float).copy()

    def dynamics(self, state, action, *, time=None) -> np.ndarray:
        """Evaluate the instantaneous ODE without advancing the simulation.

        Includes the current damage state and scheduled B737 effectiveness.
        ``action`` is a physical four-channel input, not a normalized command.
        """
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        time = self.current_time if time is None else float(time)
        if state.shape != (12,) or action.shape != (4,):
            raise ValueError("Expected state shape (12,) and action shape (4,)")
        if (
            not np.isfinite(state).all()
            or not np.isfinite(action).all()
            or not np.isfinite(time)
        ):
            raise ValueError("State, action and time must be finite")
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry
        return self._rhs(
            state, self._aerodynamic_action(action, time), time, self.param
        )

    def linearize(self, state, action, *, time=None, perturbation=1e-5):
        """Return continuous-time A and B Jacobians by central differences.

        Uses the model's native state/input units and the current damage state.
        A healthy nominal prior must be calculated before damage is applied.
        This method does not supply a reference-generator model or a trim solver.
        """
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        if not np.isfinite(perturbation) or perturbation <= 0:
            raise ValueError("perturbation must be finite and positive")
        self.dynamics(state, action, time=time)  # Validate before perturbing.
        h = float(perturbation)
        A = np.column_stack(
            [
                (
                    self.dynamics(state + dx, action, time=time)
                    - self.dynamics(state - dx, action, time=time)
                )
                / (2 * h)
                for dx in h * np.eye(12)
            ]
        )
        B = np.column_stack(
            [
                (
                    self.dynamics(state, action + du, time=time)
                    - self.dynamics(state, action - du, time=time)
                )
                / (2 * h)
                for du in h * np.eye(4)
            ]
        )
        return A, B
