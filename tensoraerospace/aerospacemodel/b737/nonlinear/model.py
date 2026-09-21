"""Pure-numpy nonlinear 6-DoF Boeing 737 model.

API parity with :class:`tensoraerospace.aerospacemodel.b747.\
nonlinear.NonlinearB747`.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.utils.nonlinear_analysis import (
    NonlinearAircraftAnalysis,
)

from ._integrators import euler, rk4
from .damage import ElevatorEffectiveness
from .dynamics import b737_ode_6dof
from .initial import STATE_LIST
from .params import (
    B737Configuration,
    B737Parameters,
    default_parameters,
    isa_density_slug_ft3,
)

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

_CONTROL_LIST = ["de", "da", "dr", "dT"]


class NonlinearB737(NonlinearAircraftAnalysis, ModelBase):
    """Nonlinear 6-DoF Boeing 737 model.

    Args:
        x0: Initial 12-element state.
        selected_state_output: Optional subset of state names.
        t0: Initial time (s).
        dt: Integration step (s). Default 0.01.
        integrator: ``"euler"`` or ``"rk4"`` (default).
        config: ``B737Configuration.B737_100`` (default) or ``.B737_800``.

    State: ``[u, v, w, p, q, r, φ, θ, ψ, x_e, y_e, z_e]``.
    Control: ``[δ_e, δ_a, δ_r, δ_T]``.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output: list[str] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        config: B737Configuration = B737Configuration.B737_100,
        *,
        elevator_fault: ElevatorEffectiveness | None = None,
        integration_substeps: int = 1,
    ) -> None:
        x0_arr = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
        if x0_arr.size != 12:
            raise ValueError(
                f"x0 must have 12 elements (see initial.py); got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.action_space_length = len(_CONTROL_LIST)
        self.param: B737Parameters = default_parameters(config)
        self.damage_state: Any = None
        self.damage_geometry: Any = None
        self.x_history = [x0_arr.reshape(12, 1)]
        self._initialize_selected_state_index(self.selected_state_output, STATE_LIST)
        self.list_state = list(STATE_LIST)
        self.control_list = list(_CONTROL_LIST)

        if integrator == "euler":
            self._step_fn = euler
        elif integrator == "rk4":
            self._step_fn = rk4
        else:
            raise ValueError(f"unknown integrator: {integrator!r}")
        self._integrator_name = integrator
        if elevator_fault is not None and not isinstance(
            elevator_fault, ElevatorEffectiveness
        ):
            raise TypeError("elevator_fault must be ElevatorEffectiveness or None")
        if (
            isinstance(integration_substeps, bool)
            or not isinstance(integration_substeps, (int, np.integer))
            or integration_substeps < 1
        ):
            raise ValueError("integration_substeps must be a positive integer")
        self.elevator_fault = elevator_fault
        self.integration_substeps = int(integration_substeps)

    _density = staticmethod(isa_density_slug_ft3)

    _rhs = staticmethod(b737_ode_6dof)

    def get_param(self) -> B737Parameters:
        """Return the live aircraft parameter object; mutations affect subsequent
        integration.
        """
        return self.param

    def set_param(self, new_param: B737Parameters) -> None:
        """Replace the aircraft parameter object used by subsequent integration steps."""
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Return an independent flat copy of the latest 12-component state."""
        return np.array(self.x_history[-1], dtype=np.float64, copy=True).reshape(-1)

    @property
    def altitude_ft(self) -> float:
        """Return altitude in feet, reversing the NED down-position sign."""
        return float(-self.current_state[11])

    @property
    def airspeed_ft_s(self) -> float:
        """Return the magnitude of body-axis velocity in feet per second."""
        s = self.current_state
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))

    def _aerodynamic_action(self, action, time):
        """Apply the scheduled elevator effectiveness to a copied native action."""
        if self.elevator_fault is None:
            return np.asarray(action, dtype=float).copy()
        return self.elevator_fault.apply(action, time)

    def run_step(self, u: ArrayLike) -> np.ndarray:
        """Integrate surface-radian and throttle commands over one sampling interval.

        Split integration at elevator-fault times and configured substeps. Store the
        transition and return selected states as a column vector.
        """
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                f"control vector size mismatch: got {u_arr.size}, "
                f"expected {self.action_space_length} ([δ_e, δ_a, δ_r, δ_T])"
            )
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * (self.time_step - 1)
        cuts = list(np.linspace(t_now, t_now + self.dt, self.integration_substeps + 1))
        if (
            self.elevator_fault is not None
            and t_now < self.elevator_fault.time < t_now + self.dt
        ):
            cuts.append(self.elevator_fault.time)
        x_next = x_prev
        cuts = sorted(set(cuts))
        for start, stop in zip(cuts[:-1], cuts[1:]):
            # Hold one effectiveness through every RK stage of each segment.
            effective = self._aerodynamic_action(u_arr, start)
            interval = (
                self.dt
                if self.elevator_fault is None and self.integration_substeps == 1
                else stop - start
            )
            x_next = self._step_fn(
                b737_ode_6dof, x_next, effective, start, interval, self.param
            )

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1).copy())
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col.copy()
