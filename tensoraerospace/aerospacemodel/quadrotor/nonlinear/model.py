"""Nonlinear quadrotor model — pure-numpy 6-DoF.

Wraps :func:`tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics.quadrotor_ode_6dof`
in the project's ``ModelBase`` interface, mirroring the F-16 nonlinear
model layout so adaptive critics (iADP / IM-GDHP / AIDI / ET-DHP /
AA-INDI) can plug in without bespoke glue.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from ._integrators import euler, rk4
from .dynamics import quadrotor_ode_6dof
from .params import QuadrotorParameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


_STATE_LIST = [
    "x_e",
    "y_e",
    "z_e",  # earth-frame position (NED), m
    "u_b",
    "v_b",
    "w_b",  # body-frame velocity, m/s
    "phi",
    "theta",
    "psi",  # Euler ZYX 321, rad
    "p",
    "q",
    "r",  # body-frame angular rates, rad/s
]
_CONTROL_LIST = ["T", "tau_x", "tau_y", "tau_z"]


class NonlinearQuadrotor(ModelBase):
    """Pure-numpy 6-DoF rigid-body quadrotor in NED frame.

    Action: ``[T, tau_x, tau_y, tau_z]`` — collective thrust (N) and
    body-frame torques (N·m). The model is "abstract" in the actuation:
    motor mixing / RPM allocation is *not* part of this class. Use the
    ``quadrotor.allocation`` module or the env layer for that.

    State: see module docstring of ``__init__.py`` for the 12-element
    layout.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
    ) -> None:
        x0_arr = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
        if x0_arr.size != 12:
            raise ValueError(
                f"x0 must have 12 elements (see module docstring); got {x0_arr.size}"
            )
        if not np.all(np.isfinite(x0_arr)):
            raise ValueError("x0 must contain only finite values")
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
            raise ValueError("dt must be finite and positive, and t0 finite")
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.action_space_length = len(_CONTROL_LIST)
        self.param: QuadrotorParameters = default_parameters()
        # Damage hooks for API parity. Rotor effectiveness is applied
        # by the environment through allocated control inputs.
        self.damage_state: Any = None
        self.damage_geometry: Any = None
        self.x_history = [x0_arr.reshape(12, 1)]
        self._initialize_selected_state_index(self.selected_state_output, _STATE_LIST)
        self.list_state = list(_STATE_LIST)
        self.control_list = list(_CONTROL_LIST)

        if integrator == "euler":
            self._step_fn = euler
        elif integrator == "rk4":
            self._step_fn = rk4
        else:
            raise ValueError(f"unknown integrator: {integrator!r}")
        self._integrator_name = integrator

    # ---- introspection -------------------------------------------------

    def get_param(self) -> QuadrotorParameters:
        return self.param

    def set_param(self, new_param: QuadrotorParameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Independent snapshot of the most recent state as a flat 1-D ndarray (12 elements)."""
        return np.array(self.x_history[-1], dtype=np.float64, copy=True).reshape(-1)

    @property
    def hover_thrust(self) -> float:
        """Collective thrust that holds the vehicle stationary at level attitude."""
        return self.param.m * self.param.g

    # ---- simulation ----------------------------------------------------

    def run_step(
        self,
        u: ArrayLike,
        *,
        control_segments: (
            Sequence[tuple[float, Callable[[float], np.ndarray]]] | None
        ) = None,
    ) -> np.ndarray:
        """Advance one sample, optionally splitting at control discontinuities.

        Each segment contains its duration and a control function of elapsed
        time within that segment. Durations must sum to ``dt``. The supplied
        ``u`` is the end-of-sample command recorded in history; a segment's
        right endpoint evaluates its own left limit before a discontinuity.
        """
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответствует {self.action_space_length}"
            )
        if not np.all(np.isfinite(u_arr)):
            raise ValueError("control must contain only finite values")
        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * (self.time_step - 1)
        if control_segments is None:
            x_next = self._step_fn(
                quadrotor_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param
            )
        else:
            x_next = self._integrate_segments(x_prev, t_now, control_segments)

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1).copy())
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col.copy()

    def _integrate_segments(self, state, time, segments) -> np.ndarray:
        durations = np.array([duration for duration, _ in segments])
        if (
            not np.all(np.isfinite(durations))
            or np.any(durations <= 0)
            or not np.isclose(durations.sum(), self.dt, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("control segment durations must be positive and sum to dt")
        for duration, control in segments:

            def rhs(x, _u, t, params):
                # Clamp only floating-point roundoff at an interval boundary.
                elapsed = np.clip(t - time, 0.0, duration)
                command = np.asarray(control(float(elapsed)), dtype=np.float64)
                if command.shape != (4,) or not np.all(np.isfinite(command)):
                    raise ValueError(
                        "segment control must be a finite 4-element vector"
                    )
                return quadrotor_ode_6dof(x, command, t, params)

            state = self._step_fn(rhs, state, np.zeros(4), time, duration, self.param)
            time += duration
        return np.asarray(state)
