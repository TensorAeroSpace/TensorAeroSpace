"""Nonlinear quadrotor model — pure-numpy 6-DoF.

Wraps :func:`tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics.quadrotor_ode_6dof`
in the project's ``ModelBase`` interface, mirroring the F-16 nonlinear
model layout so adaptive critics (iADP / IM-GDHP / AIDI / ET-DHP /
AA-INDI) can plug in without bespoke glue.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from ._integrators import euler, rk4
from .dynamics import quadrotor_ode_6dof
from .params import QuadrotorParameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


_STATE_LIST = [
    "x_e", "y_e", "z_e",      # earth-frame position (NED), m
    "u_b", "v_b", "w_b",      # body-frame velocity, m/s
    "phi", "theta", "psi",    # Euler ZYX 321, rad
    "p", "q", "r",            # body-frame angular rates, rad/s
]
_CONTROL_LIST = ["T", "tau_x", "tau_y", "tau_z"]


class NonlinearQuadrotor(ModelBase):
    """Pure-numpy 6-DoF rigid-body quadrotor in NED frame.

    Action: ``[T, tau_x, tau_y, tau_z]`` — collective thrust (N) and
    body-frame torques (N·m). The model is "abstract" in the actuation:
    motor mixing / RPM allocation is *not* part of this class. Use the
    helper in ``utils.py`` (TODO) or the env layer for that.

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
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 12:
            raise ValueError(
                f"x0 must have 12 elements (see module docstring); got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        # Carry the local list before letting ModelBase reset it.
        self.action_space_length = len(_CONTROL_LIST)
        self.param: QuadrotorParameters = default_parameters()
        # Damage hooks (kept for API parity with F-16 — actual damage
        # subsystem for quadrotor will be added in a follow-up).
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
        """Most recent state as a flat 1-D ndarray (12 elements)."""
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    @property
    def hover_thrust(self) -> float:
        """Collective thrust that holds the vehicle stationary at level attitude."""
        return self.param.m * self.param.g

    # ---- simulation ----------------------------------------------------

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответствует {self.action_space_length}"
            )
        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(
            quadrotor_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param
        )

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
