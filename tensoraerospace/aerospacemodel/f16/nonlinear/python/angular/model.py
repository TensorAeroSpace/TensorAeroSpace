"""F-16 6-DoF angular nonlinear model — pure-numpy implementation.

State (14): [alpha, beta, wx, wy, wz, gamma, psi, theta,
             stab, dstab, ail, dail, dir, ddir]
Control (3): [stab_act, ail_act, dir_act]
"""
from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from .._integrators import euler, rk4
from .dynamics import f16_ode_6dof
from .params import F16AngularParameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


class AngularF16(ModelBase):
    """F-16 with full 6-DoF angular dynamics (numpy version).

    Action: stab_act, ail_act, dir_act (rad).
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 14:
            raise ValueError(
                f"x0 must have 14 elements; got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        _list_state = [
            "alpha", "beta",
            "wx", "wy", "wz",
            "gamma", "psi", "theta",
            "stab", "dstab",
            "ail", "dail",
            "dir", "ddir",
        ]
        _control_list = ["stab", "ail", "dir"]
        self.action_space_length = len(_control_list)
        self.param: F16AngularParameters = default_parameters()
        self.x_history = [x0_arr.reshape(14, 1)]
        # NOTE: _initialize_selected_state_index resets self.list_state and
        # self.control_list to [] as a side effect (ModelBase behaviour).
        # We must therefore reassign them AFTER the call.
        self._initialize_selected_state_index(self.selected_state_output, _list_state)
        self.list_state = _list_state
        self.control_list = _control_list

        if integrator == "euler":
            self._step_fn = euler
        elif integrator == "rk4":
            self._step_fn = rk4
        else:
            raise ValueError(f"unknown integrator: {integrator!r}")
        self._integrator_name = integrator

    def get_param(self) -> F16AngularParameters:
        return self.param

    def set_param(self, new_param: F16AngularParameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Most recent state as a flat 1-D ndarray."""
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(14, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
