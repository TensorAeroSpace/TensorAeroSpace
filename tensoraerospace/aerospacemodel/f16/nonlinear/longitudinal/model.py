"""F-16 nonlinear longitudinal model — pure-numpy implementation.

State: [alpha, wz, stab, dstab]. Control: [stab_act].
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from .._integrators import euler, rk4
from .dynamics import f16_ode_long
from .params import F16LongParameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


class LongitudinalF16(ModelBase):
    """F-16 in isolated longitudinal channel.

    Action: stab_act (elevator command, rad).
    State: alpha, wz, stab, dstab (rad / rad·s⁻¹).
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
        if x0_arr.size != 4:
            raise ValueError(
                f"x0 must have 4 elements (alpha, wz, stab, dstab); got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        # ModelBase._initialize_selected_state_index has the side effect of
        # resetting self.list_state and self.control_list to []. Compute
        # them locally, pass to that method, then reassign so they survive.
        _list_state = ["alpha", "wz", "stab", "dstab"]
        _control_list = ["stab"]
        self.action_space_length = len(_control_list)
        self.param: F16LongParameters = default_parameters()
        # Damage subsystem (None = healthy aircraft, legacy behaviour)
        self.damage_state = None
        self.damage_geometry = None
        self.x_history = [x0_arr.reshape(4, 1)]
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

    def get_param(self) -> F16LongParameters:
        return self.param

    def set_param(self, new_param: F16LongParameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Most recent state as a flat 1-D ndarray (alpha, wz, stab, dstab)."""
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )
        # Damage hooks for ODE corrections (Phase 3 / Phase 7.2)
        if self.damage_state is not None and self.damage_geometry is not None:
            self.param.damage_state = self.damage_state
            self.param.damage_geometry = self.damage_geometry
        else:
            self.param.damage_state = None
            self.param.damage_geometry = None
        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_long, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(4, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
