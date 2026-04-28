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

    Action (default, split_stab=False): [stab_act, ail_act, dir_act] (rad).
    Action when split_stab=True:        [stab_left, stab_right, ail, dir] (rad);
        differential = (L - R)/2 generates a roll moment via
        F16AngularParameters.delta_stab_roll_gain.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
        split_stab: bool = False,
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 14:
            raise ValueError(f"x0 must have 14 elements; got {x0_arr.size}")
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.split_stab = split_stab
        _list_state = [
            "alpha", "beta", "wx", "wy", "wz",
            "gamma", "psi", "theta",
            "stab", "dstab", "ail", "dail", "dir", "ddir",
        ]
        if split_stab:
            _control_list = ["stab_left", "stab_right", "ail", "dir"]
        else:
            _control_list = ["stab", "ail", "dir"]
        self.action_space_length = len(_control_list)
        self.param: F16AngularParameters = default_parameters()

        # Damage subsystem (None = healthy aircraft, legacy behaviour)
        self.damage_state = None
        self.damage_geometry = None

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

        # Apply control-surface failures BEFORE split-stab merging.
        # The failure layer mutates the raw user command (the 4-element
        # split form or the 3-element legacy form), then split-stab
        # merging proceeds with the failure-modified values.
        if self.damage_state is not None:
            from ..damage.controls import (
                ANGULAR_LEGACY_INDEX, ANGULAR_SPLIT_STAB_INDEX,
                apply_control_failures,
            )
            mapping = (
                ANGULAR_SPLIT_STAB_INDEX if self.split_stab else ANGULAR_LEGACY_INDEX
            )
            u_arr = apply_control_failures(u_arr, self.damage_state, mapping)

        if self.split_stab:
            # u = [stab_left, stab_right, ail, dir]; convert to (stab_mean, ail, dir)
            # plus a differential delta carried via params for ODE roll-moment term.
            stab_mean = 0.5 * (u_arr[0] + u_arr[1])
            delta_stab = 0.5 * (u_arr[0] - u_arr[1])  # +ve delta = LWD (left up)
            u_legacy = np.array([stab_mean, u_arr[2], u_arr[3]], dtype=np.float64)
            self.param.delta_stab_cmd = float(delta_stab)
        else:
            u_legacy = u_arr
            self.param.delta_stab_cmd = 0.0

        # Damage hooks for ODE corrections (Phase 3)
        if self.damage_state is not None and self.damage_geometry is not None:
            self.param.damage_state = self.damage_state
            self.param.damage_geometry = self.damage_geometry
        else:
            self.param.damage_state = None
            self.param.damage_geometry = None

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(f16_ode_6dof, x_prev, u_legacy, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(14, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
