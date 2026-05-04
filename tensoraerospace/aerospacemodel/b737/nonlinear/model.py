"""Pure-numpy nonlinear 6-DoF Boeing 737 model.

API parity with :class:`tensoraerospace.aerospacemodel.b747.\
nonlinear.NonlinearB747`.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from ._integrators import euler, rk4
from .dynamics import b737_ode_6dof
from .initial import STATE_LIST
from .params import B737Configuration, B737Parameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

_CONTROL_LIST = ["de", "da", "dr", "dT"]


class NonlinearB737(ModelBase):
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
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
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

    def get_param(self) -> B737Parameters:
        return self.param

    def set_param(self, new_param: B737Parameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    @property
    def altitude_ft(self) -> float:
        return float(-self.current_state[11])

    @property
    def airspeed_ft_s(self) -> float:
        s = self.current_state
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                f"control vector size mismatch: got {u_arr.size}, "
                f"expected {self.action_space_length} ([δ_e, δ_a, δ_r, δ_T])"
            )
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(b737_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
