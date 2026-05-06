"""Pure-numpy nonlinear 6-DoF Skywalker X8 model (SI units)."""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from ._integrators import euler, rk4
from .dynamics import x8_ode_6dof
from .initial import STATE_LIST
from .params import SkywalkerX8Parameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

_CONTROL_LIST = ["de", "da", "dT"]


class NonlinearSkywalkerX8(ModelBase):
    """Nonlinear 6-DoF Skywalker X8 — flying-wing UAV (~3.4 kg, 2.1 m span).

    State (SI): ``[u, v, w, p, q, r, φ, θ, ψ, x_e, y_e, z_e]`` —
    velocities in m/s, angles in rad, position in m.
    Control: ``[δ_e, δ_a, δ_T]`` — collective elevon (rad),
    differential elevon (rad), throttle [0, 1].

    Note: there is **no rudder** — lateral-directional yaw control is
    via differential aileron only.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output: list[str] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 12:
            raise ValueError(f"x0 must have 12 elements; got {x0_arr.size}")
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.action_space_length = len(_CONTROL_LIST)
        self.param: SkywalkerX8Parameters = default_parameters()
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

    def get_param(self) -> SkywalkerX8Parameters:
        return self.param

    def set_param(self, new_param: SkywalkerX8Parameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    @property
    def altitude_m(self) -> float:
        return float(-self.current_state[11])

    @property
    def airspeed_m_s(self) -> float:
        s = self.current_state
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                f"control vector size mismatch: got {u_arr.size}, "
                f"expected {self.action_space_length} ([δ_e, δ_a, δ_T])"
            )
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(x8_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param)

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
