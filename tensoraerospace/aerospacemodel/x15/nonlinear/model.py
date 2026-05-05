"""Pure-numpy nonlinear 6-DoF X-15 model with rocket-engine mass loss.

Mirrors the API of :class:`~tensoraerospace.aerospacemodel.b747.\
nonlinear.NonlinearB747` and the rest of the tensoraerospace
aerospaceplane family for downstream compatibility.

Distinguishing features:

* **13-state** vector — the standard 12-D rigid body plus a propellant
  mass channel ``m_prop`` that integrates downward at the XLR99 mass
  flow rate.
* **Hypersonic envelope** — Mach-tabulated derivatives valid from
  M = 0.4 to M = 6.7. Above M = 6.7 the model clamps the derivative
  table at the upper anchor (Walker/Wolowicz did not publish data
  beyond the X-15A-2 record).
* **Variable mass / inertia** — current values come from the
  configuration's full / empty inertia tensors interpolated by
  propellant fraction.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from ._integrators import euler, rk4
from .dynamics import x15_ode_6dof
from .initial import STATE_LIST
from .params import X15Configuration, X15Parameters, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

_CONTROL_LIST = ["de", "da", "dr", "dT"]


class NonlinearX15(ModelBase):
    """Nonlinear 6-DoF X-15 hypersonic research-aircraft model.

    Args:
        x0: Initial 13-element state. See :mod:`.initial` for layout.
        selected_state_output: Optional subset of state names to return
            from :meth:`run_step`. Defaults to the full state.
        t0: Initial time (s).
        dt: Integration step (s). Default 0.01.
        integrator: ``"euler"`` or ``"rk4"`` (default).
        config: Configuration (BASIC or A2).

    State layout: ``[u, v, w, p, q, r, φ, θ, ψ, x_e, y_e, z_e, m_prop]``.
    Control layout: ``[δ_e, δ_a, δ_r, δ_T]``.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output: list[str] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        config: X15Configuration = X15Configuration.BASIC,
    ) -> None:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != 13:
            raise ValueError(
                f"x0 must have 13 elements (see initial.STATE_LIST); "
                f"got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.action_space_length = len(_CONTROL_LIST)
        self.param: X15Parameters = default_parameters(config)
        self.damage_state: Any = None
        self.damage_geometry: Any = None
        self.x_history = [x0_arr.reshape(13, 1)]
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

    # ---- introspection ------------------------------------------------

    def get_param(self) -> X15Parameters:
        return self.param

    def set_param(self, new_param: X15Parameters) -> None:
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Most recent state as a flat 1-D ndarray (length 13)."""
        return np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)

    @property
    def altitude_ft(self) -> float:
        return float(-self.current_state[11])

    @property
    def airspeed_ft_s(self) -> float:
        s = self.current_state
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))

    @property
    def propellant_lb(self) -> float:
        """Remaining propellant in pounds."""
        return float(self.current_state[12])

    @property
    def mass_slug(self) -> float:
        """Current total mass in slugs (empty + remaining propellant)."""
        return self.param.current_mass_slug(self.propellant_lb)

    @property
    def engine_running(self) -> bool:
        """True iff the XLR99 has propellant left and the engine is not flamed out."""
        return self.propellant_lb > 0.0

    # ---- step ---------------------------------------------------------

    def run_step(self, u: ArrayLike) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                f"control vector size mismatch: got {u_arr.size}, "
                f"expected {self.action_space_length} ([δ_e, δ_a, δ_r, δ_T])"
            )

        # Propagate damage hooks (parity with B-747)
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * self.time_step
        x_next = self._step_fn(x15_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param)

        # Clamp propellant to non-negative (engine flames out below 0)
        if x_next[12] < 0.0:
            x_next[12] = 0.0

        x_next_col = x_next.reshape(13, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(u_arr.reshape(-1, 1))
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col
