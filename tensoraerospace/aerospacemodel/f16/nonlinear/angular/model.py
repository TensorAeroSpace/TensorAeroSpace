"""F-16 6-DoF angular nonlinear model — pure-numpy implementation.

State (14): [alpha, beta, wx, wy, wz, gamma, psi, theta,
             stab, dstab, ail, dail, dir, ddir]
Control (3): [stab_act, ail_act, dir_act]
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase

from .._actuators import project_actuators, surface_limits
from .._integrators import euler, integrate_events, rk4
from .dynamics import f16_ode_6dof
from .params import F16AngularParameters, _validate_air_density, default_parameters

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


class AngularF16(ModelBase):
    """F-16 with full 6-DoF angular dynamics (numpy version).

    Action (default, split_stab=False): [stab_act, ail_act, dir_act] (rad).
    Action when split_stab=True:        [stab_left, stab_right, ail, dir] (rad);
        differential = (L - R)/2 generates a roll moment via
        F16AngularParameters.delta_stab_roll_gain.

    With ``track_altitude=True``, ``V``, ``alpha`` and ``beta`` are air-relative.
    An external simulator computes wind and the global atmosphere, then supplies
    local air density in kg/m³ before a step. The model does not apply wind twice
    or integrate a full NED position. Its scalar altitude uses air-relative
    velocity and omits vertical wind, so an external world should set it before
    every step::

        model.set_air_relative_state(
            altitude_m=altitude,
            airspeed_mps=airspeed,
            alpha_rad=alpha,
            beta_rad=beta,
        )
        model.param.air_density_kg_m3 = local_density
        model.run_step(control, events=damage_events)
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output=None,
        t0: float = 0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
        split_stab: bool = False,
        track_altitude: bool = False,
        thrust_mode: Literal["constant", "control"] = "constant",
    ) -> None:
        x0_arr = self._prepare_initial_state(x0, dt, track_altitude, thrust_mode)
        n_state = x0_arr.size
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.split_stab = split_stab
        self.track_altitude = track_altitude
        self.thrust_mode = thrust_mode
        _list_state = [
            "alpha",
            "beta",
            "wx",
            "wy",
            "wz",
            "gamma",
            "psi",
            "theta",
            "stab",
            "dstab",
            "ail",
            "dail",
            "dir",
            "ddir",
        ]
        if track_altitude:
            _list_state.extend(["h", "V"])
        if split_stab:
            _control_list = ["stab_left", "stab_right", "ail", "dir"]
        else:
            _control_list = ["stab", "ail", "dir"]
        if thrust_mode == "control":
            _control_list = list(_control_list) + ["thrust"]
        self.action_space_length = len(_control_list)
        self.param: F16AngularParameters = default_parameters()

        # Damage subsystem (None = healthy aircraft, legacy behaviour)
        self.damage_state: Any = None
        self.damage_geometry: Any = None

        self.n_state = n_state
        self.x_history = [x0_arr.reshape(n_state, 1)]
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

    @staticmethod
    def _prepare_initial_state(x0, dt, track_altitude, thrust_mode):
        """Validate physical inputs before creating histories or model parameters."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if thrust_mode not in ("constant", "control"):
            raise ValueError("thrust_mode must be constant or control")
        x0_arr = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
        if not np.all(np.isfinite(x0_arr)):
            raise ValueError("x0 must be finite")
        n_state = 16 if track_altitude else 14
        if x0_arr.size == 14 and track_altitude:
            # Auto-pad with trim altitude and trim airspeed from params.
            params_default = default_parameters()
            x0_arr = np.append(x0_arr, [params_default.Oy, params_default.V])
        if x0_arr.size != n_state:
            raise ValueError(
                f"x0 must have {n_state} elements (track_altitude="
                f"{track_altitude}); got {x0_arr.size}"
            )
        if track_altitude and x0_arr[15] <= 0:
            raise ValueError("initial airspeed must be positive")
        return x0_arr

    def get_param(self) -> F16AngularParameters:
        """Return the live aircraft parameter object; mutations affect subsequent
        integration.
        """
        return self.param

    def set_param(self, new_param: F16AngularParameters) -> None:
        """Replace the aircraft parameter object used by subsequent integration steps."""
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Independent snapshot of the most recent state as a flat 1-D ndarray."""
        return np.array(self.x_history[-1], dtype=np.float64, copy=True).reshape(-1)

    def set_air_relative_state(
        self,
        *,
        altitude_m: float,
        airspeed_mps: float,
        alpha_rad: float,
        beta_rad: float,
    ) -> None:
        """Set altitude and air-relative velocity before the next integration step.

        ``V``, ``alpha`` and ``beta`` describe motion relative to the air. An
        external simulator must convert inertial velocity using its wind field;
        wind is not applied again inside the ODE. This method does not advance
        time or append a history sample: it replaces the most recent sample in
        ``x_history`` in place.

        Args:
            altitude_m: Altitude in metres.
            airspeed_mps: Positive air-relative speed in metres per second.
            alpha_rad: Air-relative angle of attack in radians.
            beta_rad: Air-relative sideslip angle in radians, with magnitude
                strictly less than pi/2.

        Raises:
            RuntimeError: If ``track_altitude`` is disabled.
            ValueError: If a value is nonfinite, airspeed is not positive, or
                sideslip reaches the singular range.
        """
        if not self.track_altitude:
            raise RuntimeError("set_air_relative_state requires track_altitude=True")
        values = (altitude_m, airspeed_mps, alpha_rad, beta_rad)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("air-relative state values must be finite")
        if airspeed_mps <= 0:
            raise ValueError("airspeed_mps must be positive")
        if not -np.pi / 2 < beta_rad < np.pi / 2:
            raise ValueError("beta_rad must satisfy |beta_rad| < pi/2")

        state = self.current_state
        state[14] = altitude_m
        state[15] = airspeed_mps
        state[0] = alpha_rad
        state[1] = beta_rad
        self.x_history[-1] = state.reshape(self.n_state, 1)

    def run_step(self, u: ArrayLike, *, events=()) -> np.ndarray:
        """Advance surface commands in radians and optional thrust in newtons.

        Timed event callbacks split the integration interval. Store the new state and
        applied controls, then return the configured output column.
        """
        _validate_air_density(self.param.air_density_kg_m3)
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                "Размерность управляющего вектора задана неверно."
                f" Текущее значение {u_arr.size}, не соответсвует {self.action_space_length}"
            )

        if not np.all(np.isfinite(u_arr)):
            raise ValueError("control input must be finite")

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * (self.time_step - 1)
        x_next = integrate_events(
            lambda x, t, dt: self._advance(x, u_arr, t, dt),
            x_prev,
            t_now,
            self.dt,
            events,
        )

        x_next_col = x_next.reshape(self.n_state, 1)
        self.x_history.append(x_next_col)
        _, recorded_control = self._prepare_control(u_arr)
        self.u_history.append(recorded_control.reshape(-1, 1).copy())
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col.copy()

    def _advance(self, x, u, t, dt):
        """Integrate one event-free interval and project actuator states onto physical
        limits.
        """
        control, _ = self._prepare_control(u)
        next_state = self._step_fn(f16_ode_6dof, x, control, t, dt, self.param)
        return project_actuators(
            next_state, self.param, ((8, "stab"), (10, "ail"), (12, "dir"))
        )

    def _prepare_control(self, u_arr):
        """Recompute applied commands after events, before split-stab merging."""
        # Thrust input handling
        if self.thrust_mode == "control":
            thrust_cmd = float(u_arr[-1])
            thrust_cmd = float(np.clip(thrust_cmd, 0.0, self.param.T_max_thrust))
            self.param.T_active = thrust_cmd
            u_arr = u_arr[:-1]  # drop thrust from u; rest is stab/ail/dir
        else:
            self.param.T_active = self.param.T_thrust

        # Saturate requested travel before efficiency loss, so an oversized
        # request cannot cancel a failure. Reapply stops after a jam command.
        limits = surface_limits(self.param, self.split_stab)
        u_arr = np.clip(u_arr, -limits, limits)

        # Apply control-surface failures BEFORE split-stab merging.
        # The failure layer mutates the raw user command (the 4-element
        # split form or the 3-element legacy form), then split-stab
        # merging proceeds with the failure-modified values.
        if self.damage_state is not None:
            from ..damage.controls import (
                ANGULAR_LEGACY_INDEX,
                ANGULAR_SPLIT_STAB_INDEX,
                apply_control_failures,
            )

            mapping = (
                ANGULAR_SPLIT_STAB_INDEX if self.split_stab else ANGULAR_LEGACY_INDEX
            )
            u_arr = apply_control_failures(u_arr, self.damage_state, mapping)
            u_arr = np.clip(u_arr, -limits, limits)

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

        # Keep the applied thrust channel aligned with control_list.
        recorded_control = (
            np.append(u_arr, self.param.T_active)
            if self.thrust_mode == "control"
            else u_arr
        )
        return u_legacy, recorded_control
