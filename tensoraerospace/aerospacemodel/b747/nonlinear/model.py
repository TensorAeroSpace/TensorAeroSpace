"""B-747 nonlinear 6-DoF model — pure-numpy implementation.

Mirrors the API of :class:`tensoraerospace.aerospacemodel.quadrotor.\
nonlinear.NonlinearQuadrotor` and
:class:`tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.\
LongitudinalF16` for downstream compatibility (PID/MPC/IHDP/ET-DHP all
work the same way).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.utils.nonlinear_analysis import (
    NonlinearAircraftAnalysis,
)

from ._integrators import euler, rk4
from .damage import B747DamageManager, EngineFailureEvent
from .damage.events import AnyDamageEvent
from .dynamics import b747_ode_6dof
from .initial import STATE_LIST
from .params import (
    B747Configuration,
    B747Parameters,
    default_parameters,
    isa_density_slug_ft3,
)

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

_CONTROL_LIST = ["de", "da", "dr", "dT"]


class NonlinearB747(NonlinearAircraftAnalysis, ModelBase):
    """Nonlinear 6-DoF Boeing 747-100 model.

    Args:
        x0: Initial 12-element state. See :mod:`.initial` for layout
            and helpers.
        selected_state_output: Optional subset of state names to return
            from :meth:`run_step`. Defaults to the full state.
        t0: Initial time (s).
        dt: Integration step (s). Default 0.01.
        integrator: Either ``"euler"`` or ``"rk4"``. RK4 recommended
            for accurate trim hold; Euler is faster.
        config: Aerodynamic configuration to use.

    State layout: ``[u, v, w, p, q, r, φ, θ, ψ, x_e, y_e, z_e]``.
    Control layout: ``[δ_e, δ_a, δ_r, δ_T]``.
    """

    def __init__(
        self,
        x0: ArrayLike,
        selected_state_output: list[str] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        config: B747Configuration = B747Configuration.NOMINAL,
        *,
        damage_profile=None,
        damage_event_callback=None,
    ) -> None:
        x0_arr = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
        if x0_arr.size != 12:
            raise ValueError(
                f"x0 must have 12 elements (see initial.py); got {x0_arr.size}"
            )
        super().__init__(x0_arr, selected_state_output, t0, dt)
        self.action_space_length = len(_CONTROL_LIST)
        self.param: B747Parameters = default_parameters(config)
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
        self.damage_manager = (
            None if damage_profile is None else B747DamageManager(damage_profile)
        )
        self.damage_event_callback = damage_event_callback
        self.damage_events_log: list[dict[str, Any]] = []
        self.last_damage_events: list[AnyDamageEvent] = []
        self._damage_time = -np.inf
        if self.damage_manager is not None:
            self.damage_state = self.damage_manager.state
            self._apply_damage_at(self.t0)

    # ---- introspection -------------------------------------------------

    _density = staticmethod(isa_density_slug_ft3)

    _rhs = staticmethod(b747_ode_6dof)

    def get_param(self) -> B747Parameters:
        """Return the live aircraft parameter object; mutations affect subsequent
        integration.
        """
        return self.param

    def set_param(self, new_param: B747Parameters) -> None:
        """Replace the aircraft parameter object used by subsequent integration steps."""
        self.param = new_param

    @property
    def current_state(self) -> np.ndarray:
        """Independent snapshot of the most recent state as a flat 1-D ndarray."""
        return np.array(self.x_history[-1], dtype=np.float64, copy=True).reshape(-1)

    @property
    def altitude_ft(self) -> float:
        """Convenience accessor: altitude is ``-z_e`` (NED)."""
        return float(-self.current_state[11])

    @property
    def airspeed_ft_s(self) -> float:
        """Convenience accessor: ‖V_body‖."""
        s = self.current_state
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))

    @staticmethod
    def lateral_state(
        obs,
        integrals=None,
        *,
        roll_ref_deg=0.0,
        heading_ref_deg=0.0,
        integral_scale=0.1,
    ):
        """Return beta/p/r/roll/heading in degrees and scaled angle integrals.

        ``obs`` uses native Boeing units; integrals use degree-seconds.
        References are degrees. Heading error is wrapped to [-180, 180).
        """
        obs = np.asarray(obs, dtype=float)
        if (
            obs.shape != (12,)
            or not np.isfinite(obs).all()
            or np.linalg.norm(obs[:3]) <= 0
        ):
            raise ValueError("Expected finite 12-state observation with nonzero speed")
        if (
            not np.isfinite([roll_ref_deg, heading_ref_deg, integral_scale]).all()
            or integral_scale <= 0
        ):
            raise ValueError("References must be finite and integral_scale positive")
        if integrals is None:
            integrals = np.zeros(2)
        integrals = np.asarray(integrals, dtype=float)
        if integrals.shape != (2,) or not np.isfinite(integrals).all():
            raise ValueError("Expected two finite degree-second integrals")
        beta = np.rad2deg(np.arcsin(np.clip(obs[1] / np.linalg.norm(obs[:3]), -1, 1)))
        angles = np.rad2deg(obs[[3, 5, 6, 8]])
        angles[2] -= roll_ref_deg
        angles[3] -= heading_ref_deg
        angles[-1] = (angles[-1] + 180) % 360 - 180
        return np.r_[beta, angles, integral_scale * np.asarray(integrals)]

    def lateral_transition(
        self, state, action, trim_action, *, dt=None, integral_scale=0.1
    ):
        """Local lateral RK transition in degrees, including scaled integrals.

        State order: beta, p, r, roll, heading, scaled roll/heading integrals.
        Action order: aileron/rudder in degrees. The model's current state
        supplies the nominal speed, pitch and altitude; no model history moves.
        """
        state, action = np.asarray(state, dtype=float), np.asarray(action, dtype=float)
        if (
            state.shape != (7,)
            or action.shape != (2,)
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("Expected finite lateral state (7,) and input (2,)")
        dt = self.dt if dt is None else float(dt)
        if (
            not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(integral_scale)
            or integral_scale <= 0
        ):
            raise ValueError("dt and integral_scale must be positive")
        obs = self.current_state
        speed = np.linalg.norm(obs[:3])
        obs[1] = speed * np.sin(np.deg2rad(state[0]))
        obs[:3] *= speed / np.linalg.norm(obs[:3])
        obs[[3, 5, 6, 8]] = np.deg2rad(state[1:5])
        control = np.array(trim_action, dtype=float, copy=True)
        control[1:3] = np.deg2rad(action)
        nxt = self._step_fn(
            lambda x, u, t, params: self.dynamics(x, u, time=t),
            obs,
            control,
            self.current_time,
            dt,
            self.param,
        )
        integrals = state[5:] / integral_scale + dt * np.rad2deg(nxt[[6, 8]])
        return self.lateral_state(nxt, integrals, integral_scale=integral_scale)

    def lateral_linearization(self, trim_action, *, dt=None, integral_scale=0.1):
        """Discrete A/B for the seven-state degree-based lateral transition."""
        h = 1e-3
        zero = np.zeros(7)

        def transition(x, u):
            """Evaluate the lateral discrete transition at the fixed longitudinal trim
            action.
            """
            return self.lateral_transition(
                x, u, trim_action, dt=dt, integral_scale=integral_scale
            )

        A = np.column_stack(
            [
                (transition(h * x, np.zeros(2)) - transition(-h * x, np.zeros(2)))
                / (2 * h)
                for x in np.eye(7)
            ]
        )
        B = np.column_stack(
            [
                (transition(zero, h * u) - transition(zero, -h * u)) / (2 * h)
                for u in np.eye(2)
            ]
        )
        return A, B

    def _apply_damage_at(self, time):
        """Apply newly reached damage events and record them before the next interval."""
        if self.damage_manager is None:
            return
        # Boundary tolerance only resolves floating-point representation of a
        # scheduled time. Endpoint events are not applied to a preceding interval.
        boundary = float(time) + 1e-12
        events = self.damage_manager.update(boundary, self._damage_time, 0.0)
        self._damage_time = boundary
        for event in events:
            self.last_damage_events.append(event)
            record = {
                "time": float(event.trigger_time),
                "label": event.label or type(event).__name__,
                "kind": type(event).__name__,
            }
            if isinstance(event, EngineFailureEvent):
                record.update(
                    engine_id=event.engine_id, thrust_fraction=event.thrust_fraction
                )
            self.damage_events_log.append(record)
            if self.damage_event_callback is not None:
                self.damage_event_callback(event, self.damage_state)

    # ---- step ----------------------------------------------------------

    def run_step(self, u: ArrayLike) -> np.ndarray:
        """Integrate surface-radian and throttle commands with causal damage timing.

        Append state and input histories, update applied-action telemetry, and return
        the selected output-state column or the full state column.
        """
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != self.action_space_length:
            raise ValueError(
                f"control vector size mismatch: got {u_arr.size}, "
                f"expected {self.action_space_length} ([δ_e, δ_a, δ_r, δ_T])"
            )
        # Propagate damage state to params so aero / engine modules can
        # read engines_mu and flap_jam_config on every ODE evaluation.
        # damage_geometry is optional (parity with the F-16 hook for
        # missing-section scenarios; B-747 doesn't use it yet).
        self.param.damage_state = self.damage_state
        self.param.damage_geometry = self.damage_geometry

        x_prev = np.asarray(self.x_history[-1], dtype=np.float64).reshape(-1)
        t_now = self.t0 + self.dt * (self.time_step - 1)
        self.last_damage_events = []
        if self.damage_manager is None:
            x_next = self._step_fn(
                b747_ode_6dof, x_prev, u_arr, t_now, self.dt, self.param
            )
            applied = u_arr
        else:
            stop_time = t_now + self.dt
            events = self.damage_manager.scheduled_events
            cuts = (
                [t_now]
                + sorted(
                    set(
                        float(event.trigger_time)
                        for event in events
                        if t_now + 1e-12 < event.trigger_time < stop_time - 1e-12
                    )
                )
                + [stop_time]
            )
            x_next = x_prev
            applied = np.zeros(4)
            for start, stop in zip(cuts[:-1], cuts[1:]):
                self._apply_damage_at(start)
                effective = self.damage_state.apply(u_arr)
                interval = self.dt if len(cuts) == 2 else stop - start
                x_next = self._step_fn(
                    b747_ode_6dof, x_next, effective, start, interval, self.param
                )
                applied += effective * (interval / self.dt)
                self.damage_state.step_decay(interval)

        x_next_col = x_next.reshape(12, 1)
        self.x_history.append(x_next_col)
        self.u_history.append(applied.reshape(-1, 1).copy())
        self.time_step += 1

        if self.selected_state_output:
            return x_next_col[self.selected_state_index]
        return x_next_col.copy()
