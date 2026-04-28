"""Gymnasium env wrapping the pure-numpy nonlinear F-16 6-DoF angular model.

State vector exposed by the underlying model::

    [alpha, beta, wx, wy, wz, gamma, psi, theta,
     stab, dstab, ail, dail, dir, ddir]

Control vector::

    [stab_act, ail_act, dir_act]   (commanded deflections, rad)

The env additionally tracks per-step inertial position (reconstructed
via constant-airspeed kinematics) and a configurable subset of state
channels for chart display under ``env.unwrapped.{position_history,
attitude_history, time_history, chart_history}``.

Render API (Gymnasium-standard ``render_mode`` kwarg) is added in a
separate commit.
"""

from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager, DamageProfile, load_f16_geometry,
)
from tensoraerospace.visualization.kinematics import (
    _body_to_inertial_matrix,
    _body_velocity,
)

MODEL_STATE_ORDER = [
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

DEFAULT_CHART_STATES = ("alpha", "beta", "wx", "wy", "wz", "stab", "ail", "dir")


class NonlinearAngularF16(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear F-16 6-DoF angular model.

    Args:
        initial_state: 14-element initial state in radians/rad-per-s.
        number_time_steps: Episode length cap (steps).
        dt: Discretisation step (s). Defaults to 0.01.
        integrator: ``"euler"`` or ``"rk4"`` (default).
        airspeed: True airspeed (m/s) used to reconstruct inertial position.
            Constant for the episode.
        render_mode: One of ``None``, ``"human"``, ``"rgb_array"``, ``"live"``.
            Render is wired in a follow-up commit.
        chart_states: Names of state channels to track for the chart strip.
        trail_length: Optional trail clipping (None = full trail).

    Action units: degrees, range ``[-25, 25]``, converted to radians before
    being passed to the underlying numpy model. This matches the linear /
    longitudinal convention so existing controllers transfer.
    """

    metadata = {"render_modes": ["human", "rgb_array", "live"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        number_time_steps: int,
        *,
        dt: float = 0.01,
        integrator: str = "rk4",
        airspeed: float = 200.0,
        render_mode: Optional[str] = None,
        chart_states: Sequence[str] = DEFAULT_CHART_STATES,
        trail_length: Optional[int] = None,
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback=None,
        split_stab: bool = False,
    ) -> None:
        super().__init__()
        if initial_state.shape != (14,):
            raise ValueError(
                f"initial_state must be 14-element; got {initial_state.shape}"
            )
        for name in chart_states:
            if name not in MODEL_STATE_ORDER:
                raise ValueError(
                    f"chart_states entry {name!r} not in MODEL_STATE_ORDER"
                )

        self.initial_state = np.asarray(initial_state, dtype=np.float64).copy()
        self.number_time_steps = int(number_time_steps)
        self.dt = float(dt)
        self.integrator = integrator
        self.airspeed = float(airspeed)
        self.render_mode = render_mode
        self.chart_states = tuple(chart_states)
        self.trail_length = trail_length
        self.split_stab = split_stab
        self.damage_profile = damage_profile
        self.damage_observable = damage_observable
        self.damage_event_callback = damage_event_callback

        self.max_action_value = 25.0  # deg

        action_shape = (4,) if split_stab else (3,)
        self.action_space = spaces.Box(
            low=-self.max_action_value, high=self.max_action_value,
            shape=action_shape, dtype=np.float64,
        )

        # Observation: 14 model states + optional damage state vector
        obs_size = 14
        if damage_observable:
            geo = load_f16_geometry()
            obs_size += len(geo.section_names())  # section_loss vector
            obs_size += 1  # engine.thrust_factor
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64,
        )
        self._geo_for_obs = (
            load_f16_geometry() if (damage_observable or damage_profile) else None
        )
        self.damage_manager: Optional[DamageManager] = None

        # Filled in reset()
        self.model: AngularF16 | None = None
        self._step_index: int = 0
        self.position_history = np.zeros((0, 3))
        self.attitude_history = np.zeros((0, 3))
        self.time_history = np.zeros((0,))
        self.chart_history: dict[str, np.ndarray] = {}
        self._live_renderer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.model = AngularF16(
            x0=self.initial_state,
            t0=0,
            dt=self.dt,
            integrator=self.integrator,
            split_stab=self.split_stab,
        )
        self._step_index = 0

        # Damage manager
        if self.damage_profile is not None or self.damage_observable:
            geo = self._geo_for_obs
            self.damage_manager = DamageManager(
                geometry=geo, params=self.model.param,
                profile=(self.damage_profile or DamageProfile(events=[])),
            )
            if options and "damage_profile" in options:
                self.damage_manager.set_profile(options["damage_profile"])
            self.damage_manager.reset(seed=seed)
            self.model.damage_state = self.damage_manager.state
            self.model.damage_geometry = geo

        self.position_history = np.zeros((1, 3), dtype=np.float64)
        self.attitude_history = self._extract_attitude(self.initial_state).reshape(1, 3)
        self.time_history = np.zeros((1,), dtype=np.float64)
        self.chart_history = {
            name: np.array([self.initial_state[MODEL_STATE_ORDER.index(name)]])
            for name in self.chart_states
        }
        self._live_renderer = None

        obs = self._build_observation(self.initial_state)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = (4,) if self.split_stab else (3,)
        if action.shape != expected:
            raise ValueError(f"action must be {expected}; got {action.shape}")
        action_clipped = np.clip(action, -self.max_action_value, self.max_action_value)
        u_rad = np.deg2rad(action_clipped)

        # Time bookkeeping (BEFORE stepping the model)
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt

        # Damage events
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            triggered = self.damage_manager.update(t_now, t_prev)
            for ev in triggered:
                if self.damage_event_callback:
                    self.damage_event_callback(ev, self.damage_manager.state)
                triggered_labels.append(ev.label or ev.event_type)

        assert self.model is not None
        self.model.run_step(u_rad)
        next_state = self.model.current_state.copy()

        # Update tracking
        self._update_history(next_state)
        self._step_index += 1
        terminated = False
        truncated = self._step_index >= self.number_time_steps
        reward = 0.0

        info: dict = {}
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            if triggered_labels:
                info["damage_events_triggered"] = triggered_labels

        obs = self._build_observation(next_state)
        return obs, reward, terminated, truncated, info

    def _build_observation(self, model_state: np.ndarray) -> np.ndarray:
        if not self.damage_observable or self.damage_manager is None:
            return model_state.copy()
        geo = self._geo_for_obs
        names = geo.section_names()
        loss_vec = np.array(
            [self.damage_manager.state.section_loss.get(n, 0.0) for n in names],
            dtype=np.float64,
        )
        thrust_vec = np.array(
            [self.damage_manager.state.engine.thrust_factor], dtype=np.float64
        )
        return np.concatenate([model_state, loss_vec, thrust_vec])

    def _extract_attitude(self, state: np.ndarray) -> np.ndarray:
        """(roll, pitch, yaw) = (gamma, theta, psi) from the state vector."""
        return np.array([state[5], state[7], state[6]])

    def _update_history(self, next_state: np.ndarray) -> None:
        """Append one row to position/attitude/time/chart histories."""
        # Reconstruct the inertial-step from the PREVIOUS state's velocity vector
        # (Euler integration, consistent with reconstruct_position_6dof).
        if len(self.model.x_history) >= 2:
            prev_state = self.model.x_history[-2].reshape(-1)
        else:
            prev_state = self.initial_state
        alpha, beta = prev_state[0], prev_state[1]
        roll, yaw, pitch = prev_state[5], prev_state[6], prev_state[7]
        v_body = _body_velocity(self.airspeed, alpha, beta)
        v_inertial = _body_to_inertial_matrix(roll, pitch, yaw) @ v_body
        new_pos = self.position_history[-1] + v_inertial * self.dt

        self.position_history = np.vstack([self.position_history, new_pos[None, :]])
        self.attitude_history = np.vstack(
            [
                self.attitude_history,
                self._extract_attitude(next_state).reshape(1, 3),
            ]
        )
        self.time_history = np.append(
            self.time_history, self._step_index * self.dt + self.dt
        )
        for name in self.chart_states:
            idx = MODEL_STATE_ORDER.index(name)
            self.chart_history[name] = np.append(
                self.chart_history[name],
                next_state[idx],
            )

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._render_human()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "live":
            return self._render_live()
        raise ValueError(f"Unknown render_mode: {self.render_mode!r}")

    def _build_figure(self):
        from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

        return build_flight_3d_figure(
            positions=self.position_history,
            attitudes=self.attitude_history,
            time=self.time_history,
            chart_data=self.chart_history,
            trail_length=self.trail_length,
        )

    def _render_human(self):
        # Don't auto-open the browser in tests; the caller invokes .show()
        # explicitly when they want the figure displayed.
        return self._build_figure()

    def _render_rgb_array(self):
        from io import BytesIO

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "rgb_array render mode requires Pillow. "
                "Install with `pip install Pillow`."
            ) from e
        fig = self._build_figure()
        png_bytes = fig.to_image(format="png")  # requires kaleido
        return np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))

    def _render_live(self):
        from tensoraerospace.visualization.live import LivePlotlyRenderer

        if not hasattr(self, "_live_renderer") or self._live_renderer is None:
            self._live_renderer = LivePlotlyRenderer(trail_length=self.trail_length)
            self._live_renderer.init_from(
                self.position_history,
                self.attitude_history,
                self.time_history,
                self.chart_history,
            )
            return self._live_renderer._fig
        # Subsequent calls: append the latest step
        self._live_renderer.extend(
            position_row=self.position_history[-1],
            attitude_row=self.attitude_history[-1],
            t=float(self.time_history[-1]),
            chart_row={
                name: float(self.chart_history[name][-1]) for name in self.chart_states
            },
        )
        return self._live_renderer._fig

    def close(self):
        return None
