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

from typing import Any, Literal, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager,
    DamageProfile,
    load_f16_geometry,
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

    metadata = {"render_modes": ["human", "rgb_array", "live", "3d_web"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        number_time_steps: int,
        *,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        airspeed: float = 200.0,
        render_mode: Optional[str] = None,
        chart_states: Sequence[str] = DEFAULT_CHART_STATES,
        trail_length: Optional[int] = None,
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback=None,
        split_stab: bool = False,
        track_altitude: bool = False,
        thrust_mode: Literal["constant", "control"] = "constant",
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
        self.track_altitude = track_altitude
        self.thrust_mode = thrust_mode
        self.damage_profile = damage_profile
        self.damage_observable = damage_observable
        self.damage_event_callback = damage_event_callback

        self.max_action_value = 25.0  # deg

        n_action = 4 if split_stab else 3
        if thrust_mode == "control":
            n_action += 1
        action_shape = (n_action,)
        self.action_space = spaces.Box(
            low=-self.max_action_value,
            high=self.max_action_value,
            shape=action_shape,
            dtype=np.float64,
        )

        # Observation: model state size + optional damage state vector
        obs_size = 16 if track_altitude else 14
        if damage_observable:
            geo = load_f16_geometry()
            obs_size += len(geo.section_names())
            obs_size += 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )
        self._geo_for_obs = (
            load_f16_geometry() if (damage_observable or damage_profile) else None
        )
        self.damage_manager: Optional[DamageManager] = None

        # Damage history accumulators — populated across an episode for the
        # 3D web exporter (tensoraerospace.visualization.three_d). Empty
        # lists when no damage_profile is configured.
        self.damage_events_log: list[dict] = []
        self.damage_state_log: list[dict] = []

        # Optional reference / commanded signals for visualization. Maps a
        # chart channel key (e.g. "alpha", "theta", "V", "h") to a sequence
        # of length T — values must be in the same DISPLAY units the
        # corresponding 3D-viewer chart uses (deg for angular states, m/s
        # for airspeed, m for altitude). The 3D exporter passes these
        # through to ``traj.references`` for the viewer to overlay on each
        # chart. Set by the user code before calling ``env.render()``.
        self.reference_signals: dict[str, list[float]] = {}

        # Filled in reset()
        self.model: AngularF16 | None = None
        self._step_index: int = 0
        self.position_history = np.zeros((0, 3))
        self.attitude_history = np.zeros((0, 3))
        self.time_history = np.zeros((0,))
        self.chart_history: dict[str, np.ndarray] = {}
        self._live_renderer: Any = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.model = AngularF16(
            x0=self.initial_state,
            t0=0,
            dt=self.dt,
            integrator=self.integrator,
            split_stab=self.split_stab,
            track_altitude=self.track_altitude,
            thrust_mode=self.thrust_mode,
        )
        self._step_index = 0

        # Damage manager
        if self.damage_profile is not None or self.damage_observable:
            geo = self._geo_for_obs
            if geo is None:
                raise RuntimeError("Damage mode requires F-16 geometry.")
            self.damage_manager = DamageManager(
                geometry=geo,
                params=self.model.param,
                profile=(self.damage_profile or DamageProfile(events=[])),
            )
            if options and "damage_profile" in options:
                self.damage_manager.set_profile(options["damage_profile"])
            self.damage_manager.reset(seed=seed)
            setattr(self.model, "damage_state", self.damage_manager.state)
            setattr(self.model, "damage_geometry", geo)
        else:
            self.damage_manager = None

        # Reset accumulator buffers and snapshot initial damage state
        self.damage_events_log = []
        self.damage_state_log = []
        if self.damage_manager is not None:
            self.damage_state_log.append(
                {
                    "time": 0.0,
                    "state": self.damage_manager.state.snapshot(),
                }
            )

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
        n_action = 4 if self.split_stab else 3
        if self.thrust_mode == "control":
            n_action += 1
        expected = (n_action,)
        if action.shape != expected:
            raise ValueError(f"action must be {expected}; got {action.shape}")
        if self.thrust_mode == "control":
            # Last element is thrust in Newtons; don't deg→rad it.
            surfaces = action[:-1]
            thrust = action[-1:]
            surfaces_clipped = np.clip(
                surfaces, -self.max_action_value, self.max_action_value
            )
            u_rad = np.concatenate([np.deg2rad(surfaces_clipped), thrust])
        else:
            action_clipped = np.clip(
                action, -self.max_action_value, self.max_action_value
            )
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
                self.damage_events_log.append(
                    {
                        "time": float(t_now),
                        "label": ev.label or ev.event_type,
                        "event_type": ev.event_type,
                        "payload": dict(ev.payload),
                    }
                )
            if triggered:
                # Snapshot the post-event damage state
                self.damage_state_log.append(
                    {
                        "time": float(t_now),
                        "state": self.damage_manager.state.snapshot(),
                    }
                )

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
        if geo is None:
            raise RuntimeError("Damage observation requires F-16 geometry.")
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
        assert self.model is not None
        # Reconstruct the inertial-step from the PREVIOUS state's velocity vector
        # (Euler integration, consistent with reconstruct_position_6dof).
        if len(self.model.x_history) >= 2:
            prev_state = self.model.x_history[-2].reshape(-1)
        else:
            prev_state = self.initial_state
        alpha, beta = prev_state[0], prev_state[1]
        roll, yaw, pitch = prev_state[5], prev_state[6], prev_state[7]
        # Use the integrated airspeed when available, otherwise the constant.
        v_for_step = (
            float(prev_state[15])
            if self.track_altitude and prev_state.size >= 16
            else self.airspeed
        )
        v_body = _body_velocity(v_for_step, alpha, beta)
        v_inertial = _body_to_inertial_matrix(roll, pitch, yaw) @ v_body
        new_pos = self.position_history[-1] + v_inertial * self.dt
        # When the model integrates altitude itself (state[14]), use that as
        # the source of truth for the vertical position. The kinematic
        # reconstruction above relies on a body-z-up DCM that disagrees
        # with the body-z-down convention assumed by the 3D viewer
        # (`y_three = -pos[2]`), so for non-trivial alpha the visual
        # vertical motion would not match the true altitude trajectory.
        if self.track_altitude and next_state.size >= 16:
            h_init = float(self.model.x_history[0].reshape(-1)[14])
            h_now = float(next_state[14])
            new_pos[2] = h_init - h_now

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
        if self.render_mode == "3d_web":
            return self._render_3d_web()
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
        try:
            png_bytes = fig.to_image(format="png")  # requires kaleido
        except RuntimeError as exc:
            # kaleido>=1 renders through a system Google Chrome that the user
            # must install themselves (env-side dependency, not packaged here).
            msg = str(exc)
            if any(
                token in msg
                for token in ("Chrome", "plotly_get_chrome", "ChromeNotFound")
            ):
                raise RuntimeError(
                    "Для render(mode='rgb_array') требуется системный Google Chrome "
                    "(kaleido>=1). Установите его: poetry run plotly_get_chrome"
                ) from exc
            raise
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

    def _render_3d_web(self):
        from tensoraerospace.visualization.three_d import render as _render_3d

        return _render_3d(self)

    def close(self):
        return None
