"""Gymnasium environment wrapping the pure-numpy nonlinear F-16 longitudinal model.

This is the side-by-side numpy counterpart to :mod:`linear_longitudial`. It
wraps :class:`tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.LongitudinalF16`
which uses the cubic-spline aerodynamic tables ported from the matlab source.

State vector exposed by the underlying model::

    [alpha, wz, stab, dstab]   (rad, rad/s, rad, rad/s)

Control vector::

    [stab_act]                 (commanded elevator deflection, rad)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager,
    DamageProfile,
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
    LongitudinalF16,
)
from tensoraerospace.visualization.kinematics import (
    _body_to_inertial_matrix,
    _body_velocity,
)

MODEL_STATE_ORDER = ["alpha", "wz", "stab", "dstab"]


class NonlinearLongitudinalF16(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear F-16 longitudinal model.

    Args:
        initial_state: Initial state vector. Either the full 4-element model
            state ``[alpha, wz, stab, dstab]`` or a shorter vector matching
            ``state_space`` (missing components are zero-filled). Values are
            interpreted in radians (matching the underlying model).
        reference_signal: Reference trajectory of shape ``(1, T)``, radians.
        number_time_steps: Episode length.
        tracking_states: Names of tracked states. Defaults to ``["alpha"]``.
        state_space: Subset of model states to expose as the observation.
            Observations are returned in radians. Defaults to ``["alpha", "wz"]``.
        control_space: Names of control channels. Defaults to ``["stab"]``.
        output_space: Compatibility alias of ``state_space``.
        reward_func: Custom reward callable ``(state, ref_signal, ts) -> float``.
        use_reward: If False, reward is fixed at 1.0 each step.
        dt: Discretisation step (s). Defaults to 0.01.
        integrator: ``"euler"`` (default, matches matlab) or ``"rk4"``.
        control_bias: Constant offset (degrees) added to every action before
            clipping and conversion to radians. Use this to operate the agent
            in "delta around trim" space when the trim control is non-zero
            (e.g., on the nonlinear F-16, trim elevator is ~-4.45°). Default 0.
        feedforward_fn: Optional callable ``(time_step, reference_signal) ->
            ff_action_deg``. Whenever set, the env adds the returned offset
            (degrees) to the agent's action before clipping. Use this to
            inject a precomputed feedforward map (e.g., trim elevator as a
            function of reference angle of attack) so a reactive agent like
            IHDP only has to learn the small disturbance correction instead
            of the full reference-tracking trajectory. ``time_step`` is the
            current step index; ``reference_signal`` is the same array given
            at construction time. The callable can return a scalar or an
            array of length ``len(control_space)``.

    Action units: by convention IHDP and most existing tensoraerospace agents
    were tuned with the linear F-16 env, whose elevator action is interpreted
    in **degrees** with magnitude limit 25 and rate limit 60. This env keeps
    the same convention so those agent settings transfer: ``action`` is in
    degrees, range ``[-25, 25]``, and is converted to radians internally
    before being handed to the underlying numpy model.
    """

    metadata = {"render_modes": ["human", "rgb_array", "live", "3d_web"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: (
            Callable[[np.ndarray, np.ndarray, int], np.ndarray | float] | None
        ) = None,
        use_reward: bool = True,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "euler",
        control_bias: float = 0.0,
        feedforward_fn: Callable[[int, np.ndarray], float] | None = None,
        airspeed: float = 200.0,
        render_mode: Optional[str] = None,
        chart_states: Sequence[str] = ("alpha", "wz", "stab"),
        trail_length: Optional[int] = None,
        initial_pitch: float = 0.0,
        damage_profile: Optional[DamageProfile] = None,
        damage_observable: bool = False,
        damage_event_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # Action is provided by the agent in DEGREES, converted to radians
        # before being passed to the underlying numpy model. This matches the
        # linear F-16 env so existing IHDP/PID/PPO settings transfer directly.
        self.max_action_value = 25.0  # elevator command limit (deg)
        self.initial_state = initial_state
        self.reference_signal = reference_signal
        self.number_time_steps = number_time_steps
        self.dt = dt
        self.integrator = integrator
        self.control_bias = float(control_bias)
        self.feedforward_fn = feedforward_fn
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["alpha"]
        )
        self.state_space = state_space if state_space is not None else ["alpha", "wz"]
        self.control_space = control_space if control_space is not None else ["stab"]
        self.output_space = (
            output_space if output_space is not None else list(self.state_space)
        )
        self.use_reward = use_reward
        self.reward_func = (
            reward_func if reward_func is not None else self.default_reward
        )
        self.damage_profile = damage_profile
        self.damage_observable = damage_observable
        self.damage_event_callback = damage_event_callback
        self._geo_for_damage = (
            load_f16_geometry()
            if (damage_observable or damage_profile is not None)
            else None
        )

        model_x0 = self._build_model_initial_state(self.initial_state, self.state_space)

        self.init_args = locals()
        self.model = LongitudinalF16(
            model_x0,
            selected_state_output=self.state_space,
            dt=dt,
            integrator=integrator,
        )

        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-self.max_action_value,
            high=self.max_action_value,
            shape=(len(self.control_space),),
            dtype=np.float32,
        )
        obs_size = len(self.state_space)
        if damage_observable:
            geo = self._geo_for_damage
            if geo is None:
                raise RuntimeError("Damage observation requires F-16 geometry.")
            obs_size += len(geo.section_names())
            obs_size += 1  # engine.thrust_factor
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.done = False

        self.airspeed = float(airspeed)
        self.render_mode = render_mode
        self.chart_states = tuple(chart_states)
        self.trail_length = trail_length
        self.initial_pitch = float(initial_pitch)
        self.damage_manager: Optional[DamageManager] = None

        # Damage history accumulators — populated across an episode for the
        # 3D web exporter (tensoraerospace.visualization.three_d). Empty
        # lists when no damage_profile is configured.
        self.damage_events_log: list[dict] = []
        self.damage_state_log: list[dict] = []

        # Initialised in reset()
        self.position_history = np.zeros((0, 3))
        self.attitude_history = np.zeros((0, 3))
        self.time_history = np.zeros((0,))
        self.chart_history: dict[str, np.ndarray] = {}
        self._live_renderer: Any = None

    @staticmethod
    def _build_model_initial_state(
        init_state: np.ndarray, state_names: list[str]
    ) -> np.ndarray:
        """Map a user-supplied initial vector into the full ``MODEL_STATE_ORDER``.

        Components for states not in ``state_names`` are zero-filled. If
        ``init_state`` already has 4 elements it is used as-is.
        """
        vals = np.asarray(init_state, dtype=float).reshape(-1)
        if vals.size == len(MODEL_STATE_ORDER):
            return vals.copy()
        x0 = np.zeros(len(MODEL_STATE_ORDER), dtype=float)
        for i, name in enumerate(state_names):
            if name in MODEL_STATE_ORDER and i < vals.size:
                x0[MODEL_STATE_ORDER.index(name)] = vals[i]
        return x0

    def get_init_args(self) -> dict[str, object]:
        init_args = self.init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        init_args.pop("model_x0", None)
        return init_args

    def _get_info(self) -> dict[str, object]:
        return {}

    def _build_observation(self, base_obs: np.ndarray) -> np.ndarray:
        if not self.damage_observable or self.damage_manager is None:
            return base_obs.astype(np.float32)
        geo = self._geo_for_damage
        if geo is None:
            raise RuntimeError("Damage observation requires F-16 geometry.")
        names = geo.section_names()
        loss_vec = np.array(
            [self.damage_manager.state.section_loss.get(n, 0.0) for n in names],
            dtype=np.float32,
        )
        thrust_vec = np.array(
            [self.damage_manager.state.engine.thrust_factor], dtype=np.float32
        )
        return np.concatenate([base_obs.astype(np.float32), loss_vec, thrust_vec])

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_deg = (
            np.asarray(action, dtype=np.float64).reshape(-1) + self.control_bias
        )
        if self.feedforward_fn is not None:
            ff = np.asarray(
                self.feedforward_fn(self.current_step, self.reference_signal),
                dtype=np.float64,
            ).reshape(-1)
            action_deg = action_deg + ff
        action_deg = np.clip(action_deg, -self.max_action_value, self.max_action_value)
        action_rad = np.deg2rad(action_deg)
        self.current_step += 1

        # Damage events (window: prior step → current step)
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            t_now = self.current_step * self.dt
            t_prev = (self.current_step - 1) * self.dt
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

        next_state = self.model.run_step(action_rad)
        # Track histories using the FULL 4-element model state (next_state may
        # be a sliced observation, depending on selected_state_output).
        self._update_history(self.model.current_state)

        reward: np.ndarray | float = 1.0
        if self.use_reward:
            reward = self.reward_func(
                next_state, self.reference_signal, self.current_step
            )

        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            if triggered_labels:
                info["damage_events_triggered"] = triggered_labels

        reward_value = float(np.asarray(reward, dtype=float).squeeze())

        base_obs = np.asarray(next_state).reshape(-1).astype(np.float32)
        return (
            self._build_observation(base_obs),
            reward_value,
            self.done,
            False,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False

        model_x0 = self._build_model_initial_state(self.initial_state, self.state_space)
        self.model = LongitudinalF16(
            model_x0,
            selected_state_output=self.state_space,
            dt=self.dt,
            integrator=self.integrator,
        )
        if self.damage_profile is not None or self.damage_observable:
            geo = self._geo_for_damage
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
        info = self._get_info()
        base_obs = np.asarray(model_x0, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape(-1)
        observation = self._build_observation(base_obs)

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
        self.attitude_history = np.array([[0.0, self.initial_pitch, 0.0]])
        self.time_history = np.zeros((1,), dtype=np.float64)
        self.chart_history = {
            name: np.array(
                [self.model.x_history[0].reshape(-1)[MODEL_STATE_ORDER.index(name)]]
            )
            for name in self.chart_states
        }
        self._live_renderer = None

        return observation, info

    def close(self) -> None:
        pass

    @staticmethod
    def default_reward(
        state: np.ndarray, ref_signal: np.ndarray, ts: int
    ) -> np.ndarray:
        """Tracking-error reward with a small angular-rate penalty."""
        state_vec = np.asarray(state).reshape(-1)
        ts_safe = int(np.clip(ts, 0, ref_signal.shape[1] - 1))
        tracked_val = float(state_vec[0])
        ref_val = float(np.asarray(ref_signal[:, ts_safe]).reshape(-1)[0])
        angle_error = abs(tracked_val - ref_val)
        rate_penalty = float(abs(state_vec[1])) if state_vec.size > 1 else 0.0
        reward = -angle_error - 0.1 * rate_penalty
        return np.array(reward)

    def _update_history(self, next_state: np.ndarray) -> None:
        """Append one row to position/attitude/time/chart histories."""
        next_state = np.asarray(next_state, dtype=np.float64).reshape(-1)
        prev_pitch = self.attitude_history[-1, 1]
        # Pull previous full model state for body-velocity computation.
        if hasattr(self.model, "x_history") and len(self.model.x_history) >= 2:
            prev_model_state = self.model.x_history[-2].reshape(-1)
        else:
            prev_model_state = next_state  # fallback: use current state
        alpha = prev_model_state[0]
        wz = prev_model_state[1]
        new_pitch = prev_pitch + wz * self.dt
        v_body = _body_velocity(self.airspeed, alpha, beta=0.0)
        v_inertial = _body_to_inertial_matrix(0.0, prev_pitch, 0.0) @ v_body
        new_pos = self.position_history[-1] + v_inertial * self.dt

        self.position_history = np.vstack([self.position_history, new_pos[None, :]])
        self.attitude_history = np.vstack(
            [
                self.attitude_history,
                np.array([[0.0, new_pitch, 0.0]]),
            ]
        )
        self.time_history = np.append(
            self.time_history,
            self.time_history[-1] + self.dt,
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
            return self._build_figure()
        if self.render_mode == "rgb_array":
            from io import BytesIO

            try:
                from PIL import Image
            except ImportError as e:
                raise ImportError(
                    "rgb_array render mode requires Pillow. "
                    "Install with `pip install Pillow`."
                ) from e
            fig = self._build_figure()
            png_bytes = fig.to_image(format="png")
            return np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))
        if self.render_mode == "live":
            from tensoraerospace.visualization.live import LivePlotlyRenderer

            if self._live_renderer is None:
                self._live_renderer = LivePlotlyRenderer(
                    trail_length=self.trail_length,
                )
                self._live_renderer.init_from(
                    self.position_history,
                    self.attitude_history,
                    self.time_history,
                    self.chart_history,
                )
                return self._live_renderer._fig
            self._live_renderer.extend(
                position_row=self.position_history[-1],
                attitude_row=self.attitude_history[-1],
                t=float(self.time_history[-1]),
                chart_row={
                    name: float(self.chart_history[name][-1])
                    for name in self.chart_states
                },
            )
            return self._live_renderer._fig
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

    def _render_3d_web(self):
        from tensoraerospace.visualization.three_d import render as _render_3d

        return _render_3d(self)
