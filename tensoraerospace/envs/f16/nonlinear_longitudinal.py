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
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
    LongitudinalF16,
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

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: callable = None,
        use_reward: bool = True,
        dt: float = 0.01,
        integrator: str = "euler",
        control_bias: float = 0.0,
        feedforward_fn: Callable[[int, np.ndarray], float] | None = None,
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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_space),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.done = False

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

    def _get_info(self) -> dict[str, float]:
        return {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
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

        next_state = self.model.run_step(action_rad)

        reward = 1.0
        if self.use_reward:
            reward = self.reward_func(
                next_state, self.reference_signal, self.current_step
            )

        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()

        reward_value = float(np.asarray(reward, dtype=float).squeeze())

        return (
            np.asarray(next_state).reshape(-1).astype(np.float32),
            reward_value,
            self.done,
            False,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
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
        info = self._get_info()
        observation = np.asarray(model_x0, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape(-1)
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
