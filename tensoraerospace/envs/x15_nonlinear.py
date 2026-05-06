"""Gymnasium environment wrapping the pure-numpy nonlinear X-15.

The X-15 env follows the same conventions as :class:`~tensoraerospace.\
envs.b747_nonlinear.NonlinearB747Env`:

* ``action_space="virtual"`` — agent passes raw ``[δ_e, δ_a, δ_r, δ_T]``
  in physical units (radians + throttle in [0, 1]).
* ``action_space="normalized"`` — agent passes ``[u_e, u_a, u_r, u_T]``
  in ``[-1, +1]^4``; the env rescales internally.

The observation is the **13-D model state** including the propellant
mass channel. Agents that care only about the rigid-body dynamics
can ignore ``obs[12]``; agents that need to plan around burnout
(e.g. choose throttle to extend powered phase) can use it directly.

Three initialisation modes:

* ``initial_state`` — full 13-element vector.
* ``flight_condition_id`` — pick one of the 5 anchor FCs (1..5).
* ``trim_at`` — solve for steady flight at ``(altitude_ft, V_ft_s)``
  via :func:`tensoraerospace.aerospacemodel.x15.nonlinear.trim`. This
  uses the powered trim by default; for a glide pass
  ``glide=True``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.x15.nonlinear import (
    NonlinearX15,
    X15Configuration,
    initial_state_from_fc,
    trim,
)
from tensoraerospace.aerospacemodel.x15.nonlinear.flight_conditions import (
    get_flight_condition,
)

STATE_ORDER = [
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "phi",
    "theta",
    "psi",
    "x_e",
    "y_e",
    "z_e",
    "m_prop",
]


class NonlinearX15Env(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear X-15.

    Args:
        initial_state: 13-element initial state. If ``None``, must
            supply ``trim_at`` or ``flight_condition_id``.
        flight_condition_id: One of the 5 published anchor FCs (1..5).
        trim_at: ``(altitude_ft, V_ft_s)`` for the powered trim
            finder. Solves for steady flight at the given throttle
            (default 1.0).
        trim_throttle: Throttle setting used by the trim finder
            (defaults to 1.0). Ignored unless ``trim_at`` is set.
        number_time_steps: Episode length cap.
        dt: Discretisation step (s).
        integrator: ``"euler"`` or ``"rk4"`` (default).
        action_space: ``"virtual"`` or ``"normalized"``.
        config: BASIC or A2.
        damage_profile: Reserved for the future damage subsystem
            (parity with B-747 env). Currently a no-op pass-through.
        damage_event_callback: Same as B-747 env.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        initial_state: Optional[np.ndarray] = None,
        flight_condition_id: Optional[int] = None,
        trim_at: Optional[tuple[float, float]] = None,
        trim_throttle: float = 1.0,
        number_time_steps: int = 2000,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        action_space: Literal["virtual", "normalized"] = "virtual",
        config: X15Configuration = X15Configuration.BASIC,
        damage_profile: Optional[Any] = None,
        damage_event_callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        super().__init__()

        x0 = self._resolve_initial_state(
            initial_state, flight_condition_id, trim_at, trim_throttle, config
        )
        self.initial_state = x0
        self.number_time_steps = int(number_time_steps)
        self.dt = float(dt)
        self.integrator = integrator
        self.action_mode = action_space
        if action_space not in ("virtual", "normalized"):
            raise ValueError(
                'action_space must be "virtual" or "normalized"; '
                f"got {action_space!r}"
            )
        self.config = config
        self.damage_profile = damage_profile
        self.damage_event_callback = damage_event_callback

        # Observation: 13-D state (12-D rigid body + propellant)
        high_obs = np.full(13, np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float64
        )

        if action_space == "virtual":
            high_act = np.array(
                [
                    np.deg2rad(15.0),  # all-flying stab
                    np.deg2rad(15.0),
                    np.deg2rad(8.5),
                    1.0,
                ],
                dtype=np.float64,
            )
            low_act = np.array(
                [
                    -np.deg2rad(15.0),
                    -np.deg2rad(15.0),
                    -np.deg2rad(8.5),
                    0.0,
                ],
                dtype=np.float64,
            )
        else:
            high_act = np.ones(4, dtype=np.float64)
            low_act = -np.ones(4, dtype=np.float64)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float64)

        self.model: Optional[NonlinearX15] = None
        self._step_index: int = 0

    @staticmethod
    def _resolve_initial_state(
        initial_state, flight_condition_id, trim_at, trim_throttle, config
    ) -> np.ndarray:
        provided = sum(
            int(x is not None) for x in (initial_state, flight_condition_id, trim_at)
        )
        if provided == 0:
            raise ValueError(
                "must supply one of: initial_state, flight_condition_id, trim_at"
            )
        if provided > 1:
            raise ValueError(
                "specify exactly one of: initial_state, flight_condition_id, trim_at"
            )
        if initial_state is not None:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
            if x0.size != 13:
                raise ValueError(f"initial_state must have 13 elements; got {x0.size}")
            return x0
        if flight_condition_id is not None:
            fc = get_flight_condition(int(flight_condition_id))
            return initial_state_from_fc(fc, config=config)
        alt, V = trim_at
        result = trim(
            altitude_ft=float(alt),
            V_ft_s=float(V),
            config=config,
            throttle=float(trim_throttle),
        )
        # X-15 trim is approximate at most operating points — accept
        # any state fsolve returned, but flag a warning if residual is
        # huge (> 100 ft/s² or rad/s²).
        if result.residual > 100.0:
            import warnings

            warnings.warn(
                f"X-15 trim at h={alt:.0f}ft, V={V:.1f}ft/s, throttle="
                f"{trim_throttle:.2f} did not converge "
                f"(residual = {result.residual:.2e}). Use the returned "
                f"state as a best-effort initial condition.",
                stacklevel=3,
            )
        return result.to_state()

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mode == "virtual":
            return action.astype(np.float64, copy=True)
        u_e, u_a, u_r, u_T = action[0], action[1], action[2], action[3]
        return np.array(
            [
                float(u_e) * np.deg2rad(15.0),
                float(u_a) * np.deg2rad(15.0),
                float(u_r) * np.deg2rad(8.5),
                (float(u_T) + 1.0) * 0.5,
            ],
            dtype=np.float64,
        )

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.model = NonlinearX15(
            x0=self.initial_state,
            dt=self.dt,
            integrator=self.integrator,
            config=self.config,
        )
        self._step_index = 0
        return self.model.current_state.copy(), {}

    def step(self, action):
        if self.model is None:
            raise RuntimeError("env.reset() must be called before step()")

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError(f"action must have 4 elements; got {action.size}")
        u_virtual = self._scale_action(action)

        self.model.run_step(u_virtual)
        self._step_index += 1

        next_state = self.model.current_state.copy()
        terminated = False
        truncated = self._step_index >= self.number_time_steps
        reward = 0.0

        info: dict = {
            "propellant_lb": float(next_state[12]),
            "engine_running": bool(next_state[12] > 0.0),
        }
        return next_state, reward, terminated, truncated, info
