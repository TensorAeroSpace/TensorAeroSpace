"""Gymnasium env wrapping the nonlinear AAI RQ-7 Shadow UAV.

API parity with the rest of the family — same `"virtual"` /
`"normalized"` action modes, same trim-finder + initial-state
initialisation patterns. The Shadow has a 4-channel control vector
(unlike the Skywalker X8's 3-channel flying wing) because of its
V-tail rudder.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
    NonlinearAAIShadow,
    trim,
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
]


class NonlinearAAIShadowEnv(gym.Env):
    """Gymnasium env over the nonlinear AAI RQ-7 Shadow.

    4-channel control: ``[δ_e, δ_a, δ_r, δ_T]`` (mixed-V-tail
    convention). Use ``"virtual"`` for raw physical units (rad / [0, 1])
    or ``"normalized"`` for ``[-1, +1]^4`` rescaled internally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        initial_state: Optional[np.ndarray] = None,
        trim_at: Optional[tuple[float, float]] = None,
        number_time_steps: int = 2000,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        action_space: Literal["virtual", "normalized"] = "virtual",
        damage_profile: Optional[Any] = None,
        damage_event_callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        super().__init__()

        x0 = self._resolve_initial_state(initial_state, trim_at)
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
        self.damage_profile = damage_profile
        self.damage_event_callback = damage_event_callback

        high_obs = np.full(12, np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float64
        )

        if action_space == "virtual":
            high_act = np.array(
                [
                    np.deg2rad(20.0),  # elevator (collective ruddervator)
                    np.deg2rad(20.0),  # aileron
                    np.deg2rad(15.0),  # rudder (differential ruddervator)
                    1.0,  # throttle
                ],
                dtype=np.float64,
            )
            low_act = np.array(
                [
                    -np.deg2rad(20.0),
                    -np.deg2rad(20.0),
                    -np.deg2rad(15.0),
                    0.0,
                ],
                dtype=np.float64,
            )
        else:
            high_act = np.ones(4, dtype=np.float64)
            low_act = -np.ones(4, dtype=np.float64)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float64)

        self.model: Optional[NonlinearAAIShadow] = None
        self._step_index: int = 0

    @staticmethod
    def _resolve_initial_state(initial_state, trim_at) -> np.ndarray:
        provided = sum(int(x is not None) for x in (initial_state, trim_at))
        if provided == 0:
            raise ValueError("must supply one of: initial_state, trim_at")
        if provided > 1:
            raise ValueError("specify exactly one of: initial_state, trim_at")
        if initial_state is not None:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
            if x0.size != 12:
                raise ValueError(f"initial_state must have 12 elements; got {x0.size}")
            return x0
        alt, V = trim_at
        result = trim(altitude_m=float(alt), V_m_s=float(V))
        if not result.converged:
            raise RuntimeError(
                f"trim failed at altitude={alt} m, V={V} m/s "
                f"(residual {result.residual:.3e})"
            )
        return result.to_state()

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mode == "virtual":
            return action.astype(np.float64, copy=True)
        u_e, u_a, u_r, u_T = action[0], action[1], action[2], action[3]
        return np.array(
            [
                float(u_e) * np.deg2rad(20.0),
                float(u_a) * np.deg2rad(20.0),
                float(u_r) * np.deg2rad(15.0),
                (float(u_T) + 1.0) * 0.5,
            ],
            dtype=np.float64,
        )

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.model = NonlinearAAIShadow(
            x0=self.initial_state,
            dt=self.dt,
            integrator=self.integrator,
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
        return next_state, 0.0, terminated, truncated, {}
