"""Gymnasium env wrapping the pure-numpy nonlinear Boeing 737.

Mirror of :class:`~tensoraerospace.envs.b747_nonlinear.NonlinearB747Env`
— same 4-channel virtual / normalized action modes, same trim-finder
and flight-condition initialisation patterns. The B-737 has two
configurations (737-100 and 737-NG / 737-800); pick via the
``config`` argument.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.b737.nonlinear import (
    B737Configuration,
    NonlinearB737,
    trim,
)


STATE_ORDER = [
    "u", "v", "w",
    "p", "q", "r",
    "phi", "theta", "psi",
    "x_e", "y_e", "z_e",
]


class NonlinearB737Env(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear 6-DoF Boeing 737."""

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
        config: B737Configuration = B737Configuration.B737_100,
        damage_profile: Optional[Any] = None,
        damage_event_callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        super().__init__()

        x0 = self._resolve_initial_state(initial_state, trim_at, config)
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

        # Observation: 12-D state
        high_obs = np.full(12, np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float64)

        if action_space == "virtual":
            high_act = np.array([
                np.deg2rad(17.2),   # elevator
                np.deg2rad(20.1),   # aileron
                np.deg2rad(20.1),   # rudder
                1.0,
            ], dtype=np.float64)
            low_act = np.array([
                -np.deg2rad(17.2),
                -np.deg2rad(20.1),
                -np.deg2rad(20.1),
                0.0,
            ], dtype=np.float64)
        else:
            high_act = np.ones(4, dtype=np.float64)
            low_act = -np.ones(4, dtype=np.float64)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float64)

        self.model: Optional[NonlinearB737] = None
        self._step_index: int = 0

    @staticmethod
    def _resolve_initial_state(
        initial_state, trim_at, config
    ) -> np.ndarray:
        provided = sum(int(x is not None) for x in (initial_state, trim_at))
        if provided == 0:
            raise ValueError("must supply one of: initial_state, trim_at")
        if provided > 1:
            raise ValueError("specify exactly one of: initial_state, trim_at")
        if initial_state is not None:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
            if x0.size != 12:
                raise ValueError(
                    f"initial_state must have 12 elements; got {x0.size}"
                )
            return x0
        alt, V = trim_at
        result = trim(altitude_ft=float(alt), V_ft_s=float(V), config=config)
        if not result.converged:
            raise RuntimeError(
                f"trim failed at altitude={alt} ft, V={V} ft/s "
                f"(residual {result.residual:.3e})"
            )
        return result.to_state()

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mode == "virtual":
            return action.astype(np.float64, copy=True)
        u_e, u_a, u_r, u_T = action[0], action[1], action[2], action[3]
        return np.array([
            float(u_e) * np.deg2rad(17.2),
            float(u_a) * np.deg2rad(20.1),
            float(u_r) * np.deg2rad(20.1),
            (float(u_T) + 1.0) * 0.5,
        ], dtype=np.float64)

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.model = NonlinearB737(
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
        return next_state, 0.0, terminated, truncated, {}
