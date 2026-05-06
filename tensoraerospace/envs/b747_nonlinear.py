"""Gymnasium env wrapping the pure-numpy Boeing 747 nonlinear 6-DoF model.

Two action-space modes:

* ``action_space="virtual"`` (default) — agent gives raw physical
  commands ``[δ_e, δ_a, δ_r, δ_T]`` (radians + throttle in [0, 1]).
  Suitable for PID, MPC, IHDP, ET-DHP — anything that already speaks
  the model's native units.

* ``action_space="normalized"`` — agent gives ``[u_e, u_a, u_r, u_T]``
  in ``[-1, +1]^4``. The env rescales internally:

      δ_e = u_e · elevator_max
      δ_a = u_a · aileron_max
      δ_r = u_r · rudder_max
      δ_T = (u_T + 1) / 2          (in [0, 1])

  This is the standard "RL-friendly" wrapper, matching the Improved-*
  family of envs already in TensorAeroSpace.

Without a ``damage_profile`` the env is bit-identical to the bare
model. With one, surface effectiveness is multiplied by the manager's
state vector before each integrator step (see :mod:`.b747.nonlinear.damage`).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.b747.nonlinear import (
    B747Configuration,
    NonlinearB747,
    initial_state_from_fc,
    trim,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.flight_conditions import (
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
]


class NonlinearB747Env(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear 6-DoF B-747.

    Args:
        initial_state: 12-element initial state. If ``None``, must
            supply ``trim_at`` or ``flight_condition_id``.
        flight_condition_id: Convenience initialiser — pick one of the
            10 published trim points (1..10).
        trim_at: ``(altitude_ft, V_ft_s)`` pair to compute trim
            on-the-fly via Newton-Raphson. Mutually exclusive with the
            other two initialisers.
        number_time_steps: Episode length cap.
        dt: Discretisation step (s).
        integrator: ``"euler"`` or ``"rk4"`` (default).
        action_space: ``"virtual"`` (physical units) or ``"normalized"``
            (RL [-1, +1]).
        config: Aerodynamic configuration to use.
        damage_profile: Optional :class:`DamageProfile` to apply over
            the episode.
        damage_event_callback: Optional ``f(event, state)`` called after
            each event triggers (for logging / TensorBoard hooks).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        initial_state: Optional[np.ndarray] = None,
        flight_condition_id: Optional[int] = None,
        trim_at: Optional[tuple[float, float]] = None,
        number_time_steps: int = 2000,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        action_space: Literal["virtual", "normalized"] = "virtual",
        config: B747Configuration = B747Configuration.NOMINAL,
        damage_profile: Optional[Any] = None,
        damage_event_callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        super().__init__()

        x0 = self._resolve_initial_state(
            initial_state, flight_condition_id, trim_at, config
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

        # Observation: 12-D state (no extra dressing — agents that need
        # a different observation shape should compose this env).
        high_obs = np.full(12, np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float64
        )

        if action_space == "virtual":
            # [δ_e, δ_a, δ_r, δ_T] — physical units
            high_act = np.array(
                [
                    np.deg2rad(25.0),
                    np.deg2rad(20.0),
                    np.deg2rad(25.0),
                    1.0,
                ],
                dtype=np.float64,
            )
            low_act = np.array(
                [
                    -np.deg2rad(25.0),
                    -np.deg2rad(20.0),
                    -np.deg2rad(25.0),
                    0.0,
                ],
                dtype=np.float64,
            )
        else:
            high_act = np.ones(4, dtype=np.float64)
            low_act = -np.ones(4, dtype=np.float64)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float64)

        # Lazy-init in reset()
        self.model: Optional[NonlinearB747] = None
        self.damage_manager: Optional[Any] = None
        self._step_index: int = 0
        self.damage_events_log: list[dict] = []

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _resolve_initial_state(
        initial_state, flight_condition_id, trim_at, config
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
            if x0.size != 12:
                raise ValueError(f"initial_state must have 12 elements; got {x0.size}")
            return x0
        if flight_condition_id is not None:
            fc = get_flight_condition(int(flight_condition_id))
            return initial_state_from_fc(fc)
        # trim_at = (alt, V)
        alt, V = trim_at
        result = trim(altitude_ft=float(alt), V_ft_s=float(V), config=config)
        if not result.converged:
            raise RuntimeError(
                f"trim failed at altitude={alt} ft, V={V} ft/s "
                f"(residual {result.residual:.3e})"
            )
        return result.to_state()

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Convert agent action to virtual command ``[δ_e, δ_a, δ_r, δ_T]``."""
        if self.action_mode == "virtual":
            return action.astype(np.float64, copy=True)
        # normalized: [-1, +1] → physical
        u_e, u_a, u_r, u_T = action[0], action[1], action[2], action[3]
        return np.array(
            [
                (
                    float(u_e) * self.action_space.high[0]
                    if False
                    else float(u_e) * np.deg2rad(25.0)
                ),
                float(u_a) * np.deg2rad(20.0),
                float(u_r) * np.deg2rad(25.0),
                (float(u_T) + 1.0) * 0.5,
            ],
            dtype=np.float64,
        )

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.model = NonlinearB747(
            x0=self.initial_state,
            dt=self.dt,
            integrator=self.integrator,
            config=self.config,
        )
        self._step_index = 0

        if self.damage_profile is not None or (options and "damage_profile" in options):
            from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
                B747DamageManager,
                DamageProfile,
            )

            profile = self.damage_profile or DamageProfile(events=[])
            if options and "damage_profile" in options:
                profile = options["damage_profile"]
            self.damage_manager = B747DamageManager(profile=profile)
            self.damage_manager.reset(seed=seed)
            # Link the manager's mutable state into the model so aero /
            # engine read engines_mu and flap_jam_config on every step.
            self.model.damage_state = self.damage_manager.state
        else:
            self.damage_manager = None
            self.model.damage_state = None

        self.damage_events_log = []
        return self.model.current_state.copy(), {}

    def step(self, action):
        if self.model is None:
            raise RuntimeError("env.reset() must be called before step()")

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError(f"action must have 4 elements; got {action.size}")
        u_virtual = self._scale_action(action)

        # Damage update (BEFORE applying to control surfaces)
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            triggered = self.damage_manager.update(t_now, t_prev, self.dt)
            for ev in triggered:
                if self.damage_event_callback is not None:
                    self.damage_event_callback(ev, self.damage_manager.state)
                triggered_labels.append(ev.label or type(ev).__name__)
                self.damage_events_log.append(
                    {
                        "time": float(t_now),
                        "label": ev.label or type(ev).__name__,
                        "kind": type(ev).__name__,
                    }
                )
            # Apply per-surface effectiveness multipliers + jam holds
            ds = self.damage_manager.state
            u_virtual = ds.apply(u_virtual)

        self.model.run_step(u_virtual)
        self._step_index += 1

        next_state = self.model.current_state.copy()
        terminated = False
        truncated = self._step_index >= self.number_time_steps
        reward = 0.0

        info: dict = {}
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            info["u_virtual_eff"] = u_virtual.tolist()
            if triggered_labels:
                info["damage_events_triggered"] = list(triggered_labels)

        return next_state, reward, terminated, truncated, info
