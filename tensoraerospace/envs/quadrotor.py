"""Gymnasium env wrapping the pure-numpy nonlinear 6-DoF quadrotor model.

Two action-space modes:

- ``action_space="virtual"`` (default) — agent gives ``[T, τ_x, τ_y, τ_z]``,
  fed straight into the ODE. Suitable for trajectory / attitude
  controllers (PID, MPC, iADP, AIDI) that already work in physical
  primitives.
- ``action_space="rotor"`` — agent gives ``[ω₁², ω₂², ω₃², ω₄²]`` (4 rotor
  speeds-squared, units :math:`(\\mathrm{rad/s})^2`). The env applies
  the X-config allocator to map them into virtual ``[T, τ]`` before
  integration. Required for fault-tolerant scenarios where damage acts
  at the rotor level.

When a :class:`DamageProfile` is supplied, every step the env:

1. Reads the commanded action.
2. Converts it to commanded rotor speeds-squared
   :math:`\\omega^2_{i,\\text{cmd}}` (a no-op in ``"rotor"`` mode).
3. Multiplies element-wise by the per-rotor effectiveness ``mu`` from
   :class:`RotorDamageState`.
4. Re-mixes back to virtual ``[T_eff, τ_eff]``.
5. Integrates the ODE with the effective virtual.

Without ``damage_profile``, behaviour is bit-identical to the bare
``NonlinearQuadrotor`` model.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.quadrotor.allocation import (
    XConfigAllocator,
    default_allocator,
)
from tensoraerospace.aerospacemodel.quadrotor.damage import (
    DamageProfile,
    RotorDamageManager,
)
from tensoraerospace.aerospacemodel.quadrotor.nonlinear import NonlinearQuadrotor

MODEL_STATE_ORDER = [
    "x_e",
    "y_e",
    "z_e",
    "u_b",
    "v_b",
    "w_b",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
]


class NonlinearQuadrotorEnv(gym.Env):
    """Gymnasium env over the pure-numpy nonlinear 6-DoF quadrotor.

    Args:
        initial_state: 12-element initial state. See
            :mod:`tensoraerospace.aerospacemodel.quadrotor.nonlinear` for
            the layout.
        number_time_steps: Episode length cap (steps).
        dt: Discretisation step (s). Default 0.01.
        integrator: ``"euler"`` or ``"rk4"`` (default).
        action_space: ``"virtual"`` (4 = T+τ) or ``"rotor"`` (4 = ω²
            per motor). See module docstring.
        allocator: X-config allocator to use when ``action_space="rotor"``
            or when damage is active. ``None`` → :func:`default_allocator`.
        damage_profile: Optional :class:`DamageProfile` to apply over the
            episode.
        damage_event_callback: Optional ``f(event, state)`` called after
            each event triggers (for logging / TensorBoard).
        omega_min: Lower rotor-speed bound, rad/s. Used in the env's
            saturation step before mixing. Default 0.
        omega_max: Upper rotor-speed bound, rad/s. Default 1000.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_state: np.ndarray,
        number_time_steps: int,
        *,
        dt: float = 0.01,
        integrator: Literal["euler", "rk4"] = "rk4",
        action_space: Literal["virtual", "rotor"] = "virtual",
        allocator: Optional[XConfigAllocator] = None,
        damage_profile: Optional[DamageProfile] = None,
        damage_event_callback: Optional[Callable[[Any, Any], None]] = None,
        omega_min: float = 0.0,
        omega_max: float = 1000.0,
    ) -> None:
        super().__init__()

        x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if x0.size != 12:
            raise ValueError(f"initial_state must have 12 elements; got {x0.size}")
        self.initial_state = x0

        self.number_time_steps = int(number_time_steps)
        self.dt = float(dt)
        self.integrator = integrator
        self.action_mode = action_space
        if action_space not in ("virtual", "rotor"):
            raise ValueError(
                f'action_space must be "virtual" or "rotor"; got {action_space!r}'
            )

        self.allocator = allocator if allocator is not None else default_allocator()
        self.damage_profile = damage_profile
        self.damage_event_callback = damage_event_callback
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        # Observation: 12-D state vector (no extra dressing in v0).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
        )

        # Action space depends on mode.
        if action_space == "virtual":
            # Generous bounds so PID/MPC can issue any [T, τ] they like.
            high_virt = np.array([1e3, 1e2, 1e2, 1e2], dtype=np.float64)
            self.action_space = spaces.Box(
                low=-high_virt, high=high_virt, dtype=np.float64
            )
        else:  # "rotor"
            high_rot = np.full(4, omega_max**2, dtype=np.float64)
            self.action_space = spaces.Box(
                low=np.zeros(4), high=high_rot, dtype=np.float64
            )

        # Lazy-init in reset()
        self.model: Optional[NonlinearQuadrotor] = None
        self.damage_manager: Optional[RotorDamageManager] = None
        self._step_index: int = 0
        self.damage_events_log: list[dict] = []
        self.damage_state_log: list[dict] = []

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self.model = NonlinearQuadrotor(
            x0=self.initial_state, dt=self.dt, integrator=self.integrator
        )
        self._step_index = 0

        if self.damage_profile is not None or (options and "damage_profile" in options):
            self.damage_manager = RotorDamageManager(
                profile=self.damage_profile or DamageProfile(events=[])
            )
            if options and "damage_profile" in options:
                self.damage_manager.set_profile(options["damage_profile"])
            self.damage_manager.reset(seed=seed)
        else:
            self.damage_manager = None

        self.damage_events_log = []
        self.damage_state_log = []

        return self.model.current_state.copy(), {}

    def step(self, action):  # type: ignore[override]
        if self.model is None:
            raise RuntimeError("env.reset() must be called before step()")

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError(f"action must have 4 elements; got {action.size}")

        # Time bookkeeping (BEFORE step)
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt

        # Damage update (advance decay, fire scheduled events)
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            triggered = self.damage_manager.update(t_now, t_prev, self.dt)
            for ev in triggered:
                if self.damage_event_callback:
                    self.damage_event_callback(ev, self.damage_manager.state)
                triggered_labels.append(ev.label or type(ev).__name__)
                self.damage_events_log.append(
                    {
                        "time": float(t_now),
                        "label": ev.label or type(ev).__name__,
                        "rotor_id": ev.rotor_id,
                        "kind": type(ev).__name__,
                    }
                )

        # Resolve commanded rotor speeds² (or compute from virtual)
        if self.action_mode == "rotor":
            omega2_cmd = action.copy()
        else:  # virtual
            omega2_cmd = self.allocator.unmix(action)

        # Apply damage at rotor level (if active)
        if self.damage_manager is not None:
            mu = self.damage_manager.state.mu
            omega2_eff = mu * omega2_cmd
            self.damage_state_log.append(
                {
                    "time": float(t_now),
                    "state": self.damage_manager.state.snapshot(),
                }
            )
        else:
            omega2_eff = omega2_cmd

        # Saturate to physical bounds (also handles negative ω² from
        # non-realisable virtual commands)
        omega2_eff = self.allocator.saturate(
            omega2_eff, omega_min=self.omega_min, omega_max=self.omega_max
        )

        # Mix back to virtual; this is what the ODE consumes
        u_virtual_eff = self.allocator.mix(omega2_eff)

        # Integrate
        self.model.run_step(u_virtual_eff)
        self._step_index += 1

        next_state = self.model.current_state.copy()
        terminated = False
        truncated = self._step_index >= self.number_time_steps
        reward = 0.0

        info: dict = {}
        if self.damage_manager is not None:
            info["damage_state"] = self.damage_manager.state.snapshot()
            info["omega2_cmd"] = omega2_cmd.tolist()
            info["omega2_eff"] = omega2_eff.tolist()
            info["u_virtual_eff"] = u_virtual_eff.tolist()
            if triggered_labels:
                info["damage_events_triggered"] = list(triggered_labels)

        return next_state, reward, terminated, truncated, info
