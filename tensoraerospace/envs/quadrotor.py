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

With damage, integration is split at the exact event times. Rotor
commands are saturated before effectiveness is applied. Exponential wear
is evaluated at each integrator stage within a smooth interval, so events
at the end of a step cannot reduce thrust earlier in that step. The history
and ``info`` record effective controls at the end of the sample.

Without damage, the allocator still enforces rotor-speed limits. Realisable
virtual commands match the bare model up to allocation roundoff.
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
    RotorDamageState,
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
    action_space: spaces.Box

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

        x0 = np.array(initial_state, dtype=np.float64, copy=True).reshape(-1)
        if x0.size != 12:
            raise ValueError(f"initial_state must have 12 elements; got {x0.size}")
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must contain only finite values")
        self.initial_state = x0

        self.number_time_steps = int(number_time_steps)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0 or self.number_time_steps <= 0:
            raise ValueError(
                "dt must be finite and positive; number_time_steps positive"
            )
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
        self.allocator.saturate(np.zeros(4), self.omega_min, self.omega_max)

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
                low=np.full(4, omega_min**2), high=high_rot, dtype=np.float64
            )

        # Lazy-init in reset()
        self.model: Optional[NonlinearQuadrotor] = None
        self.damage_manager: Optional[RotorDamageManager] = None
        self._step_index: int = 0
        self.damage_events_log: list[dict] = []
        self.damage_state_log: list[dict] = []

    # ---- gym API -------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset the vehicle and rotor faults, optionally using an episode-specific
        damage profile.
        """
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

    def step(self, action):
        """Advance commanded rotor speeds squared or virtual thrust/moments by one step.

        Virtual controls use newtons and N m; rotor commands use (rad/s)². Allocation,
        saturation and timed rotor faults determine applied forces. Return the Gymnasium
        transition with effective-control and damage info.
        """
        if self.model is None:
            raise RuntimeError("env.reset() must be called before step()")

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 4:
            raise ValueError(f"action must have 4 elements; got {action.size}")

        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        # Time bookkeeping (BEFORE step)
        t_prev = self._step_index * self.dt
        t_now = (self._step_index + 1) * self.dt

        # Resolve commanded rotor speeds² (or compute from virtual)
        if self.action_mode == "rotor":
            omega2_cmd = action.copy()
        else:  # virtual
            omega2_cmd = self.allocator.unmix(action)

        # Limit motor commands before applying the surviving effectiveness.
        # Clipping effective thrust afterwards can resurrect a stopped rotor
        # at omega_min or hide partial damage under an excessive command.
        omega2_cmd = self.allocator.saturate(
            omega2_cmd, omega_min=self.omega_min, omega_max=self.omega_max
        )

        segments = None
        triggered_labels: list[str] = []
        if self.damage_manager is not None:
            segments, triggered_labels = self._damage_segments(
                omega2_cmd, t_prev, t_now
            )
            omega2_eff = self.damage_manager.state.mu * omega2_cmd
            self.damage_state_log.append(
                {"time": float(t_now), "state": self.damage_manager.state.snapshot()}
            )
        else:
            omega2_eff = omega2_cmd
        u_virtual_eff = self.allocator.mix(omega2_eff)
        self.model.run_step(u_virtual_eff, control_segments=segments)
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

    def _record_damage_events(self, events) -> list[str]:
        """Invoke rotor-fault callbacks, append event records and return their display
        labels.
        """
        manager = self.damage_manager
        assert manager is not None
        labels = []
        for ev in events:
            if self.damage_event_callback:
                self.damage_event_callback(ev, manager.state)
            label = ev.label or type(ev).__name__
            labels.append(label)
            self.damage_events_log.append(
                {
                    "time": float(ev.trigger_time),
                    "label": label,
                    "rotor_id": ev.rotor_id,
                    "kind": type(ev).__name__,
                }
            )
        return labels

    def _damage_segments(self, omega2_cmd, start, end):
        """Build smooth control intervals while advancing damage chronologically."""
        manager = self.damage_manager
        assert manager is not None
        labels = self._record_damage_events(manager.update(start, start, 0.0))
        boundaries = sorted(
            {ev.trigger_time for ev in manager.pending_events(end, start)} | {end}
        )
        segments = []
        cursor = start
        for boundary in boundaries:
            duration = boundary - cursor
            if duration > 0:
                state = RotorDamageState(**manager.state.snapshot())

                def control(elapsed, state=state):
                    """Mix rotor commands with the effectiveness reached after the
                    segment elapsed time.
                    """
                    return self.allocator.mix(
                        state.effectiveness_after(elapsed) * omega2_cmd
                    )

                segments.append((duration, control))
            labels.extend(
                self._record_damage_events(manager.update(boundary, cursor, duration))
            )
            cursor = boundary
        return segments, labels
