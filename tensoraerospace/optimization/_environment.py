"""Explicit physical-unit adapters for declarative tuning experiments."""

from __future__ import annotations

import copy
import importlib
import inspect
from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.signal import cont2discrete

from .metrics import TrialRejected

_KINDS = {
    "LinearLongitudinalF16": "linear",
    "LinearLongitudinalB747": "linear",
    "NonlinearLongitudinalF16": "f16_long",
    "NonlinearAngularF16": "f16_angular",
    "NonlinearB737Env": "boeing",
    "NonlinearB747Env": "boeing",
    "ImprovedB747Env": "hdp_b747",
}
_ANGLES = {
    "alpha",
    "beta",
    "theta",
    "phi",
    "psi",
    "gamma",
    "q",
    "p",
    "r",
    "wx",
    "wy",
    "wz",
    "stab",
    "dstab",
    "ail",
    "dail",
    "dir",
    "ddir",
}


def _resolve_environment(environment, kwargs):
    """Resolve a Gymnasium ID or environment class with copied constructor options."""
    env_class: Any
    if isinstance(environment, str):
        spec = gym.spec(environment)
        env_class = (
            gym.envs.registration.load_env_creator(spec.entry_point)
            if isinstance(spec.entry_point, str)
            else spec.entry_point
        )
        settings = {**copy.deepcopy(spec.kwargs), **copy.deepcopy(kwargs)}
    elif inspect.isclass(environment) and issubclass(environment, gym.Env):
        env_class, settings = environment, copy.deepcopy(kwargs)
    else:
        raise TypeError("env must be a registered Gymnasium id or environment class")
    return env_class, settings


class ExperimentEnvironment:
    """Store a validated physical environment specification and sampling horizon."""

    def __init__(self, environment, kwargs, duration, references):
        self.env_class: Any
        self.env_class, self.kwargs = _resolve_environment(environment, kwargs)
        if self.env_class.__name__ not in _KINDS:
            raise ValueError(
                f"No physical-state adapter for {self.env_class.__name__}; supported: {sorted(_KINDS)}"
            )
        self.kind = _KINDS[self.env_class.__name__]
        parameters = inspect.signature(self.env_class).parameters
        if "reference_signal" in self.kwargs or "number_time_steps" in self.kwargs:
            raise ValueError(
                "Use reference= and duration=; do not duplicate reference_signal/number_time_steps in env_kwargs"
            )
        default_dt = parameters["dt"].default if "dt" in parameters else 0.01
        self.dt = float(self.kwargs.get("dt", default_dt))
        if (
            not np.isfinite(self.dt)
            or self.dt <= 0
            or not np.isclose(duration / self.dt, round(duration / self.dt))
        ):
            raise ValueError(
                "Positive finite dt and duration must give an integer number of steps"
            )
        self.steps = round(duration / self.dt)
        if self.steps < 3:
            raise ValueError("At least three simulation steps are required")
        self._validate_options()
        count = self.steps if self.kind in ("boeing", "f16_angular") else self.steps + 1
        self.kwargs["number_time_steps"] = count
        if "reference_signal" in parameters:
            self.kwargs["reference_signal"] = np.zeros(
                (len(references), self.steps + 1)
            )
        if "tracking_states" in parameters:
            names = self.kwargs.get("state_space")
            if names is None:
                names = {
                    "LinearLongitudinalF16": ["alpha", "q"],
                    "NonlinearLongitudinalF16": ["alpha", "wz"],
                    "LinearLongitudinalB747": ["theta", "q"],
                }.get(self.env_class.__name__)
            if names is not None and not set(references) <= set(names):
                raise ValueError(
                    f"Unknown state in reference; choose from state_space={names}"
                )
            self.kwargs["tracking_states"] = list(references)
        if "use_reward" in parameters:
            self.kwargs.setdefault("use_reward", False)

    def _validate_options(self):
        """Reject environment modes that the physical-state adapters cannot represent."""
        if self.kind == "f16_long" and self.kwargs.get("damage_observable", False):
            raise ValueError(
                "The longitudinal F16 adapter requires physical state observations without damage metadata"
            )
        if self.kind == "f16_angular" and any(
            self.kwargs.get(k, False)
            for k in ("split_stab", "track_altitude", "damage_observable")
        ):
            raise ValueError(
                "The F16 angular adapter currently requires the standard 14-state, three-surface model"
            )
        if self.kind == "f16_long" and self.kwargs.get("feedforward_fn") is not None:
            raise ValueError(
                "Step tuning supports fixed control_bias; custom feedforward_fn needs the advanced ControlOptimizer API"
            )
        if (
            self.kind == "f16_angular"
            and self.kwargs.get("thrust_mode", "constant") != "constant"
        ):
            raise ValueError("The F16 angular tuning adapter requires constant thrust")
        if (
            self.kind == "boeing"
            and self.kwargs.get("action_space", "virtual") != "virtual"
        ):
            raise ValueError(
                "Boeing tuning uses action_space='virtual' (physical radians and throttle)"
            )

    @contextmanager
    def open(self, seed):
        """Yield a freshly seeded physical environment and always close it on exit."""
        env = self.env_class(**copy.deepcopy(self.kwargs))
        try:
            physical = PhysicalEnvironment(env, self.kind, self.dt)
            physical.reset(seed)
            yield physical
        finally:
            env.close()


class PhysicalEnvironment:
    """Expose native states, units, control limits and applied actions for tuning."""

    def __init__(self, env, kind, dt):
        self.env, self.kind, self.dt = env, kind, dt
        self.reference = None
        self.tracking = None

    def reset(self, seed):
        """Reset the plant, derive nominal trim controls and restore the reference."""
        observation, _ = self.env.reset(seed=seed)
        self.observation = np.asarray(observation).reshape(-1)
        self.model = self.env.model
        self._configure_state_units()
        actual_dt = getattr(
            self.env, "dt", getattr(self.model, "discretisation_time", self.dt)
        )
        if not np.isclose(actual_dt, self.dt):
            raise ValueError(
                "Requested dt does not match the environment's integration step"
            )
        self.initial = self.state.copy()
        self.bias = np.zeros(self.env.action_space.shape[0])
        if self.kind == "boeing":
            module = importlib.import_module(type(self.env).__module__)
            altitude, speed = (
                self.env.trim_at
                if hasattr(self.env, "trim_at")
                else (-self.state[11], np.linalg.norm(self.state[:3]))
            )
            result = module.trim(altitude, speed, config=self.env.config)
            if not result.converged:
                raise ValueError(
                    "Healthy trim action could not be obtained for this flight condition"
                )
            self.bias = np.array([result.elevator_rad, 0.0, 0.0, result.throttle])
        elif self.kind == "f16_angular":
            self.bias = np.rad2deg(self.state[[8, 10, 12]])
        self.low = np.asarray(self.env.action_space.low, dtype=float)
        self.high = np.asarray(self.env.action_space.high, dtype=float)
        if self.kind == "f16_long":
            self.low = np.maximum(
                self.low, -self.env.max_action_value - self.env.control_bias
            )
            self.high = np.minimum(
                self.high, self.env.max_action_value - self.env.control_bias
            )
        if (
            np.any(self.low >= self.high)
            or np.any(self.bias <= self.low)
            or np.any(self.bias >= self.high)
        ):
            raise ValueError(
                "Nominal action must lie inside the physical action limits"
            )
        self.scale = np.maximum(
            np.minimum(self.bias - self.low, self.high - self.bias), 1e-8
        )
        self.last_action = self.bias.copy()
        self.applied = self.last_action.copy()
        self.previous_state = self.state.copy()
        if self.reference is not None:
            self.set_reference(self.reference, self.tracking)

    @property
    def state(self):
        """Return the current physical state vector in the declared channel order."""
        if self.kind in ("boeing", "f16_angular"):
            return np.asarray(self.model.current_state).reshape(-1)
        if self.kind == "hdp_b747":
            return np.asarray(self.model.xt).reshape(-1)
        return self.observation

    def set_reference(self, reference, tracking):
        """Store a copied sample-by-channel reference and update the native environment."""
        self.reference, self.tracking = reference.copy(), tracking
        if hasattr(self.env, "reference_signal"):
            self.env.reference_signal[...] = reference.T

    def step(self, action):
        """Apply a clipped native action and retain actual actuator feedback.

        Return Gymnasium termination and truncation flags; rewards are not used by the
        tuning objective, which scores the completed physical trajectory.
        """
        self.previous_state = self.state.copy()
        action = np.clip(
            np.asarray(action, dtype=float).reshape(-1), self.low, self.high
        )
        self.observation, _, terminated, truncated, _ = self.env.step(action)
        self.observation = np.asarray(self.observation).reshape(-1)
        self.last_action = action.copy()
        if self.kind == "boeing":
            self.applied = np.asarray(self.model.applied_action).copy()
        elif self.kind == "f16_angular":
            # State-space agents identify the command channel, including servos.
            self.applied = np.rad2deg(np.asarray(self.model.u_history[-1]).reshape(-1))
        elif self.kind == "f16_long":
            # Feedback is expressed in the same residual degrees as env.step.
            self.applied = (
                np.rad2deg(np.asarray(self.model.u_history[-1]).reshape(-1))
                - self.env.control_bias
            )
        elif self.kind == "linear":
            k = self.model.time_step - 1
            self.applied = np.asarray(self.model.store_input[:, k]).copy()
        else:
            self.applied = action.copy()
        return terminated, truncated

    def _configure_state_units(self):
        """Declare channel names and native units for the selected model adapter."""
        if self.kind == "boeing":
            self.names = [
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
            self.units = ["ft/s"] * 3 + ["rad/s"] * 3 + ["rad"] * 3 + ["ft"] * 3
        elif self.kind == "f16_angular":
            self.names = [
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
            self.units = [
                "rad/s" if n in ("wx", "wy", "wz", "dstab", "dail", "ddir") else "rad"
                for n in self.names
            ]
        elif self.kind == "hdp_b747":
            self.names, self.units = ["u", "w", "q", "theta"], [
                "m/s",
                "m/s",
                "rad",
                "rad",
            ]
        else:
            self.names = list(self.env.state_space)
            self.units = [
                (
                    "rad/s"
                    if n in ("p", "q", "r", "wx", "wy", "wz", "dstab", "dail", "ddir")
                    else "rad" if n in _ANGLES else "deg" if n == "ele" else "native"
                )
                for n in self.names
            ]

    def validate_envelope(self):
        """Reject trajectories outside the adapter's declared small-maneuver limits."""
        state = self.state
        if self.kind == "boeing":
            if (
                abs(state[6]) > np.deg2rad(30)
                or abs(state[7]) > np.deg2rad(30)
                or not 250 < np.linalg.norm(state[:3]) < 1000
                or not 0 < -state[11] < 50000
            ):
                raise TrialRejected(
                    "Boeing left the declared small-maneuver flight envelope"
                )
        elif self.kind == "f16_angular":
            if max(abs(state[0]), abs(state[1])) > np.deg2rad(25) or np.max(
                abs(state[2:5])
            ) > np.deg2rad(60):
                raise TrialRejected("F16 left the angle/rate envelope")
        else:
            for i, name in enumerate(self.names):
                if name in ("alpha", "theta", "q", "wz") and abs(state[i]) > np.deg2rad(
                    30
                ):
                    raise TrialRejected(
                        f"State {name} left the 30-degree small-maneuver envelope"
                    )

    def nominal_discrete(self, indices):
        """Nominal Jacobian prior in physical observation / normalized-action units.

        Linear models use their discrete A/B. Boeing uses its public nominal
        continuous linearization and ZOH. F16 nonlinear models use central differences of a healthy one-step map.
        Unobserved states are held at trim in these local reduced priors.
        """
        if self.kind in ("linear", "hdp_b747"):
            all_names = list(getattr(self.model, "selected_states", self.names))
            rows = [all_names.index(n) for n in self.names]
            A = np.asarray(self.model.filt_A)[np.ix_(rows, rows)]
            B = np.asarray(self.model.filt_B)[rows][:, indices]
            return A, B * self.scale[indices]
        if self.kind == "boeing":
            model = type(self.model)(
                x0=self.initial.copy(), dt=self.dt, config=self.env.config
            )
            A, B = model.linearize(self.initial, self.bias)
            A, B, *_ = cont2discrete(
                (A, B[:, indices], np.eye(len(A)), np.zeros((len(A), len(indices)))),
                self.dt,
            )
            return A, B * self.scale[indices]
        if self.kind in ("f16_long", "f16_angular"):
            return self._nominal_f16_discrete(indices)
        raise ValueError("No nominal model adapter for this environment")

    def _nominal_f16_discrete(self, indices):
        """Linearize one undamaged F-16 integration step without mutating the live
        plant.
        """
        full_state = np.asarray(self.model.current_state).reshape(-1).copy()
        rows = [self.model.list_state.index(n) for n in self.names]
        parameters = copy.deepcopy(self.model.param)
        if hasattr(parameters, "damage_state"):
            parameters.damage_state = None
        if hasattr(parameters, "damage_geometry"):
            parameters.damage_geometry = None

        def advance(x, u):
            """Propagate a temporary healthy model with controls converted to radians."""
            model = type(self.model)(x0=x, dt=self.dt, integrator=self.env.integrator)
            model.set_param(copy.deepcopy(parameters))
            command = u.copy()
            if self.kind == "f16_long":
                command += self.env.control_bias
                if self.env.feedforward_fn is not None:
                    raise ValueError(
                        "Nominal-prior profiles do not support custom feedforward_fn"
                    )
            model.run_step(np.deg2rad(command))
            return np.asarray(model.current_state).reshape(-1)[rows]

        A = np.empty((len(rows), len(rows)))
        for j, row in enumerate(rows):
            dx = np.zeros_like(full_state)
            dx[row] = 1e-6
            A[:, j] = (
                advance(full_state + dx, self.bias)
                - advance(full_state - dx, self.bias)
            ) / (2e-6)
        B = np.empty((len(rows), len(indices)))
        for j, index in enumerate(indices):
            du = np.zeros_like(self.bias)
            du[index] = 1e-4
            B[:, j] = (
                advance(full_state, self.bias + du)
                - advance(full_state, self.bias - du)
            ) / (2e-4)
        return A, B * self.scale[indices]
