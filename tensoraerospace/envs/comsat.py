"""Module for modeling communication satellite in longitudinal control channel.

This module contains a Gymnasium environment implementation for training agents
to control a communication satellite. The environment provides an interface
for interaction with the satellite model, including control of radial position,
radial velocity and angular velocity through tangential thrust.
"""

from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import ComSat
from tensoraerospace.envs._rendering import telemetry_render, validate_render_mode


class ComSatEnv(gym.Env):
    """Gymnasium environment for normalized ComSat state deviations.

    Args:
        initial_state: Initial state.
        reference_signal: Reference signal.
        number_time_steps: Number of simulation steps.
        tracking_states: Tracked states.
        state_space: State space.
        control_space: Control space.
        output_space: Full output space (including noise).
        reward_func: Reward function (WIP status).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        initial_state: np.ndarray | list[float],
        reference_signal: np.ndarray | Callable,
        number_time_steps: int,
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: Callable | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize communication satellite environment."""
        validate_render_mode(render_mode, self.metadata["render_modes"])
        super().__init__()
        self.render_mode = render_mode
        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["rho_dot", "theta_dot"]
        )
        self.state_space = (
            state_space if state_space is not None else ["rho", "rho_dot", "theta_dot"]
        )
        self.control_space = control_space if control_space is not None else ["u2"]
        self.output_space = (
            output_space
            if output_space is not None
            else ["rho", "rho_dot", "theta_dot"]
        )
        self.selected_state_output = self.output_space
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        # Constructor already invokes initialise_system internally.
        self.model = ComSat(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
        )
        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.ref_signal = reference_signal
        self.number_time_steps = number_time_steps

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
        self._last_observation = np.array(initial_state, dtype=np.float32).reshape(-1)
        self._last_action: np.ndarray | None = None
        self._last_reward: float | None = None

    def _get_info(self):
        """Return extra diagnostic info (none for now)."""
        return {}

    @staticmethod
    def reward(state, ref_signal, ts):
        """Compute tracking reward (negative absolute error)."""
        ts_safe = int(np.clip(ts, 0, ref_signal.shape[1] - 1))
        return -float(np.abs(state[0] - ref_signal[:, ts_safe]).item())

    def step(self, action: np.ndarray):
        """Run one environment step (Gymnasium API)."""
        action = np.asarray(action).reshape(-1)
        next_state = self.model.run_step(action)
        self.current_step += 1
        reward = self.reward_func(
            next_state[self.indices_tracking_states],
            self.reference_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()
        observation = np.asarray(next_state).reshape(-1).astype(np.float32)
        self._last_observation = observation
        self._last_action = self.model.store_input[:, self.model.time_step - 1].astype(
            np.float32
        )
        self._last_reward = float(reward)

        return (
            observation,
            reward,
            False,
            self.done,
            info,
        )

    def reset(self, seed=None, options=None):
        """Reset environment to the initial state (Gymnasium API)."""
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False

        # Constructor already invokes initialise_system internally.
        self.model = ComSat(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
        )
        self.ref_signal = self.reference_signal
        info = self._get_info()
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape(-1)
        self._last_observation = observation
        self._last_action = None
        self._last_reward = None
        return observation, info

    def render(self, mode: str | None = None):
        """Render a lightweight telemetry snapshot.

        The legacy ComSat environment does not ship a graphical viewer. Human
        mode prints one concise state line; ``ansi`` returns it as a string for
        tests and logging.
        """
        selected_mode = self.render_mode if mode is None else mode
        return telemetry_render(
            "ComSatEnv",
            selected_mode,
            step=self.current_step,
            total_steps=self.number_time_steps,
            state=self._last_observation,
            action=self._last_action,
            reward=self._last_reward,
        )


class ImprovedComSatEnv(gym.Env):
    """Improved communication satellite environment with normalized spaces.

    This environment provides:
        - Normalized action space [-1, 1] for tangential thrust u2
        - Normalized observation space for better RL training
        - LQR-style reward function with multiple objectives:
            * Angular velocity tracking (theta_dot)
            * Orbital radius stabilization (rho)
            * Energy efficiency (minimize thrust)
            * Control smoothness
        - Realistic termination conditions

    Attributes:
        action_space (spaces.Box): Normalized action space [-1, 1].
        observation_space (spaces.Box): Normalized observation space.
        max_angular_velocity (float): Angular-rate normalization scale per tau.
        max_radial_position_deviation (float): Normalized radial-deviation scale.
        max_thrust (float): Maximum tangential thrust magnitude.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_thrust: float = 0.0,
        use_initial_action_on_first_step: bool = True,
        nominal_rho: float = 6371.0,
        render_mode: str | None = None,
    ):
        """Initialize ImprovedComSatEnv environment.

        Args:
            initial_state (np.ndarray): Initial state [rho, rho_dot, theta_dot]
                as [nominal_rho + delta_rho, delta_rho_prime, delta_theta_prime].
                Dynamics use dimensionless perturbations, not SI units.
            reference_signal (np.ndarray): Reference angular velocity
                deviation trajectory per normalized time tau. Shape: (1, number_time_steps).
            number_time_steps (int): Total number of simulation steps.
            dt (float): Step in normalized time tau. Defaults to 0.01.
            initial_thrust (float): Initial thrust value. Defaults to 0.0.
            use_initial_action_on_first_step (bool): If True, applies
                initial_thrust on first step. Defaults to True.
            nominal_rho (float): External coordinate offset subtracted before
                linear dynamics. The legacy default 6371.0 is retained as an
                offset only; it is not an Earth radius in km. Use 0 for deviations.
            render_mode (str | None): ``None``, ``"human"`` or ``"ansi"``.
        """
        validate_render_mode(render_mode, self.metadata["render_modes"])
        super().__init__()
        self.render_mode = render_mode

        # Normalization parameters and physical constraints
        # Increased to match actual dynamics range
        self.max_angular_velocity = 0.1  # normalized angular rate
        self.max_radial_velocity = 200.0  # normalized radial velocity
        self.max_radial_position_deviation = 100.0  # normalized radial displacement
        self.max_thrust = 25.0  # normalized input limit (legacy simulation setting)
        self.nominal_rho = float(nominal_rho)
        if not np.isfinite(self.nominal_rho):
            raise ValueError("nominal_rho must be finite")

        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [norm_theta_dot_error, norm_rho_error,
        #               norm_rho_dot, norm_prev_action]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Simulation parameters
        self.dt = dt
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        if self.initial_state.size != 3 or not np.all(np.isfinite(self.initial_state)):
            raise ValueError("initial_state must contain three finite values")
        self.reference_signal = np.array(reference_signal, dtype=float)
        if (
            self.reference_signal.ndim != 2
            or self.reference_signal.shape[0] != 1
            or self.reference_signal.shape[1] == 0
            or not np.all(np.isfinite(self.reference_signal))
        ):
            raise ValueError(
                "reference_signal must be finite with shape (1, T), T >= 1"
            )
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0
        # External state: [nominal_rho + delta_rho, delta_rho_prime, delta_theta_prime]
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

        # Initial thrust (normalized)
        self.initial_thrust = float(initial_thrust)
        self.initial_action_norm = float(
            np.clip(self.initial_thrust / self.max_thrust, -1.0, 1.0)
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0

        # Reward scale for Q-value stability
        self.reward_scale = 0.1  # Balanced for gradient signal

        # Cost function weights (tunable) - scaled for stable training
        self.w_theta_dot = 5.0  # Angular velocity tracking (primary)
        self.w_rho = 0.01  # Orbital radius stabilization (low weight)
        self.w_rho_dot = 0.2  # Radial velocity damping
        self.w_action = 0.001  # Energy cost (minimize thrust)
        self.w_smooth = 0.01  # Control smoothness
        self.w_jerk = 0.001  # Jitter suppression

        # Store initialization arguments
        self.init_args = locals()

        # Only deviations enter the linearization. nominal_rho is an external
        # coordinate offset, never a force-producing model state.
        self._state_offset = np.array([self.nominal_rho, 0.0, 0.0])
        self.model = ComSat(
            self.initial_state - self._state_offset,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
            initial_control=self.initial_thrust,
        )

    # State indices
    @property
    def _idx_rho(self) -> int:
        """Index of radial position in state vector."""
        return 0

    @property
    def _idx_rho_dot(self) -> int:
        """Index of radial velocity in state vector."""
        return 1

    @property
    def _idx_theta_dot(self) -> int:
        """Index of angular velocity in state vector."""
        return 2

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation.

        Returns:
            np.ndarray: Array of shape (4,), dtype float32:
                [norm_theta_dot_error, norm_rho_error,
                 norm_rho_dot, norm_prev_action]
        """
        rho = float(self.state[self._idx_rho])
        rho_dot = float(self.state[self._idx_rho_dot])
        theta_dot = float(self.state[self._idx_theta_dot])

        # Target angular velocity for current step
        idx_safe = int(
            np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta_dot = float(self.reference_signal[0, idx_safe])

        # Normalized observations
        theta_dot_error_norm = float(
            (theta_dot - target_theta_dot) / self.max_angular_velocity
        )
        rho_error_norm = float(
            (rho - self.nominal_rho) / self.max_radial_position_deviation
        )
        rho_dot_norm = float(rho_dot / self.max_radial_velocity)
        prev_action_norm = float(self.previous_action)

        obs = np.array(
            [
                theta_dot_error_norm,
                rho_error_norm,
                rho_dot_norm,
                prev_action_norm,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def get_init_args(self) -> Dict[str, Any]:
        """Get initialization arguments as a dictionary.

        Returns:
            dict: Dictionary of initialization arguments.
        """
        init_args = self.init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        return init_args

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional reset options.

        Returns:
            tuple: Initial observation and empty info dict.
        """
        super().reset(seed=seed)
        self.model.initialise_system(
            self.initial_state - self._state_offset, self.number_time_steps
        )
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step.

        Args:
            action (np.ndarray): Normalized action in range [-1, 1].

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        # Convert and clip action
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1 or not np.all(np.isfinite(action)):
            raise ValueError("action must contain one finite value")
        action = np.clip(action, -1.0, 1.0)

        # Scale from [-1, 1] to thrust range
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_thrust = np.array([self.initial_thrust], dtype=np.float32)
        else:
            scaled_thrust = action * self.max_thrust

        # Simulation step
        self.state = self.model.run_step(scaled_thrust).reshape(-1) + self._state_offset
        self.current_step += 1

        # Get current state values
        rho = float(self.state[self._idx_rho])
        rho_dot = float(self.state[self._idx_rho_dot])
        theta_dot = float(self.state[self._idx_theta_dot])

        # Target angular velocity
        idx_safe = int(
            np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta_dot = float(self.reference_signal[0, idx_safe])

        # Note: Could use reference theta_dot derivative for advanced damping
        # Currently using direct state feedback instead

        # Normalized errors for LQR-style cost
        e_theta_dot = float((theta_dot - target_theta_dot) / self.max_angular_velocity)
        e_rho = float((rho - self.nominal_rho) / self.max_radial_position_deviation)
        e_rho_dot = float(rho_dot / self.max_radial_velocity)

        # Normalized applied action
        u_applied_norm = float(
            self.model.store_input[0, self.model.time_step - 1] / self.max_thrust
        )
        u = u_applied_norm
        du = u_applied_norm - float(self.previous_action)
        ddu = (
            u_applied_norm
            - 2.0 * float(self.previous_action)
            + float(self.pre_previous_action)
        )

        # Quadratic cost (LQR-like)
        cost = (
            self.w_theta_dot * (e_theta_dot**2)
            + self.w_rho * (e_rho**2)
            + self.w_rho_dot * (e_rho_dot**2)
            + self.w_action * (u**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )

        reward = float(-cost)
        reward *= float(self.reward_scale)

        # Add survival bonus to encourage longer episodes
        reward += 0.1  # Small bonus for each successful step

        # Update action history
        self.pre_previous_action = float(self.previous_action)
        self.previous_action = float(u_applied_norm)
        # Termination conditions with scaled penalties
        terminated = False
        # Numerical bounds in normalized perturbation coordinates
        if abs(theta_dot) > 50.0 * self.max_angular_velocity:
            reward = -10.0  # Reduced penalty (was -100)
            terminated = True
        # Radial displacement bound
        if (
            abs(rho - self.nominal_rho) > 5.0 * self.max_radial_position_deviation
        ):  # normalized radial displacement
            reward = -10.0  # Reduced penalty
            terminated = True
        # Excessive radial velocity
        if abs(rho_dot) > 10.0 * self.max_radial_velocity:
            reward = -10.0  # Reduced penalty
            terminated = True

        truncated = self.current_step >= self.number_time_steps - 1
        self._last_reward = float(reward)

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    def render(self, mode: str | None = None):
        """Render a lightweight telemetry snapshot."""
        selected_mode = self.render_mode if mode is None else mode
        return telemetry_render(
            "ImprovedComSatEnv",
            selected_mode,
            step=self.current_step,
            total_steps=self.number_time_steps,
            state=self.state,
            action=np.array([self.previous_action], dtype=np.float32),
            reward=self._last_reward,
        )
