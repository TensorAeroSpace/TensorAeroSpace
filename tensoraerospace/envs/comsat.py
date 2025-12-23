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


class ComSatEnv(gym.Env):
    """Simulation of "Communication satellite in longitudinal control channel" control object in OpenAI Gym environment for training AI agents.

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

    def __init__(
        self,
        initial_state: np.ndarray | list[float],
        reference_signal: np.ndarray | Callable,
        number_time_steps: int,
        tracking_states: list[str] = ["rho_dot", "theta_dot"],
        state_space: tuple[float, float] = ["rho", "rho_dot", "theta_dot"],
        control_space: tuple[float, float] = ["u2"],
        output_space: tuple[float, float] = ["rho", "rho_dot", "theta_dot"],
        reward_func: Callable | None = None,
    ) -> None:
        """Initialize communication satellite environment."""
        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.selected_state_output = output_space
        self.tracking_states = tracking_states
        self.state_space = state_space
        self.control_space = control_space
        self.output_space = output_space
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = ComSat(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=output_space,
            t0=0,
        )
        self.indices_tracking_states = [
            state_space.index(tracking_states[i]) for i in range(len(tracking_states))
        ]

        self.ref_signal = reference_signal
        self.model.initialise_system(
            x0=initial_state, number_time_steps=number_time_steps
        )
        self.number_time_steps = number_time_steps

        self.action_space = spaces.Box(
            low=-60, high=60, shape=(len(control_space), 1), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(state_space), 1), dtype=np.float32
        )

        self.current_step = 0
        self.done = False

    def _get_info(self):
        """Return extra diagnostic info (none for now)."""
        return {}

    @staticmethod
    def reward(state, ref_signal, ts):
        """Compute tracking error used as reward."""
        return np.abs(state[0] - ref_signal[:, ts])

    def step(self, action: np.ndarray):
        """Run one environment step (Gymnasium API)."""
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = self.reward_func(
            next_state[self.indices_tracking_states],
            self.reference_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()

        return next_state.reshape([-1, 1]), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        """Reset environment to the initial state (Gymnasium API)."""
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False

        self.model = None
        self.model = ComSat(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
        )
        self.ref_signal = self.reference_signal
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )
        info = self._get_info()
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape([-1, 1])
        return observation, info

    def render(self):
        """Render the environment (not implemented)."""
        raise NotImplementedError()


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
        max_angular_velocity (float): Maximum angular velocity in rad/s.
        max_radial_position_deviation (float): Maximum radial deviation in km.
        max_thrust (float): Maximum tangential thrust magnitude.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_thrust: float = 0.0,
        use_initial_action_on_first_step: bool = True,
        nominal_rho: float = 6371.0,
    ):
        """Initialize ImprovedComSatEnv environment.

        Args:
            initial_state (np.ndarray): Initial state [rho, rho_dot, theta_dot]
                in SI units (km, m/s, rad/s).
            reference_signal (np.ndarray): Reference angular velocity
                trajectory in rad/s. Shape: (1, number_time_steps).
            number_time_steps (int): Total number of simulation steps.
            dt (float): Simulation time step in seconds. Defaults to 0.01.
            initial_thrust (float): Initial thrust value. Defaults to 0.0.
            use_initial_action_on_first_step (bool): If True, applies
                initial_thrust on first step. Defaults to True.
            nominal_rho (float): Nominal orbital radius in km.
                Defaults to 6371.0 (Earth radius).
        """
        super().__init__()

        # Normalization parameters and physical constraints
        # Increased to match actual dynamics range
        self.max_angular_velocity = 0.1  # rad/s (increased for stability)
        self.max_radial_velocity = 200.0  # m/s (increased)
        self.max_radial_position_deviation = 100.0  # km from nominal
        self.max_thrust = 25.0  # Maximum tangential thrust
        self.nominal_rho = float(nominal_rho)

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
        self.reference_signal = np.array(reference_signal, dtype=float)
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0
        # Full state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

        # Initial thrust (normalized)
        self.initial_thrust = float(initial_thrust)
        self.initial_action_norm = float(
            np.clip(self.initial_thrust / self.max_thrust, -1.0, 1.0)
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = 0.0
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

        # Model
        self.model = ComSat(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
        )
        self.model.initialise_system(
            x0=self.initial_state,
            number_time_steps=self.number_time_steps,
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
        self.model.initialise_system(self.initial_state, self.number_time_steps)
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
        action = np.clip(action, -1.0, 1.0)

        # Scale from [-1, 1] to thrust range
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_thrust = np.array([self.initial_thrust], dtype=np.float32)
        else:
            scaled_thrust = action * self.max_thrust

        # Simulation step
        self.state = self.model.run_step(scaled_thrust).reshape(-1)
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
            np.asarray(scaled_thrust).reshape(-1)[0] / self.max_thrust
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
        self._last_reward = float(reward)

        # Termination conditions with scaled penalties
        terminated = False
        # Excessive angular velocity (very lenient for unstable dynamics)
        if abs(theta_dot) > 50.0 * self.max_angular_velocity:  # 0.5 rad/s
            reward = -10.0  # Reduced penalty (was -100)
            terminated = True
        # Excessive radial deviation (orbit instability)
        if (
            abs(rho - self.nominal_rho) > 5.0 * self.max_radial_position_deviation
        ):  # 2500 km
            reward = -10.0  # Reduced penalty
            terminated = True
        # Excessive radial velocity
        if abs(rho_dot) > 10.0 * self.max_radial_velocity:  # 1000 m/s
            reward = -10.0  # Reduced penalty
            terminated = True

        truncated = self.current_step >= self.number_time_steps - 2

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    def render(self):
        """Render the environment (not implemented).

        Raises:
            NotImplementedError: Rendering not yet implemented.
        """
        raise NotImplementedError()
