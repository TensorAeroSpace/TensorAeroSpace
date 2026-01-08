"""Module for modeling longitudinal motion of rocket/guided missile.

This module contains a Gymnasium environment implementation for
training agents to control longitudinal motion of rocket. The
environment provides an interface for interaction with the rocket
model, including control of pitch angle and pitch angular velocity
through stabilizers.
"""

from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import MissileModel


class LinearLongitudinalMissileModel(gym.Env):
    """Simulation of MissileModel in Gym for training AI agents.

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
        initial_state: Union[np.ndarray, list[float]],
        reference_signal: Union[np.ndarray, Callable],
        number_time_steps: int,
        tracking_states: Optional[list[str]] = None,
        state_space: Optional[list[str]] = None,
        control_space: Optional[list[str]] = None,
        output_space: Optional[list[str]] = None,
        reward_func: Optional[Callable] = None,
    ) -> None:
        """Initialize linear missile model environment."""
        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.selected_state_output = (
            output_space if output_space is not None else ["theta", "q"]
        )
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["theta", "q"]
        )
        self.state_space = state_space if state_space is not None else ["theta", "q"]
        self.control_space = control_space if control_space is not None else ["stab"]
        self.output_space = output_space if output_space is not None else ["theta", "q"]
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = MissileModel(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=None,
            t0=0,
        )
        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.ref_signal = reference_signal
        self.model.initialise_system(
            x0=initial_state, number_time_steps=number_time_steps
        )
        self.number_time_steps = number_time_steps

        self.action_space = spaces.Box(
            low=-60,
            high=60,
            shape=(len(self.control_space), 1),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_space), 1),
            dtype=np.float32,
        )

        self.current_step = 0
        self.done = False

    def _get_info(self):
        """Return auxiliary info for Gym API (currently empty)."""
        return {}

    @staticmethod
    def reward(state, ref_signal, ts):
        """Evaluate control performance.

        Args:
            state (_type_): Current state.
            ref_signal (_type_): Reference state.
            ts (_type_): Time step.

        Returns:
            reward (float): Control performance evaluation.
        """
        return np.abs(state[0] - ref_signal[:, ts])

    def step(self, action: np.ndarray):
        """Execute a simulation step.

        Args:
            action (np.ndarray): Array of control signals for selected
                control surfaces.

        Returns:
            next_state (np.ndarray): Next state of the control object.
            reward (np.ndarray): Evaluation of control algorithm actions.
            done (bool): Simulation status, whether completed or not.
            logging (any): Additional information (not used).
        """
        if action[0] > self.max_action_value:
            action[0] = self.max_action_value
        if action[0] < self.max_action_value * -1:
            action[0] = self.max_action_value * -1
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
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional options for initialization.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.model = MissileModel(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
        )
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )
        info = self._get_info()

        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape([-1, 1])
        return observation, info

    def render(self):
        """Visual display of actions in the environment. Status: WIP.

        Raises:
            NotImplementedError: Rendering is not implemented.
        """
        raise NotImplementedError("Rendering is not implemented for RocketEnv.")


class ImprovedMissileEnv(gym.Env):
    """Improved missile longitudinal control environment.

    RL-friendly API with normalized spaces and shaped reward.

    Observation (shape: (4,)):
        [norm_pitch_error, norm_q, norm_theta, norm_prev_action]

    Action (shape: (1,)):
        Normalized elevator command in [-1, 1]. Internally scaled to degrees
        and then converted to radians for the model.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_elevator_deg: float = 0.0,
        use_initial_action_on_first_step: bool = True,
    ) -> None:
        """Initialize improved missile environment with normalized action/obs."""
        super().__init__()

        # Physical/normalization limits
        self.max_pitch_rad = np.deg2rad(20.0)  # |theta| <= 20 deg
        self.max_pitch_rate_rad_s = np.deg2rad(5.0)  # |q| <= 5 deg/s
        self.max_elevator_angle_deg = 25.0  # |ele| <= 25 deg

        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Simulation parameters
        self.dt = float(dt)
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        self.reference_signal = np.array(reference_signal, dtype=float)
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

        # Initial elevator and action history (normalized)
        self.initial_elevator_deg = float(initial_elevator_deg)
        self.initial_action_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_elevator_angle_deg,
                -1.0,
                1.0,
            )
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0

        # Reward shaping
        self.reward_scale = 0.1
        self.w_pitch = 8.0
        self.w_q = 0.2
        self.w_action = 0.003
        self.w_smooth = 0.01
        self.w_jerk = 0.001

        # Store init args for (de)serialization helpers
        self.init_args = locals()

        # Underlying model (keep full state output order: [u, w, q, theta])
        self.model = MissileModel(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
        )
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )

    @property
    def _idx_q(self) -> int:
        """Index of pitch rate state."""
        return 2  # [u, w, q, theta]

    @property
    def _idx_theta(self) -> int:
        """Index of pitch angle state."""
        return 3  # [u, w, q, theta]

    def _get_obs(self) -> np.ndarray:
        """Return normalized observation vector."""
        theta = float(self.state[self._idx_theta])
        q = float(self.state[self._idx_q])
        idx = int(np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1))
        target_theta = float(self.reference_signal[0, idx])

        pitch_error = target_theta - theta
        norm_pitch_error = float(np.clip(pitch_error / self.max_pitch_rad, -1.0, 1.0))
        norm_q = float(np.clip(q / self.max_pitch_rate_rad_s, -1.0, 1.0))
        norm_theta = float(np.clip(theta / self.max_pitch_rad, -1.0, 1.0))
        norm_prev_action = float(self.previous_action)

        return np.array(
            [norm_pitch_error, norm_q, norm_theta, norm_prev_action],
            dtype=np.float32,
        )

    def get_init_args(self):
        """Return initialization arguments for reproducibility."""
        init_args = self.init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        return init_args

    def reset(self, seed=None, options=None):
        """Reset environment state and counters."""
        super().reset(seed=seed)
        self.model.initialise_system(self.initial_state, self.number_time_steps)
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Apply normalized action and advance simulation by one step."""
        # action in [-1, 1]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Scale to degrees, optionally use initial elevator at first step
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_action_deg = np.array([self.initial_elevator_deg], dtype=np.float32)
        else:
            scaled_action_deg = action * self.max_elevator_angle_deg
        # Convert to radians for the model
        scaled_action_rad = np.deg2rad(scaled_action_deg)

        # Simulate one step
        self.state = self.model.run_step(scaled_action_rad).reshape(-1)
        self.current_step += 1

        # Reward
        theta = float(self.state[self._idx_theta])
        q = float(self.state[self._idx_q])
        idx_safe = int(
            np.clip(
                self.current_step,
                0,
                self.reference_signal.shape[1] - 1,
            )
        )
        target_theta = float(self.reference_signal[0, idx_safe])

        if self.current_step > 0:
            idx_prev = int(
                np.clip(
                    self.current_step - 1,
                    0,
                    self.reference_signal.shape[1] - 1,
                )
            )
            ref_theta_prev = float(self.reference_signal[0, idx_prev])
        else:
            ref_theta_prev = target_theta
        ref_theta_dot = float((target_theta - ref_theta_prev) / self.dt)

        e_theta = float((theta - target_theta) / self.max_pitch_rad)
        e_q_rel = float((q - ref_theta_dot) / self.max_pitch_rate_rad_s)

        u_applied_norm = float(
            np.asarray(scaled_action_deg).reshape(-1)[0] / self.max_elevator_angle_deg
        )
        u = u_applied_norm
        du = u_applied_norm - float(self.previous_action)
        ddu = (
            u_applied_norm
            - 2.0 * float(self.previous_action)
            + float(self.pre_previous_action)
        )

        cost = (
            self.w_pitch * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action * (u**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )
        reward = float(-cost) * float(self.reward_scale)

        self.pre_previous_action = float(self.previous_action)
        self.previous_action = float(u_applied_norm)
        self._last_reward = float(reward)

        terminated = False
        if abs(theta) > self.max_pitch_rad:
            reward = -100.0
            terminated = True

        truncated = self.current_step >= self.number_time_steps - 2

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    def render(self, mode: str = "human"):
        """Rendering not implemented for missile environment."""
        return

    def close(self):
        """Close environment resources."""
        return
