"""Geostationary satellite Gymnasium environment.

This module provides a Gymnasium-compatible environment for controlling a
geostationary satellite model (GeoSat). It is intended for RL training and
benchmarking of control algorithms.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import GeoSat


class GeoSatEnv(gym.Env):
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
        tracking_states: list[str] = ["theta", "omega"],
        state_space: tuple[float, float] = ["rho", "theta", "omega"],
        control_space: tuple[float, float] = ["thrust"],
        output_space: tuple[float, float] = ["rho", "theta", "omega"],
        reward_func: Callable | None = None,
    ) -> None:
        """Initialize geostationary satellite environment."""
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

        self.model = GeoSat(
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
            low=-1000.0, high=1000.0, shape=(len(state_space), 1), dtype=np.float32
        )

        self.current_step = 0
        self.done = False

    def _get_info(self) -> dict[str, float]:
        """Return auxiliary info for Gym API (currently empty)."""
        return {}

    @staticmethod
    def reward(state: np.ndarray, ref_signal: np.ndarray, ts: int) -> float:
        """Evaluate control performance.

        Args:
            state (np.ndarray): Current state.
            ref_signal (np.ndarray): Reference signal.
            ts (int): Time step.

        Returns:
            float: Control evaluation reward.
        """
        return float(np.abs(state[0] - ref_signal[:, ts]).item())

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        """Execute one simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators.

        Returns:
            tuple: Tuple containing:
                - next_state (np.ndarray): Next state of the control object.
                - reward (np.ndarray): Evaluation of control algorithm actions.
                - done (bool): Simulation status, whether completed or not.
                - truncated (bool): Whether episode was truncated.
                - info (dict): Additional information.
        """
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = self.reward_func(
            next_state[self.indices_tracking_states],
            self.reference_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()

        return (
            next_state.astype(np.float32).reshape([-1, 1]),
            float(reward),
            self.done,
            False,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Additional initialization options. Defaults to None.

        Returns:
            tuple: Tuple containing:
                - observation (np.ndarray): Initial observation.
                - info (dict): Additional information.
        """
        super().reset(seed=seed)

        self.model = None
        self.model = GeoSat(
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
        self.current_step = 0
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape([-1, 1])
        return observation, info

    def render(self) -> None:
        """Visual rendering of actions in the environment. Work in progress.

        Raises:
            NotImplementedError: Rendering is not yet implemented.
        """
        raise NotImplementedError("Rendering is not implemented for GeoSatEnv.")
