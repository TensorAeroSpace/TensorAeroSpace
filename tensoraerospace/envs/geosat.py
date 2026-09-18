"""Geostationary satellite Gymnasium environment.

This module provides a Gymnasium-compatible environment for controlling a
geosationary satellite model (GeoSat). It is intended for RL training and
benchmarking of control algorithms.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import GeoSat


class GeoSatEnv(gym.Env):
    """Gymnasium environment for geostationary satellite control.

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
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: Callable | None = None,
        dt: float = 0.01,
    ) -> None:
        """Initialize geosationary satellite environment."""
        super().__init__()
        self.initial_state = np.array(
            initial_state, dtype=np.float64, copy=True
        ).reshape(-1)
        self.dt = dt
        self.number_time_steps = number_time_steps
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["theta", "omega"]
        )
        self.state_space = (
            state_space if state_space is not None else ["rho", "theta", "omega"]
        )
        self.control_space = control_space if control_space is not None else ["thrust"]
        self.output_space = (
            output_space if output_space is not None else list(self.state_space)
        )
        self.selected_state_output = self.output_space
        if callable(reference_signal):
            reference_signal = np.array(
                [
                    np.atleast_1d(reference_signal(i * dt))
                    for i in range(number_time_steps)
                ]
            ).T
        self.reference_signal = np.array(reference_signal, dtype=np.float64, copy=True)
        if (
            self.reference_signal.ndim != 2
            or self.reference_signal.shape[1] < 1
            or not np.all(np.isfinite(self.reference_signal))
        ):
            raise ValueError(
                "reference_signal must be finite with shape (channels, T), T >= 1"
            )
        if self.reference_signal.shape[0] not in (1, len(self.tracking_states)):
            raise ValueError("reference channels must be one or match tracking_states")
        if len(self.control_space) != 1:
            raise ValueError("GeoSat supports one tangential thrust input")
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        # Constructor already invokes initialise_system internally.
        self.model = GeoSat(
            self.initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.indices_tracking_states = [
            self.model.list_state.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.ref_signal = self.reference_signal
        self.number_time_steps = number_time_steps

        self.action_space = spaces.Box(
            low=-np.array(self.model.input_magnitude_limits, dtype=np.float32),
            high=np.array(self.model.input_magnitude_limits, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.output_space),),
            dtype=np.float32,
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
            float: Control evaluation reward (negative absolute error).
        """
        ts_safe = int(np.clip(ts, 0, ref_signal.shape[1] - 1))
        reference = ref_signal[:, ts_safe]
        tracked = np.asarray(state).reshape(-1)
        # One reference channel keeps the legacy first-tracked-state objective.
        error = tracked[:1] - reference if reference.size == 1 else tracked - reference
        return -float(np.mean(np.abs(error)))

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
        action = np.asarray(action).reshape(-1)
        next_state = self.model.run_step(action)
        self.current_step += 1
        reward = self.reward_func(
            np.asarray(self.model.xt).reshape(-1, 1)[self.indices_tracking_states],
            self.reference_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()

        return (
            np.asarray(next_state).astype(np.float32).reshape(-1),
            float(reward),
            False,
            self.done,
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

        # Constructor already invokes initialise_system internally.
        self.model = GeoSat(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.ref_signal = self.reference_signal
        self.current_step = 0
        self.done = False
        info = self._get_info()
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape(-1)
        return observation, info

    def render(self) -> None:
        """Visual rendering of actions in the environment. Work in progress.

        Raises:
            NotImplementedError: Rendering is not yet implemented.
        """
        raise NotImplementedError("Rendering is not implemented for GeoSatEnv.")
