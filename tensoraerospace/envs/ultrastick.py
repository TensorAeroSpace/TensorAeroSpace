"""
Module for modeling Ultrastick-25e aircraft.

This module contains a Gymnasium environment implementation for training agents
to control longitudinal motion of Ultrastick-25e aircraft. The environment provides an interface
for interaction with the aircraft model, including control of pitch angle and angular
velocity through stabilizers.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import Ultrastick


class LinearLongitudinalUltrastick(gym.Env):
    """Simulation of Ultrastick-25e control object in OpenAI Gym environment for training AI agents.

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
        tracking_states: list[str] = ["theta", "q"],
        state_space: tuple[float, float] = ["theta", "q"],
        control_space: tuple[float, float] = ["stab"],
        output_space: tuple[float, float] = ["theta", "q"],
        reward_func: Callable | None = None,
    ) -> None:
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

        self.model = Ultrastick(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=output_space,
            t0=0,
        )
        self.indices_tracking_states = [
            state_space.index(tracking_states[i]) for i in range(len(tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-25, high=25, shape=(len(control_space), 1), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(state_space), 1), dtype=np.float32
        )

        self.ref_signal = reference_signal
        self.model.initialise_system(
            x0=initial_state, number_time_steps=number_time_steps
        )
        self.number_time_steps = number_time_steps
        self.current_step = 0
        self.done = False

    @staticmethod
    def reward(state: np.ndarray, ref_signal: np.ndarray, ts: int) -> np.ndarray:
        """Control evaluation.

        Args:
            state: Current state.
            ref_signal: Reference state.
            ts: Time step.

        Returns:
            reward (float): Control evaluation.
        """
        return np.abs(state[0] - ref_signal[:, ts])

    def _get_info(self):
        return {}

    def step(self, action: np.ndarray):
        """Execute simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators.

        Returns:
            next_state (np.ndarray): Next state of control object.
            reward (np.ndarray): Evaluation of control algorithm actions.
            done (bool): Simulation status, completed or not.
            logging (any): Additional information (not used).
        """
        if action[0] > self.max_action_value:
            action[0] = self.max_action_value
        if action[0] < self.max_action_value * -1:
            action[0] = self.max_action_value * -1
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = self.reward_func(
            next_state[self.indices_tracking_states], self.ref_signal, self.current_step
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
        self.model = None
        self.model = Ultrastick(
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
        """Visual display of actions in environment. WIP status. Raises:
        NotImplementedError: Rendering is not implemented.
        """
        raise NotImplementedError()
