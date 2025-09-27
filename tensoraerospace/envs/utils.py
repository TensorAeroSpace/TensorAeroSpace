"""Utilities for working with Gymnasium environments.

This module contains helper classes and functions for working with
reinforcement learning environments, including action normalization and other wrappers
to improve training performance.
"""

import gymnasium as gym
import numpy as np


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions.

    This wrapper normalizes actions from the range (-1, 1) to the actual
    action space bounds (low, high) of the environment.
    """

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high).

        Args:
            action (np.ndarray): Action in range (-1, 1).

        Returns:
            np.ndarray: Action in environment's action space range.
        """
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1).

        Args:
            action (np.ndarray): Action in environment's action space range.

        Returns:
            np.ndarray: Action in range (-1, 1).
        """
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)
        return action
