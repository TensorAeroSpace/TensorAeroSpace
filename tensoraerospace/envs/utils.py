"""Utilities for working with Gymnasium environments.

This module contains helper classes and functions for working with
reinforcement learning environments, including action normalization and other wrappers
to improve training performance.
"""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions.

    This wrapper normalizes actions from the range (-1, 1) to the actual
    action space bounds (low, high) of the environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, Box):
            raise TypeError("ActionNormalizer requires a Box action space")
        self._action_low = env.action_space.low.copy()
        self._action_high = env.action_space.high.copy()
        if not (
            np.all(np.isfinite(self._action_low))
            and np.all(np.isfinite(self._action_high))
            and np.all(self._action_high > self._action_low)
        ):
            raise ValueError("ActionNormalizer requires finite, nonzero action ranges")
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=cast(
                type[np.floating[Any]] | type[np.integer[Any]],
                np.dtype(env.action_space.dtype).type,
            ),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high).

        Args:
            action (np.ndarray): Action in range (-1, 1).

        Returns:
            np.ndarray: Action in environment's action space range.
        """
        if not isinstance(self.action_space, Box):
            raise TypeError("ActionNormalizer requires a Box action space")
        low = self._action_low
        high = self._action_high

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
        if not isinstance(self.action_space, Box):
            raise TypeError("ActionNormalizer requires a Box action space")
        low = self._action_low
        high = self._action_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)
        return action
