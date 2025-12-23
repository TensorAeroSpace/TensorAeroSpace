"""Unity ML-Agents integration helpers.

This module provides lightweight wrappers to make Unity-based environments
compatible with common RL algorithms and Gymnasium-style APIs.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class unity_discrete_env(gym.Wrapper):
    """Discrete-action wrapper for Unity environments.

    The wrapped Unity environment expects a 7-dimensional continuous action
    vector with each component in ``{-1, 0, 1}``. This wrapper exposes a single
    discrete action in ``[0, 3**7)`` and decodes it into that 7D vector.
    """

    def __init__(self, env):
        """Wrap Unity environment exposing a discrete action space."""
        super().__init__(env)
        self.action_space = Discrete(3**7)
        self.env = env

    def reset(self):
        """Reset the underlying environment.

        Returns:
            Any: First observation returned by the wrapped Unity environment.
        """
        return self.env.reset()

    def step(self, action):
        """Convert a discrete action into a 7D continuous action and step.

        Args:
            action (int): Discrete action index in ``[0, 3**7)``.

        Returns:
            Any: Transition tuple returned by the wrapped environment.
        """
        actions = np.array([0.0] * 7, dtype=np.float32)
        for i in range(7):
            actions[i] = (action // 3**i) % 3 - 1.0
        return self.env.step(actions)

    def close(self):
        """Close the underlying Unity environment."""
        self.env.close()
