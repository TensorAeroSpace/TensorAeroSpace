"""Unity ML-Agents integration helpers.

This module provides lightweight wrappers to make Unity-based environments
compatible with common RL algorithms and Gymnasium-style APIs.
"""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


def get_plane_env(
    env_path: Optional[str] = "",
    server: bool = False,
    worker: int = 0,
    log_dir: str = "/app/logs",
    additional_args: Optional[list[str]] = None,
):
    """Create a Gym wrapper for a Unity airplane environment.

    Args:
        env_path: Path to a built Unity binary. ``None`` or ``""`` connects
            to a running Unity Editor instance.
        server: If ``True``, run headless (no graphics).
        worker: Unique worker ID to avoid port conflicts when running
            multiple copies in parallel.
        log_dir: Directory where Unity writes log files.
        additional_args: Extra CLI arguments forwarded to the Unity process.

    Returns:
        A Gym-compatible environment wrapping the Unity environment.
    """
    from mlagents_envs.environment import (
        UnityEnvironment,  # type: ignore[import-untyped]
    )
    from mlagents_envs.envs.unity_gym_env import (
        UnityToGymWrapper,  # type: ignore[import-untyped]
    )

    resolved_path = "" if env_path in (None, "") else str(env_path)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    unity_args = additional_args or ["-logfile", str(log_dir_path / "unity.log")]

    unity_env = UnityEnvironment(
        resolved_path,
        worker_id=worker,
        no_graphics=server,
        log_folder=str(log_dir_path),
        additional_args=unity_args,
    )

    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    return env


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

    def reset(self, seed=None, options=None):
        """Reset the underlying environment (Gymnasium API).

        Args:
            seed: Random seed (forwarded for API compatibility; the underlying
                Unity wrapper does not accept it).
            options: Reset options (forwarded for API compatibility).

        Returns:
            tuple: ``(observation, info)`` following the Gymnasium API. If the
            wrapped Unity environment still follows the legacy Gym API and
            returns a single observation, an empty info dict is appended.
        """
        result: object = self.env.reset()
        # Handle both legacy gym (just obs) and modern gymnasium (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

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
