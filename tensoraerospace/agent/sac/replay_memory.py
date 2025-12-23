"""Replay buffer implementation for SAC.

This module implements experience replay utilities used by the SAC agent.
"""

import os
import pickle
import random
from typing import List, Tuple, Union

import numpy as np


class ReplayMemory:
    """Replay buffer for off-policy reinforcement learning algorithms.

    Args:
        capacity (int): Maximum number of transitions to store.
        seed (int): Random seed used for sampling.

    Attributes:
        capacity (int): Maximum capacity.
        buffer (List): Stored transitions.
        position (int): Current write position.

    """

    def __init__(self, capacity: int, seed: int):
        """Initialize replay buffer with fixed capacity and seed.

        Args:
            capacity: Maximum number of transitions.
            seed: Random seed for sampling.
        """
        random.seed(seed)
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position: int = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: Union[bool, float],
    ) -> None:
        """Add a transition to the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Episode termination flag (bool) or mask (0.0/1.0).

        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ``(state, action, reward, next_state, done)`` batch.

        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """Return the current number of stored transitions.

        Returns:
            int: Buffer size.

        """
        return len(self.buffer)

    def save_buffer(
        self, env_name: str, suffix: str = "", save_path: str | None = None
    ) -> None:
        """Save the replay buffer to disk.

        Args:
            env_name (str): Environment name used in the default path.
            suffix (str): Optional filename suffix.
            save_path (str): Optional explicit save path. If None, a default
                path under ``checkpoints/`` is used.

        """
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print("Saving buffer to {}".format(save_path))

        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path: str) -> None:
        """Load a replay buffer from disk.

        Args:
            save_path (str): Path to a previously saved buffer.

        """
        print("Loading buffer from {}".format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
