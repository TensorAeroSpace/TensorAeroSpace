"""Small replay buffer used by the ADP agent (optional).

The original ACD/ADP formulations are typically online; however, for stability
and sample efficiency we optionally support a replay buffer similar to DDPG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    act: np.ndarray
    reward: float
    next_obs: np.ndarray
    done_bootstrap: float


class ReplayBuffer:
    """Uniform random replay buffer."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: List[Transition] = []
        self._pos = 0
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done_bootstrap: float,
    ) -> None:
        tr = Transition(
            obs=np.asarray(obs, dtype=np.float32),
            act=np.asarray(act, dtype=np.float32),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            done_bootstrap=float(done_bootstrap),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(tr)
        else:
            self._buf[self._pos] = tr
        self._pos = (self._pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > len(self._buf):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self._buf)}"
            )
        idx = self._rng.integers(0, len(self._buf), size=(batch_size,))
        obs = np.stack([self._buf[i].obs for i in idx], axis=0)
        act = np.stack([self._buf[i].act for i in idx], axis=0)
        rew = np.asarray([self._buf[i].reward for i in idx], dtype=np.float32).reshape(
            -1, 1
        )
        nxt = np.stack([self._buf[i].next_obs for i in idx], axis=0)
        done = np.asarray(
            [self._buf[i].done_bootstrap for i in idx], dtype=np.float32
        ).reshape(-1, 1)
        return obs, act, rew, nxt, done
