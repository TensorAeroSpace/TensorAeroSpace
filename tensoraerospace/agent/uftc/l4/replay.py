"""Prioritised replay buffer carrying FDD/monitor metadata for L4 training.

Stores ``a_actual`` (the action that actually entered the env, i.e.
``u_safe`` after L1) so the off-policy correction is consistent with
the cascade described in the master spec.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput


@dataclass
class Transition:
    s: np.ndarray
    a_actual: np.ndarray
    r_used: np.ndarray
    reward: float
    s_next: np.ndarray
    done: bool
    fdd: FDDOutput
    alarm: str


class PrioritizedReplay:
    """Proportional-priority replay (Schaul et al. 2015) with a deque backbone."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        eps: float = 1e-3,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_init)
        self.eps = float(eps)
        self._buf: Deque[Transition] = deque(maxlen=self.capacity)
        self._pri: Deque[float] = deque(maxlen=self.capacity)
        self._max_pri = 1.0

    def push(self, t: Transition, priority: float | None = None) -> None:
        p = float(priority) if priority is not None else self._max_pri
        self._buf.append(t)
        self._pri.append(p)
        self._max_pri = max(self._max_pri, p)

    def __len__(self) -> int:
        return len(self._buf)

    def snapshot(self) -> Sequence[Transition]:
        return list(self._buf)

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if len(self._buf) < batch_size:
            raise ValueError("buffer not full enough to sample")
        rng = rng if rng is not None else np.random.default_rng()
        priorities = np.array(self._pri, dtype=np.float64) + self.eps
        probs = priorities**self.alpha
        probs /= probs.sum()
        idx = rng.choice(len(self._buf), size=batch_size, replace=False, p=probs)
        weights = (len(self._buf) * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        transitions = [self._buf[int(i)] for i in idx]
        return transitions, idx, weights.astype(np.float32)

    def update_priorities(
        self, indices: Sequence[int], td_errors: Sequence[float]
    ) -> None:
        for i, e in zip(indices, td_errors):
            p = float(abs(e)) + self.eps
            self._pri[int(i)] = p
            self._max_pri = max(self._max_pri, p)
