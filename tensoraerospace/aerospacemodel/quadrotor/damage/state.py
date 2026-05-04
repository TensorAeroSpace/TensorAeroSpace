"""Mutable runtime damage state of the 4 rotors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RotorDamageState:
    """Per-rotor effectiveness and time-decay constants.

    Each rotor has a current effectiveness ``mu`` and an optional
    exponential decay time-constant ``tau`` for gradual degradation.

    Effectiveness model:

    - At each time step the env reads ``mu[i]`` and applies
      :math:`\\omega^2_{i,\\text{eff}} = \\mu_i \\cdot \\omega^2_{i,\\text{cmd}}`.
    - If ``tau[i] > 0``, ``mu[i]`` evolves over time as
      :math:`\\dot\\mu_i = -(1/\\tau_i)(\\mu_i - \\mu_i^{\\text{floor}})`,
      where ``mu_floor[i]`` is the asymptotic effectiveness (0 = full
      eventual loss; positive = partial wear). The env applies one
      explicit-Euler step per integrator tick.
    """

    mu: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float64))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    mu_floor: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64).reshape(-1)
        self.tau = np.asarray(self.tau, dtype=np.float64).reshape(-1)
        self.mu_floor = np.asarray(self.mu_floor, dtype=np.float64).reshape(-1)
        for name, arr in (
            ("mu", self.mu),
            ("tau", self.tau),
            ("mu_floor", self.mu_floor),
        ):
            if arr.size != 4:
                raise ValueError(f"{name} must have 4 elements; got {arr.size}")
        if np.any(self.mu < 0) or np.any(self.mu > 1):
            raise ValueError("mu must be in [0, 1] element-wise")
        if np.any(self.tau < 0):
            raise ValueError("tau must be non-negative element-wise")
        if np.any(self.mu_floor < 0) or np.any(self.mu_floor > 1):
            raise ValueError("mu_floor must be in [0, 1] element-wise")

    @classmethod
    def healthy(cls) -> "RotorDamageState":
        """Fresh state with all rotors at full effectiveness."""
        return cls()

    def step_decay(self, dt: float) -> None:
        """Advance time-varying decay one tick (explicit Euler).

        Active only on rotors with ``tau[i] > 0``. After this call,
        ``mu[i]`` is closer to ``mu_floor[i]``.
        """
        active = self.tau > 0
        if not np.any(active):
            return
        self.mu[active] += (
            -(1.0 / self.tau[active]) * (self.mu[active] - self.mu_floor[active]) * dt
        )
        self.mu = np.clip(self.mu, 0.0, 1.0)

    def snapshot(self) -> dict:
        """Return a JSON-friendly view of the current state."""
        return {
            "mu": self.mu.tolist(),
            "tau": self.tau.tolist(),
            "mu_floor": self.mu_floor.tolist(),
        }
