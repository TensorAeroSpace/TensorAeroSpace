"""CUSUM change-point detector with hysteresis for UFTC FDD.

Drives the binary `fault_present` flag and the soft `severity` signal
that the rest of the controller uses to scale RLS reset and trust-region
expansion.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChangePointState:
    """One-step output of :class:`ChangePointDetector`."""

    cusum: float
    alarm: bool
    severity: float          # cusum / h_alarm, clipped to [0, 10]
    time_since_alarm: int    # steps since last rising edge (0 if never fired)


class ChangePointDetector:
    """One-sided CUSUM detector on a positive scalar score (e.g. Mahalanobis).

    Args:
        n_dim: Dimension of the underlying innovation vector.
        drift: Per-step decrement subtracted from each input. ``None``
            defaults to ``n_dim * 1.5``. The factor 1.5 ensures a negative
            expected increment under H₀ (where ``E[d_t] = n_dim`` for a
            χ²_n score), preventing the CUSUM from drifting upward and
            crossing ``h_alarm`` in the absence of faults. Using ``drift =
            n_dim`` would make the process a martingale and eventually
            trigger a false alarm over long runs.
        h_alarm: Upper threshold; CUSUM crossing this triggers an alarm.
        h_clear: Lower threshold; CUSUM falling below this clears the
            alarm (after cooldown). Must be strictly less than h_alarm.
        cooldown_steps: Minimum steps between rising edges. Re-arming the
            detector during cooldown is suppressed.

    Raises:
        ValueError: If ``h_alarm <= h_clear`` or any threshold is
            non-positive.
    """

    def __init__(
        self,
        *,
        n_dim: int,
        drift: float | None = None,
        h_alarm: float = 20.0,
        h_clear: float = 5.0,
        cooldown_steps: int = 200,
    ) -> None:
        if h_alarm <= 0.0 or h_clear <= 0.0:
            raise ValueError("Thresholds must be positive")
        if h_alarm <= h_clear:
            raise ValueError("h_alarm must be strictly greater than h_clear")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        self.n_dim = int(n_dim)
        self.drift = float(drift if drift is not None else n_dim * 1.5)
        self.h_alarm = float(h_alarm)
        self.h_clear = float(h_clear)
        self.cooldown_steps = int(cooldown_steps)
        self.reset()

    def reset(self) -> None:
        """Clear CUSUM state."""
        self._cusum = 0.0
        self._alarm = False
        self._cooldown = 0
        self._time_since_alarm = 0

    def update(self, d_t: float) -> ChangePointState:
        """Feed one Mahalanobis-distance sample, update state."""
        self._cusum = max(0.0, self._cusum + float(d_t) - self.drift)

        if self._cooldown > 0:
            self._cooldown -= 1

        if not self._alarm:
            if self._cusum > self.h_alarm and self._cooldown == 0:
                self._alarm = True
                self._time_since_alarm = 0
                self._cooldown = self.cooldown_steps
        else:
            if self._cusum < self.h_clear and self._cooldown == 0:
                self._alarm = False
            self._time_since_alarm += 1

        severity = max(0.0, min(self._cusum / self.h_alarm, 10.0))
        return ChangePointState(
            cusum=float(self._cusum),
            alarm=bool(self._alarm),
            severity=float(severity),
            time_since_alarm=int(self._time_since_alarm),
        )
