"""DamageManager — owns the runtime damage state for an episode."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .events import DamageEvent, DamageProfile
from .state import RotorDamageState


class RotorDamageManager:
    """Owns :class:`RotorDamageState` and replays events over the episode.

    Used by the env layer: on every integrator tick the env calls
    :meth:`update` with the current and previous timestamps; events
    that fall in that window are applied to the state. Time-decay
    rotors are advanced using their exact exponential decay.
    """

    def __init__(
        self,
        profile: Optional[DamageProfile] = None,
    ) -> None:
        self.profile: DamageProfile = profile or DamageProfile(events=[])
        self.state: RotorDamageState = RotorDamageState.healthy()
        self._injected: list[DamageEvent] = []
        self._first_update = True

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear all damage and re-baseline (called by `env.reset`).

        ``seed`` is accepted for compatibility with gym.Env.reset.
        """
        self.state = RotorDamageState.healthy()
        self._injected = []
        self._first_update = True

    def set_profile(self, profile: DamageProfile) -> None:
        """Replace the scheduled event profile while retaining current rotor damage."""
        self.profile = profile

    def inject_event(self, event: DamageEvent) -> None:
        """Add a one-shot event for this episode (single-fire)."""
        self._injected.append(event)

    def pending_events(self, t_current: float, t_previous: float) -> list[DamageEvent]:
        """Return scheduled and injected events in chronological order."""
        start = float("-inf") if self._first_update and t_previous == 0 else t_previous
        events = self.profile.get_pending_events(t_current, start)
        events += [ev for ev in self._injected if start < ev.trigger_time <= t_current]
        return sorted(events, key=lambda ev: ev.trigger_time)

    def update(
        self, t_current: float, t_previous: float, dt: float
    ) -> list[DamageEvent]:
        """Advance to the end timestamp, applying events at their actual times.

        Decay uses elapsed timestamps, including only time after its activation.
        ``dt`` is retained for API compatibility; timestamps define the interval.
        Events sharing a timestamp retain profile order, then injection order.
        """
        if not np.all(np.isfinite([t_current, t_previous, dt])) or dt < 0:
            raise ValueError("times and dt must be finite; dt must be nonnegative")
        if t_current < t_previous or t_previous < 0:
            raise ValueError("timestamps must be ordered and nonnegative")
        triggered = self.pending_events(t_current, t_previous)
        self._first_update = False
        cursor = t_previous
        for ev in triggered:
            self.state.step_decay(ev.trigger_time - cursor)
            ev.apply(self.state)
            cursor = ev.trigger_time
        self.state.step_decay(t_current - cursor)
        fired_ids = {id(ev) for ev in triggered}
        self._injected = [ev for ev in self._injected if id(ev) not in fired_ids]
        return triggered
