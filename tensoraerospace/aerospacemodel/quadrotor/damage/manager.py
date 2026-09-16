"""DamageManager — owns the runtime damage state for an episode."""

from __future__ import annotations

from typing import Optional

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
        self.profile = profile

    def inject_event(self, event: DamageEvent) -> None:
        """Add a one-shot event for this episode (single-fire)."""
        self._injected.append(event)

    def update(
        self, t_current: float, t_previous: float, dt: float
    ) -> list[DamageEvent]:
        """Apply events in ``(t_previous, t_current]`` and advance decay.

        Args:
            t_current: time at the END of the integrator step.
            t_previous: time at the START of the integrator step.
            dt: integrator step size (used by ``state.step_decay``).

        Returns:
            List of events that fired during this step (for logging).
        """
        # Include t=0 once: the usual half-open interval would otherwise
        # discard initial failures for the entire episode.
        window_start = (
            float("-inf") if self._first_update and t_previous == 0 else t_previous
        )
        self._first_update = False
        triggered: list[DamageEvent] = []

        for ev in self.profile.get_pending_events(t_current, window_start):
            ev.apply(self.state)
            triggered.append(ev)

        remaining = []
        for ev in self._injected:
            if window_start < ev.trigger_time <= t_current:
                ev.apply(self.state)
                triggered.append(ev)
            else:
                remaining.append(ev)
        self._injected = remaining

        # Always advance time-decay rotors regardless of triggered events
        self.state.step_decay(dt)

        return triggered
