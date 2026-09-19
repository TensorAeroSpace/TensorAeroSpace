"""Damage manager — owns the runtime damage state for an episode."""

from __future__ import annotations

from typing import Optional

from .events import AnyDamageEvent, DamageProfile
from .state import B747DamageState


class B747DamageManager:
    """Owns :class:`B747DamageState` and replays events over the episode.

    Used by :class:`tensoraerospace.envs.b747_nonlinear.NonlinearB747Env`:
    the native model splits each integrator tick at scheduled event times,
    calls :meth:`update` at interval starts and advances surface decay after
    each segment. Standalone callers may supply a nonzero ``dt`` to update
    events and decay together.
    """

    def __init__(self, profile: Optional[DamageProfile] = None) -> None:
        self.profile: DamageProfile = profile or DamageProfile(events=[])
        self.state: B747DamageState = B747DamageState.healthy()
        self._injected: list["AnyDamageEvent"] = []

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear all damage and re-baseline (called by ``env.reset``).

        ``seed`` is accepted for compatibility with gym.Env.reset.
        """
        self.state = B747DamageState.healthy()
        self._injected = []

    @property
    def scheduled_events(self):
        """Profile and injected events, used to split physical integration."""
        return tuple(self.profile.events) + tuple(self._injected)

    def set_profile(self, profile: DamageProfile) -> None:
        self.profile = profile

    def inject_event(self, event: "AnyDamageEvent") -> None:
        """Add a one-shot event for this episode (single-fire)."""
        self._injected.append(event)

    def update(
        self, t_current: float, t_previous: float, dt: float
    ) -> list["AnyDamageEvent"]:
        """Apply events in ``(t_previous, t_current]`` and advance decay.

        Returns the list of events that fired during this step.
        """
        triggered: list["AnyDamageEvent"] = []
        for ev in self.profile.get_pending_events(t_current, t_previous):
            ev.apply(self.state)
            triggered.append(ev)

        remaining: list["AnyDamageEvent"] = []
        for ev in self._injected:
            if t_previous < ev.trigger_time <= t_current:
                ev.apply(self.state)
                triggered.append(ev)
            else:
                remaining.append(ev)
        self._injected = remaining

        # Advance time-decay surfaces every tick
        self.state.step_decay(dt)
        return triggered
