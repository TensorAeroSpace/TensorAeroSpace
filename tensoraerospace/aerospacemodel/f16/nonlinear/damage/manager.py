"""DamageManager: ties events, state, and param recomputation together.

Owns the mutable DamageState for an episode. On each `update(t_current,
t_previous)` call:
  1. Pulls events from the DamageProfile that fire in the time window
  2. Pulls injected (one-shot) events that fire in the same window
  3. Applies each event to the DamageState via _apply_event
  4. Recomputes aircraft parameters via apply_to_params (mass, S, b,
     bA, J*, rcgx) — only if any event triggered

The model owning the params reads them on the next integrator step.
"""

from __future__ import annotations

from math import isfinite
from typing import Optional

from .events import DamageEvent, DamageProfile
from .geometry import BaseGeometry
from .recompute import apply_to_params
from .state import ControlFailure, DamageState


class DamageManager:
    """Owns DamageState, applies events, drives param recomputation."""

    def __init__(
        self,
        geometry: BaseGeometry,
        params,
        profile: Optional[DamageProfile] = None,
    ) -> None:
        self.geometry = geometry
        self.params = params
        self.profile: DamageProfile = profile or DamageProfile(events=[])
        self.state = DamageState.healthy(geometry)
        self._injected: list[DamageEvent] = []
        self._applied_profile: set[int] = set()

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear all damage and re-apply baseline params.

        seed is accepted for compatibility with gym.Env.reset semantics
        (forwarded to RandomDamageProfileGenerator in Phase 8).
        """
        self.state = DamageState.healthy(self.geometry)
        self._injected = []
        self._applied_profile.clear()
        apply_to_params(self.params, self.geometry, self.state)

    def set_profile(self, profile: DamageProfile) -> None:
        """Replace the event profile and clear its consumed-event markers, retaining
        damage.
        """
        self.profile = profile
        self._applied_profile.clear()

    def inject_event(self, event: DamageEvent) -> None:
        """Add a one-shot event to be triggered on the next matching window."""
        self._injected.append(event)

    def pending_events(self, t_current: float, t_previous: float) -> list[DamageEvent]:
        """Unconsumed profile and injected events, in stable chronological order."""
        if not all(map(isfinite, (t_current, t_previous))) or t_current < t_previous:
            raise ValueError("event window must be finite and ordered")
        profile = [
            ev
            for i, ev in enumerate(self.profile.events)
            if i not in self._applied_profile
        ]
        return sorted(
            [
                ev
                for ev in profile + self._injected
                if t_previous < ev.trigger_time <= t_current
            ],
            key=lambda ev: ev.trigger_time,
        )

    def update(self, t_current: float, t_previous: float) -> list[DamageEvent]:
        """Apply each event once in (t_previous, t_current], ordered by time."""
        triggered = self.pending_events(t_current, t_previous)
        for ev in triggered:
            self._apply_event(ev)
        triggered_ids = {id(ev) for ev in triggered}
        self._applied_profile.update(
            i for i, ev in enumerate(self.profile.events) if id(ev) in triggered_ids
        )
        self._injected = [ev for ev in self._injected if id(ev) not in triggered_ids]
        if triggered:
            apply_to_params(self.params, self.geometry, self.state)
        return triggered

    def _apply_event(self, ev: DamageEvent) -> None:
        if ev.event_type == "section_loss":
            self.state.set_section_loss(
                ev.payload["section"], ev.payload["loss_fraction"]
            )
        elif ev.event_type == "control_failure":
            payload = dict(ev.payload)
            surface = payload.pop("surface")
            cf = ControlFailure(**payload)
            self.state.set_control_failure(surface, cf)
        elif ev.event_type == "engine_failure":
            if "thrust_factor" in ev.payload:
                self.state.engine.thrust_factor = float(ev.payload["thrust_factor"])
            if "hard_failure" in ev.payload:
                self.state.engine.hard_failure = bool(ev.payload["hard_failure"])
        elif ev.event_type == "structural_change":
            if "mass_delta_kg" in ev.payload:
                self.state.structural.extra_mass_delta_kg += float(
                    ev.payload["mass_delta_kg"]
                )
            if "cg_shift_m" in ev.payload:
                shift = ev.payload["cg_shift_m"]
                old = self.state.structural.extra_cg_shift_m
                self.state.structural.extra_cg_shift_m = (
                    old[0] + shift[0],
                    old[1] + shift[1],
                    old[2] + shift[2],
                )
            if "inertia_delta" in ev.payload:
                d = ev.payload["inertia_delta"]
                old = self.state.structural.extra_inertia_delta
                self.state.structural.extra_inertia_delta = (
                    old[0] + d[0],
                    old[1] + d[1],
                    old[2] + d[2],
                    old[3] + d[3],
                )
        else:
            raise ValueError(f"Unknown event_type: {ev.event_type}")
