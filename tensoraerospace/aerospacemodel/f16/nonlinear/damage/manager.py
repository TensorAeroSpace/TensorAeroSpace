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
        # Active linear-ramp engine events (Phase 2 Task 12). Each entry is
        # (t_start, t_end, value_start, value_end, target). Tracked across
        # ``update()`` calls so thrust_scale advances every step until t_end.
        self._active_ramps: list[tuple[float, float, float, float, str]] = []

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Clear all damage and re-apply baseline params.

        seed is accepted for compatibility with gym.Env.reset semantics
        (forwarded to RandomDamageProfileGenerator in Phase 8).
        """
        self.state = DamageState.healthy(self.geometry)
        self._injected = []
        self._active_ramps = []
        apply_to_params(self.params, self.geometry, self.state)

    def set_profile(self, profile: DamageProfile) -> None:
        self.profile = profile

    def inject_event(self, event: DamageEvent) -> None:
        """Add a one-shot event to be triggered on the next matching window."""
        self._injected.append(event)

    def update(self, t_current: float, t_previous: float) -> list[DamageEvent]:
        """Trigger any events in (t_previous, t_current]; return them.

        Also advances any active linear-ramp engine events (Task 12). The
        ramp continues to update ``state.engine.thrust_scale`` on every
        ``update()`` call between ``t_start`` and ``t_end``.
        """
        triggered: list[DamageEvent] = []

        # Profile events
        for ev in self.profile.get_pending_events(t_current, t_previous):
            self._apply_event(ev)
            triggered.append(ev)

        # Injected events (single-fire, then removed)
        remaining: list[DamageEvent] = []
        for ev in self._injected:
            if t_previous < ev.trigger_time <= t_current:
                self._apply_event(ev)
                triggered.append(ev)
            else:
                remaining.append(ev)
        self._injected = remaining

        # Advance active linear ramps every step (independent of triggered).
        ramp_changed = self._advance_ramps(t_current)

        if triggered or ramp_changed:
            apply_to_params(self.params, self.geometry, self.state)

        return triggered

    def _advance_ramps(self, t_current: float) -> bool:
        """Update state values for all active linear ramps; prune finished ones.

        Returns True if any ramp value changed (so ``apply_to_params`` should
        be re-run). Setting the value at exactly ``t_end`` clamps to the end
        value; ramps past their ``t_end`` are removed after one final write.
        """
        if not self._active_ramps:
            return False
        changed = False
        kept: list[tuple[float, float, float, float, str]] = []
        for t_start, t_end, v_start, v_end, target in self._active_ramps:
            if t_current <= t_start:
                # Ramp not yet started — keep, write start value.
                value = v_start
                still_active = True
            elif t_current >= t_end:
                value = v_end
                still_active = False  # final write, then prune
            else:
                frac = (t_current - t_start) / (t_end - t_start)
                value = v_start + (v_end - v_start) * frac
                still_active = True
            if target == "thrust_scale":
                if self.state.engine.thrust_scale != value:
                    self.state.engine.thrust_scale = float(value)
                    changed = True
            elif target == "thrust_factor":
                if self.state.engine.thrust_factor != value:
                    self.state.engine.thrust_factor = float(value)
                    changed = True
            if still_active:
                kept.append((t_start, t_end, v_start, v_end, target))
        self._active_ramps = kept
        return changed

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
            # Linear-ramp form (Task 12): payload has ramp="linear" plus
            # *_start / *_end values for thrust_scale or thrust_factor and
            # the event carries ``duration``. Existing instantaneous form
            # (e.g. ENGINE_FLAMEOUT) is preserved as the else-branch.
            if ev.payload.get("ramp") == "linear":
                t_start = float(ev.trigger_time)
                duration = ev.duration if ev.duration is not None else 0.0
                t_end = t_start + float(duration)
                if (
                    "thrust_scale_start" in ev.payload
                    or "thrust_scale_end" in ev.payload
                ):
                    v_start = float(ev.payload.get("thrust_scale_start", 1.0))
                    v_end = float(ev.payload.get("thrust_scale_end", 0.0))
                    self.state.engine.thrust_scale = v_start
                    self._active_ramps.append(
                        (t_start, t_end, v_start, v_end, "thrust_scale")
                    )
                elif (
                    "thrust_factor_start" in ev.payload
                    or "thrust_factor_end" in ev.payload
                ):
                    v_start = float(ev.payload.get("thrust_factor_start", 1.0))
                    v_end = float(ev.payload.get("thrust_factor_end", 0.0))
                    self.state.engine.thrust_factor = v_start
                    self._active_ramps.append(
                        (t_start, t_end, v_start, v_end, "thrust_factor")
                    )
                if "hard_failure" in ev.payload:
                    self.state.engine.hard_failure = bool(ev.payload["hard_failure"])
            else:
                if "thrust_factor" in ev.payload:
                    self.state.engine.thrust_factor = float(ev.payload["thrust_factor"])
                if "thrust_scale" in ev.payload:
                    self.state.engine.thrust_scale = float(ev.payload["thrust_scale"])
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
