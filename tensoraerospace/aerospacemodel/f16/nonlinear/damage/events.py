"""DamageEvent and DamageProfile.

DamageEvent is a frozen, atomic description of a single failure scheduled
to fire at trigger_time. DamageProfile groups events for an episode and
provides time-window queries used by DamageManager (Task 6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

EventType = Literal[
    "section_loss",
    "control_failure",
    "engine_failure",
    "structural_change",
]
_VALID_EVENT_TYPES = (
    "section_loss",
    "control_failure",
    "engine_failure",
    "structural_change",
)


@dataclass(frozen=True)
class DamageEvent:
    trigger_time: float
    event_type: EventType
    payload: dict = field(default_factory=dict)
    label: Optional[str] = None
    duration: Optional[float] = None  # None = permanent

    def __post_init__(self) -> None:
        if self.event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {self.event_type!r}; expected one of "
                f"{_VALID_EVENT_TYPES}"
            )
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be >= 0; got {self.trigger_time}")


@dataclass
class DamageProfile:
    events: list[DamageEvent] = field(default_factory=list)
    seed: Optional[int] = None

    def get_pending_events(
        self, t_current: float, t_previous: float
    ) -> list[DamageEvent]:
        """Events triggering in the half-open interval (t_previous, t_current]."""
        return [e for e in self.events if t_previous < e.trigger_time <= t_current]
