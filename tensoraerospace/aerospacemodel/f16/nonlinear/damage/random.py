"""Random damage profile generation."""

from __future__ import annotations

import random
from typing import Optional

from .events import DamageEvent, DamageProfile

_DEFAULT_LOSABLE_SECTIONS = (
    "left_tip",
    "left_mid",
    "right_tip",
    "right_mid",
    "stab_left",
    "stab_right",
)


class RandomDamageProfileGenerator:
    """Sample DamageProfiles for RL/curriculum training.

    All knobs are inclusive ranges sampled uniformly. Currently supports
    `section_loss`, `control_failure`, and `engine_failure` events.
    """

    def __init__(
        self,
        event_types: list[str],
        time_range: tuple[float, float],
        severity_range: tuple[float, float] = (0.1, 1.0),
        num_events_range: tuple[int, int] = (1, 1),
        seed: Optional[int] = None,
        sections: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.event_types = list(event_types)
        if not self.event_types:
            raise ValueError("event_types must be non-empty")
        self.time_range = time_range
        self.severity_range = severity_range
        self.num_events_range = num_events_range
        self.rng = random.Random(seed)
        self.sections = sections or _DEFAULT_LOSABLE_SECTIONS

    def sample(self) -> DamageProfile:
        n_events = self.rng.randint(*self.num_events_range)
        events: list[DamageEvent] = []
        for _ in range(n_events):
            t = self.rng.uniform(*self.time_range)
            kind = self.rng.choice(self.event_types)
            if kind == "section_loss":
                section = self.rng.choice(self.sections)
                fraction = self.rng.uniform(*self.severity_range)
                events.append(
                    DamageEvent(
                        trigger_time=t,
                        event_type="section_loss",
                        payload={"section": section, "loss_fraction": fraction},
                        label=f"random_loss_{section}_{fraction:.2f}",
                    )
                )
            elif kind == "control_failure":
                surface = self.rng.choice(
                    [
                        "stab_left",
                        "stab_right",
                        "rudder",
                        "aileron_left",
                        "aileron_right",
                    ]
                )
                mode = self.rng.choice(["jam", "efficiency_loss", "lost"])
                payload: dict[str, str | float] = {"surface": surface, "mode": mode}
                if mode == "jam":
                    payload["jam_position_rad"] = self.rng.uniform(-0.15, 0.15)
                elif mode == "efficiency_loss":
                    payload["efficiency"] = self.rng.uniform(0.2, 0.9)
                events.append(
                    DamageEvent(
                        trigger_time=t,
                        event_type="control_failure",
                        payload=payload,
                        label=f"random_{surface}_{mode}",
                    )
                )
            elif kind == "engine_failure":
                tf = self.rng.uniform(0.0, 0.6)
                events.append(
                    DamageEvent(
                        trigger_time=t,
                        event_type="engine_failure",
                        payload={"thrust_factor": tf},
                        label=f"random_engine_{tf:.2f}",
                    )
                )
        return DamageProfile(events=events)
