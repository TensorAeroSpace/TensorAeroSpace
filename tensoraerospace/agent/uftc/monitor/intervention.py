"""Variant-B macro-actions and dispatcher."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

LOG = logging.getLogger(__name__)


MacroKind = Literal[
    "force_rls_reset",
    "freeze_l4_learning",
    "degrade_reference_to_hold",
    "request_actuator_hold",
]


@dataclass
class MacroAction:
    kind: MacroKind
    payload: dict = field(default_factory=dict)


class MacroActionDispatcher:
    """Map MacroAction list onto explicit method calls on the wired layers."""

    def __init__(self, *, l3: Any | None, l4: Any | None, l1: Any | None) -> None:
        self.l3 = l3
        self.l4 = l4
        self.l1 = l1

    def dispatch(self, actions, current_step: int) -> dict:
        diag: dict = {}
        for a in actions:
            try:
                if a.kind == "force_rls_reset" and self.l3 is not None:
                    self.l3.force_reset(
                        severity_hint=float(a.payload.get("severity", 1.0))
                    )
                    diag["force_rls_reset"] = int(current_step)
                elif a.kind == "freeze_l4_learning" and self.l4 is not None:
                    until = int(current_step) + int(a.payload["duration"])
                    self.l4.freeze_learning(until_step=until)
                    diag["freeze_l4_learning_until"] = until
                elif a.kind == "degrade_reference_to_hold" and self.l4 is not None:
                    self.l4.degrade_reference_to_hold()
                    diag["degrade_reference_to_hold"] = int(current_step)
                elif a.kind == "request_actuator_hold" and self.l1 is not None:
                    self.l1.request_actuator_hold()
                    diag["request_actuator_hold"] = int(current_step)
            except Exception as e:
                LOG.warning("macro-action %s failed: %s", a.kind, e)
        return diag
