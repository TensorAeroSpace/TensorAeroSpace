"""3-level alarm state machine with hysteresis and cooldown."""
from __future__ import annotations

from dataclasses import dataclass

from .composite import AlarmLevel


@dataclass
class AlarmStateMachine:
    cooldown_steps: int = 200
    level: AlarmLevel = "OK"
    _steps_in_level: int = 0

    def update(self, *, V_total: float, mu_uub: float,
               warn_frac: float, crit_frac: float) -> AlarmLevel:
        warn = warn_frac * mu_uub
        crit = crit_frac * mu_uub
        clear_warn = 0.5 * warn
        clear_crit = 0.5 * crit

        new = self.level
        if self.level == "OK":
            if V_total > crit: new = "CRITICAL"
            elif V_total > warn: new = "WARN"
        elif self.level == "WARN":
            if V_total > crit: new = "CRITICAL"
            elif V_total < clear_warn and self._steps_in_level >= self.cooldown_steps:
                new = "OK"
        elif self.level == "CRITICAL":
            if V_total < clear_crit and self._steps_in_level >= self.cooldown_steps:
                new = "WARN" if V_total > clear_warn else "OK"

        if new != self.level:
            self.level = new
            self._steps_in_level = 0
        else:
            self._steps_in_level += 1
        return self.level

    def reset(self) -> None:
        self.level = "OK"
        self._steps_in_level = 0
