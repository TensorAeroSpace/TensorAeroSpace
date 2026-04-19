"""Event-triggered supervisor for ET-DHP.

Implements the discrete-time Lipschitz-style trigger condition from
Sun et al. (2022). Between triggers the actuator simply **holds** the
last control and no actor/critic updates run, which is the whole point
of the scheme: decouple the controller's computational rate from the
sensor/simulation rate.

Condition (equation 11 in the paper)::

    ‖x_k − x_et‖ > ρ · ‖x_et‖ · (1 − (2ρ)^{k − k_trig}) / (1 − 2ρ)

where ``x_et`` and ``k_trig`` are the state and time index at the most
recent trigger, and ``ρ ∈ (0, 0.5)`` is the designer-chosen Lipschitz
bound (smaller ⇒ more frequent triggers).
"""

from __future__ import annotations

import numpy as np


class EventTrigger:
    """Stateful Lipschitz-based event-trigger for ET-DHP.

    Keeps track of the reference state captured at the last trigger and
    compares its distance to the incoming measured state against a
    growing threshold that saturates once ``(2ρ)^Δk → 0``.

    Args:
        rho: Lipschitz constant ``ρ``. Paper uses 0.1 — lower values
            give tighter tracking at the cost of more triggers.
        trigger_first_step: If True (default), the very first call to
            :meth:`should_trigger` always returns True so the networks
            and control run once at ``k = 0``.
        min_floor: Lower bound on the threshold. Adds a small absolute
            slack so the trigger does not fire on pure measurement
            noise when ``x ≈ 0`` (the multiplicative ``‖x_et‖`` term
            collapses to zero at the origin otherwise).
    """

    def __init__(
        self,
        rho: float = 0.1,
        trigger_first_step: bool = True,
        min_floor: float = 0.0,
    ) -> None:
        if not 0.0 < rho < 0.5:
            raise ValueError("rho must lie in (0, 0.5)")
        self.rho = float(rho)
        self.trigger_first_step = bool(trigger_first_step)
        self.min_floor = float(min_floor)
        self._reset_state()

    def _reset_state(self) -> None:
        self.state_at_trigger: np.ndarray | None = None
        self.step_at_trigger: int = 0
        self.norm_state_at_trigger: float = 0.0
        self.num_triggers: int = 0
        self.last_condition: float = 0.0
        self.last_norm: float = 0.0
        self._first_call_done: bool = False

    def reset(self) -> None:
        """Re-arm the trigger for a new episode."""
        self._reset_state()

    def threshold(self, step: int) -> float:
        """Compute the current Lipschitz threshold at ``step``.

        Returns the floor when the trigger has never fired yet (so the
        first evaluation still runs).
        """
        if self.state_at_trigger is None:
            return self.min_floor
        d_k = int(step - self.step_at_trigger)
        if d_k < 0:
            d_k = 0
        denom = 1.0 - 2.0 * self.rho
        # (2ρ)^Δk decays fast for ρ < 0.5, so the threshold saturates
        # at ρ · ‖x_et‖ / (1 − 2ρ) — the asymptotic Lipschitz bound.
        factor = (1.0 - (2.0 * self.rho) ** d_k) / denom
        return max(
            self.min_floor,
            float(self.rho * self.norm_state_at_trigger * factor),
        )

    def should_trigger(self, state_measured: np.ndarray, step: int) -> bool:
        """Return ``True`` when the networks must be re-evaluated.

        Args:
            state_measured: Current (possibly noisy) state measurement.
            step: Current discrete time index.
        """
        x = np.asarray(state_measured, dtype=np.float64).reshape(-1)

        if not self._first_call_done:
            self._first_call_done = True
            if self.trigger_first_step:
                self._capture(x, step)
                return True

        if self.state_at_trigger is None:
            # Threshold is the floor — any measurement > floor triggers.
            if float(np.linalg.norm(x)) > self.min_floor:
                self._capture(x, step)
                return True
            return False

        norm_diff = float(np.linalg.norm(x - self.state_at_trigger))
        cond = self.threshold(step)
        self.last_condition = cond
        self.last_norm = norm_diff
        if norm_diff > cond:
            self._capture(x, step)
            return True
        return False

    def _capture(self, state: np.ndarray, step: int) -> None:
        """Mark a trigger at ``(state, step)`` and update internal counters."""
        self.state_at_trigger = state.copy()
        self.step_at_trigger = int(step)
        self.norm_state_at_trigger = float(np.linalg.norm(state))
        self.num_triggers += 1
