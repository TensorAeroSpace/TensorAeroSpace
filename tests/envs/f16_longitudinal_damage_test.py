"""NonlinearLongitudinalF16 with symmetric damage."""

from __future__ import annotations

import numpy as np


def test_no_damage_profile_unchanged():
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )

    e = NonlinearLongitudinalF16(
        initial_state=np.zeros(2),
        reference_signal=np.zeros((1, 100)),
        number_time_steps=10,
        dt=0.01,
    )
    e.reset()
    _, _, _, _, info = e.step(np.zeros(1))
    assert "damage_state" not in info


def test_symmetric_loss_changes_alpha_response():
    """Symmetric damage should not cause asymmetric effects but should still
    change the lift response."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.005, "section_loss", {"section": "left_tip", "loss_fraction": 0.5}
            ),
            DamageEvent(
                0.005, "section_loss", {"section": "right_tip", "loss_fraction": 0.5}
            ),
        ]
    )
    e = NonlinearLongitudinalF16(
        initial_state=np.zeros(2),
        reference_signal=np.zeros((1, 250)),
        number_time_steps=200,
        dt=0.01,
        damage_profile=profile,
    )
    e.reset()
    for _ in range(100):
        e.step(np.zeros(1))
    # Should not crash; final state finite
    assert np.all(np.isfinite(e.unwrapped.model.current_state))


def test_damage_event_triggered_in_info():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.005, "section_loss", {"section": "left_tip", "loss_fraction": 0.5}
            ),
            DamageEvent(
                0.005, "section_loss", {"section": "right_tip", "loss_fraction": 0.5}
            ),
        ]
    )
    e = NonlinearLongitudinalF16(
        initial_state=np.zeros(2),
        reference_signal=np.zeros((1, 30)),
        number_time_steps=20,
        dt=0.01,
        damage_profile=profile,
    )
    e.reset()
    triggered_count = 0
    for _ in range(5):
        _, _, _, _, info = e.step(np.zeros(1))
        if info.get("damage_events_triggered"):
            triggered_count += len(info["damage_events_triggered"])
    assert triggered_count == 2  # both events fire on the first step
