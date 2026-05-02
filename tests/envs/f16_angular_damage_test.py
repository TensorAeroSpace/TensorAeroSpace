"""NonlinearAngularF16 with damage_profile."""

from __future__ import annotations

import numpy as np


def test_no_damage_profile_unchanged_behaviour():
    """Without damage_profile, info dict has no damage_state key."""
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    e = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=10,
        dt=0.01,
        airspeed=200.0,
    )
    e.reset()
    _, _, _, _, info = e.step(np.zeros(3))
    assert "damage_state" not in info


def test_damage_profile_triggers_event():
    """A scheduled damage event must surface in info['damage_events_triggered']."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        WING_STRIKE_LEFT_TIP,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = WING_STRIKE_LEFT_TIP  # event at t=10s
    e = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=2000,
        dt=0.01,
        airspeed=200.0,
        damage_profile=profile,
        split_stab=True,
    )
    e.reset()
    triggered_seen = False
    for _ in range(1100):  # 11 seconds — past trigger
        _, _, _, _, info = e.step(np.zeros(4))
        if info.get("damage_events_triggered"):
            triggered_seen = True
            break
    assert triggered_seen


def test_damage_observable_extends_obs_space():
    """damage_observable=True extends the observation vector."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageProfile,
        load_f16_geometry,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    geo = load_f16_geometry()
    profile = DamageProfile(events=[])
    e = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=10,
        dt=0.01,
        airspeed=200.0,
        damage_profile=profile,
        damage_observable=True,
        split_stab=True,
    )
    obs, _ = e.reset()
    assert obs.shape[0] > 14, "damage_observable should extend obs"


def test_reset_clears_damage():
    """reset() must restore healthy DamageState even after a damage event fired."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.005, "section_loss", {"section": "left_tip", "loss_fraction": 1.0}
            ),
        ]
    )
    e = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=100,
        dt=0.01,
        airspeed=200.0,
        damage_profile=profile,
        split_stab=True,
    )
    e.reset()
    e.step(np.zeros(4))  # t_prev=0, t_now=0.01 → 0.005 fires
    assert e.unwrapped.damage_manager.state.section_loss["left_tip"] == 1.0
    e.reset()
    assert e.unwrapped.damage_manager.state.section_loss["left_tip"] == 0.0
