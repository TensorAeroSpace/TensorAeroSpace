"""End-to-end: damage application in AngularF16 changes trajectory."""

from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


def test_no_damage_attribute_keeps_legacy_behaviour():
    """When damage_state is None, trajectory must be bit-identical to
    pre-damage AngularF16 behaviour."""
    m_legacy = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m_legacy.run_step(np.zeros(3))
    s_legacy = m_legacy.current_state.copy()
    # New construct with damage_state=None still gives identical trajectory
    m_with = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    # damage_state attribute defaults to None
    assert getattr(m_with, "damage_state", None) is None
    m_with.run_step(np.zeros(3))
    np.testing.assert_allclose(s_legacy, m_with.current_state, atol=1e-12)


def test_damage_state_attribute_used_in_ode():
    """When damage_state describes a damaged aircraft, the trajectory must
    differ from the healthy baseline."""
    geo = load_f16_geometry()
    state_healthy = DamageState.healthy(geo)
    state_damaged = DamageState.healthy(geo)
    state_damaged.set_section_loss("left_tip", 1.0)

    # Run baseline
    m1 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m1.damage_state = state_healthy
    m1.damage_geometry = geo
    for _ in range(10):
        m1.run_step(np.zeros(3))

    # Run with damaged left tip
    m2 = AngularF16(x0=np.zeros(14), dt=0.01, integrator="rk4")
    m2.damage_state = state_damaged
    m2.damage_geometry = geo
    for _ in range(10):
        m2.run_step(np.zeros(3))

    # Trajectories must differ (specifically: roll rate wx)
    diff = np.abs(m1.current_state - m2.current_state)
    assert diff.sum() > 1e-6, "Damage application had no effect on trajectory"
