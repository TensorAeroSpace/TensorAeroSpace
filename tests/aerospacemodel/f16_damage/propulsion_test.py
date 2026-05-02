"""Engine thrust factor."""

from __future__ import annotations

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
    EngineState,
)


def test_default_thrust_factor_is_one():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )

    state = DamageState(section_loss={}, control_failures={})
    assert effective_thrust(1000.0, state) == pytest.approx(1000.0)


def test_thrust_factor_scales():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )

    state = DamageState(
        section_loss={}, control_failures={}, engine=EngineState(thrust_factor=0.4)
    )
    assert effective_thrust(1000.0, state) == pytest.approx(400.0)


def test_hard_failure_zeros_thrust():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )

    state = DamageState(
        section_loss={},
        control_failures={},
        engine=EngineState(thrust_factor=1.0, hard_failure=True),
    )
    assert effective_thrust(1000.0, state) == 0.0
