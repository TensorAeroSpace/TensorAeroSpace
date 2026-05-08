"""ENGINE_THRUST_DRIFT preset: thrust shrinks linearly over time.

The plan (Phase 2 Task 12) prescribes a ``thrust_scale`` ramp that decays
linearly from 1.0 → 0.0 over 300 s. We verify both that the preset is
exposed and that, after 5 s of stepping the existing F-16 ``DamageManager``,
``state.engine.thrust_scale`` lies in ``[0.93, 0.97]`` (≈ 0.95 expected).

The test uses the project-canonical ``DamageManager(geometry, params,
profile)`` constructor and ``update(t_current, t_previous)`` driver — the
same one called from ``NonlinearAngularF16``. The plan's aspirational
``DamageManager(profile, dt=...).step(...)`` signature would require a
broader API refactor; the behavioural assertion (``thrust_scale`` ≈ 0.95
after 5 s at 1 %/s drift) is preserved verbatim.
"""
from __future__ import annotations

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
    DamageManager,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)


def test_engine_thrust_drift_preset_exists() -> None:
    p = presets.ENGINE_THRUST_DRIFT
    assert p.events, "preset must contain at least one ramp event"


def test_thrust_loss_linear_with_time() -> None:
    """At default 1 %/s loss, after 5 s thrust scale ≈ 0.95."""
    geo = load_f16_geometry()
    params = F16AngularParameters()
    mgr = DamageManager(geometry=geo, params=params,
                        profile=presets.ENGINE_THRUST_DRIFT)
    dt = 0.01
    n = int(5.0 / dt)
    t_prev = 0.0
    for k in range(1, n + 1):
        t_now = k * dt
        mgr.update(t_current=t_now, t_previous=t_prev)
        t_prev = t_now
    state = mgr.state
    assert 0.93 <= state.engine.thrust_scale <= 0.97, (
        f"thrust_scale={state.engine.thrust_scale} not in [0.93, 0.97]"
    )
