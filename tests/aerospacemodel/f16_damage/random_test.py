"""Random damage profile generation for RL."""

from __future__ import annotations


def test_generator_seeded_is_deterministic():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g1 = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    g2 = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    p1 = g1.sample()
    p2 = g2.sample()
    assert p1.events[0].trigger_time == p2.events[0].trigger_time
    assert p1.events[0].payload == p2.events[0].payload


def test_generator_respects_time_range():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(1, 1),
        seed=42,
    )
    for _ in range(50):
        p = g.sample()
        for e in p.events:
            assert 5.0 <= e.trigger_time <= 25.0


def test_generator_respects_num_events_range():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.random import (
        RandomDamageProfileGenerator,
    )
    g = RandomDamageProfileGenerator(
        event_types=["section_loss"],
        time_range=(5.0, 25.0),
        severity_range=(0.1, 1.0),
        num_events_range=(2, 4),
        seed=42,
    )
    for _ in range(20):
        p = g.sample()
        assert 2 <= len(p.events) <= 4
