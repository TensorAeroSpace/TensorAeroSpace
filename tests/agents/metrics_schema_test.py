"""Schema invariants for tensoraerospace.agent.metrics.schema."""

import re

import pytest

from tensoraerospace.agent.metrics import schema

_VALID_TAG = re.compile(r"^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$")


def _own_class_constants(cls):
    """Yield (sub_name, value) for UPPER_SNAKE_CASE str attrs declared on cls itself.

    Uses ``vars(cls)`` rather than ``dir(cls)`` so that inherited attributes
    (e.g., DSAC inheriting from SAC) are NOT treated as duplicates of the
    parent's constants.
    """
    for sub, val in vars(cls).items():
        if sub.startswith("_"):
            continue
        if isinstance(val, str) and sub.isupper():
            yield sub, val


def _all_constants():
    """Yield (qualified_name, value) for every UPPER_SNAKE_CASE string constant.

    Module-level constants and per-class constants are both included.
    Per-class iteration uses ``vars(cls)`` so inheritance does not duplicate.
    """
    for name in dir(schema):
        if name.startswith("_"):
            continue
        obj = getattr(schema, name)
        if isinstance(obj, str) and name.isupper():
            yield name, obj
        elif isinstance(obj, type) and obj.__module__ == schema.__name__:
            for sub, val in _own_class_constants(obj):
                yield f"{name}.{sub}", val


def test_all_tags_match_canonical_pattern():
    bad = [(n, v) for n, v in _all_constants() if not _VALID_TAG.match(v)]
    assert not bad, f"Tags violating canonical pattern: {bad}"


def test_no_duplicate_tag_values():
    seen: dict[str, str] = {}
    dups: list[tuple[str, str, str]] = []
    for name, value in _all_constants():
        if value in seen and seen[value] != name:
            dups.append((value, seen[value], name))
        else:
            seen[value] = name
    assert not dups, f"Duplicate tag values across constants: {dups}"


def test_registry_contains_every_constant():
    expected = {v for _, v in _all_constants()}
    missing = expected - schema.REGISTRY
    assert not missing, f"Constants missing from REGISTRY: {missing}"


def test_strip_worker_suffix_removes_trailing_worker_id():
    assert (
        schema.strip_worker_suffix("rollout/episode_reward/worker_0")
        == "rollout/episode_reward"
    )
    assert schema.strip_worker_suffix("loss/actor/worker_42") == "loss/actor"
    assert schema.strip_worker_suffix("loss/actor") == "loss/actor"


def test_is_registered_scalar_accepts_worker_suffix():
    assert schema.is_registered_scalar("rollout/episode_reward/worker_3")
    assert schema.is_registered_scalar(schema.LOSS_ACTOR)
    assert not schema.is_registered_scalar("Performance/Reward")
    assert not schema.is_registered_scalar("loss/actor/wrkr_3")  # bad suffix


def test_is_registered_histogram_validates_two_level_prefix():
    assert schema.is_registered_histogram("weights/actor/fc1.weight")
    assert schema.is_registered_histogram("grads/critic/conv.bias")
    assert schema.is_registered_histogram("weights/q1/layer.0.weight")
    assert schema.is_registered_histogram("weights/q/layer.0.weight")  # DQN
    assert not schema.is_registered_histogram("weights/unknown/x")
    assert not schema.is_registered_histogram("foo/actor/x")
    assert not schema.is_registered_histogram("weights/actor")  # too short


def test_mandatory_subset_of_registry():
    from tensoraerospace.agent.metrics import MANDATORY_METRICS

    assert set(MANDATORY_METRICS).issubset(schema.REGISTRY)
