"""Helpers for verifying that an agent wrote canonical TB tags during train()."""

from __future__ import annotations

import glob
import os
from typing import Set

import pytest


def read_event_scalar_tags(log_dir: str) -> Set[str]:
    """Return the set of scalar tags found across all event files under log_dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        pytest.skip("tensorboard package not installed")

    if not os.path.isdir(log_dir):
        return set()
    tags: Set[str] = set()
    for ev in glob.glob(
        os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True
    ):
        ea = EventAccumulator(os.path.dirname(ev))
        ea.Reload()
        tags.update(ea.Tags().get("scalars", []))
    return tags


def assert_tags_present(log_dir: str, required: set[str]) -> None:
    """Raise AssertionError if any of ``required`` is missing from the event files."""
    found = read_event_scalar_tags(log_dir)
    missing = required - found
    assert not missing, (
        f"Missing tags in {log_dir}: sorted={sorted(missing)} " f"found={sorted(found)}"
    )


def test_helper_reads_empty_dir(tmp_path):
    """Smoke check that the helper handles a directory with no event files."""
    assert read_event_scalar_tags(str(tmp_path)) == set()
