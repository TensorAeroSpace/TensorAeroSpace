"""Experiment comparisons and atomic checkpoints for continued controller search."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def same_setting(first, second):
    """Compare native settings without ambiguous array/dataclass equality.

    Mapping order matters: reference channels and sampled parameter order affect
    the experiment and its random draws. Callable identity is preserved by
    deepcopy; external changes inside a callback remain the caller's concern.
    """
    if first is second:
        return True
    if type(first) is not type(second):
        return False
    if isinstance(first, np.ndarray):
        return first.dtype == second.dtype and np.array_equal(
            first, second, equal_nan=first.dtype.kind in "fc"
        )
    if isinstance(first, Mapping):
        return list(first) == list(second) and all(
            same_setting(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            same_setting(a, b) for a, b in zip(first, second)
        )
    if is_dataclass(first) and not isinstance(first, type):
        return all(
            same_setting(getattr(first, f.name), getattr(second, f.name))
            for f in fields(first)
        )
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def write_checkpoint(path, payload):
    """Serialize fully before replacing an existing checkpoint atomically."""
    try:
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except (TypeError, AttributeError, pickle.PicklingError) as exc:
        raise ValueError(
            "Checkpoint settings must be picklable; use importable functions/classes "
            "instead of notebook-local functions or lambdas"
        ) from exc
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
