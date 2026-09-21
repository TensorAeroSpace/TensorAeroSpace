"""Constructor-based tuning shared by controllers with different learning APIs.

No Optuna import is needed until ``optimize`` is called. Training and rollout
remain explicit in the evaluation function; this avoids guessing whether an
agent's ``learn`` means one update or a complete training run.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np


def _path_value(value, key):
    """Read one validated constructor path component."""
    if isinstance(value, dict):
        if key not in value:
            raise KeyError(f"Unknown constructor parameter path: {key}")
        current = value[key]
    elif is_dataclass(value) and not isinstance(value, type):
        if key not in {f.name for f in fields(value) if f.init}:
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(value, key)
    elif isinstance(value, (list, tuple, np.ndarray)):
        if not key.isdigit() or str(int(key)) != key or int(key) >= len(value):
            raise KeyError(f"Invalid parameter index: {key}")
        current = value[int(key)]
    else:
        raise TypeError("Paths support dataclasses, dictionaries and sequences")
    return current


def _replace_paths(value: Any, updates: dict[str, Any]) -> Any:
    """Rebuild nested constructor settings, validating dataclasses atomically."""
    groups: dict[str, dict[str, Any]] = {}
    for path, replacement in updates.items():
        head, sep, tail = path.partition(".")
        if not head or (sep and not tail):
            raise ValueError("Parameter paths must have nonempty components")
        groups.setdefault(head, {})[tail if sep else ""] = replacement
    changes = {}
    for key, children in groups.items():
        if "" in children and len(children) != 1:
            raise ValueError(f"Overlapping parameter paths at {key!r}")
        current = _path_value(value, key)
        changes[key] = (
            copy.deepcopy(children[""])
            if "" in children
            else _replace_paths(current, children)
        )
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **changes)
    # Keep unrelated constructor resources (e.g. a fresh env) by identity.
    # Plain agent_kwargs have already been deep-copied by the factory.
    result: Any = list(value) if isinstance(value, (list, tuple)) else copy.copy(value)
    if isinstance(value, np.ndarray):
        result = value.astype(
            np.result_type(value, *map(np.asarray, changes.values())), copy=True
        )
    for key, replacement in changes.items():
        result[key if isinstance(result, dict) else int(key)] = replacement
    return tuple(result) if isinstance(value, tuple) else result


class OptimizableAgent:
    """Expose the same hyperparameter search entry point on control agents."""

    @classmethod
    def optimize(
        cls,
        search_space,
        evaluate,
        *,
        agent_kwargs,
        n_trials=30,
        seed_path=None,
        initial_params=None,
        timeout=None,
        target=None,
        patience=None,
        show_progress_bar=True,
        **search_options,
    ):
        """Minimize a rollout metric using fresh agents for every scenario/seed.

        ``agent_kwargs`` is a constructor dictionary or ``callable(seed)``
        returning one (use the latter for fresh environments/checkpoints).
        Search paths address constructor inputs, e.g. ``config.actor_lr`` or
        ``actor_settings.learning_rate``. ``evaluate(agent, seed)`` returns a
        scalar or metric dictionary and owns training, rollout and env cleanup.
        It may also be a mapping of scenario names to evaluation functions.

        This class method does not retune an existing instance. The returned
        result contains parameters, not a trained policy. See ControlOptimizer.
        """
        from .control import ControlOptimizer

        optimizer = ControlOptimizer.for_agent(
            cls,
            search_space,
            evaluate,
            agent_kwargs=agent_kwargs,
            seed_path=seed_path,
            **search_options,
        )
        return optimizer.optimize(
            n_trials,
            initial_params=initial_params,
            timeout=timeout,
            target=target,
            patience=patience,
            show_progress_bar=show_progress_bar,
        )
