"""Reproducible control-parameter search with native metrics and Optuna TPE."""

from __future__ import annotations

import copy
import inspect
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)
from optuna.trial import TrialState

from .agent import _replace_paths
from .base import HyperParamOptimizationOptuna
from .metrics import TrialRejected

Objective = Callable[[dict[str, Any], int], float | Mapping[str, float | None]]


@contextmanager
def _random_state(seed):
    """Isolate Python/NumPy and already-imported Torch RNGs, including failures."""
    import sys

    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _seeds(values):
    """Validate distinct uint32 seeds and normalize NumPy integers to Python ints."""
    values = tuple(values)
    if not values or any(
        isinstance(s, bool)
        or not isinstance(s, (int, np.integer))
        or not 0 <= s < 2**32
        for s in values
    ):
        raise ValueError("seeds must be a nonempty sequence of uint32 integers")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate seeds would overweight a realization")
    return tuple(int(s) for s in values)


def _objectives(objective):
    """Normalize a callable or named scenario mapping into validated evaluators."""
    result = {"default": objective} if callable(objective) else dict(objective)
    if not result or any(
        not isinstance(name, str) or not name or not callable(fn)
        for name, fn in result.items()
    ):
        raise ValueError(
            "objective must be callable or a nonempty name/callable mapping"
        )
    return result


@dataclass
class OptimizationResult:
    """Best feasible parameters and the complete study (no trained weights)."""

    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    study: optuna.Study
    _factory: Callable | None = field(default=None, repr=False)
    _validation: Callable | None = field(default=None, repr=False)

    def validate(self, *, seeds):
        """Re-run selected parameters on new seeds without adding search trials."""
        if self._validation is None:
            raise RuntimeError("No evaluation function attached to this result")
        return self._validation(seeds=seeds)

    def create_agent(self, seed=0):
        """Construct a fresh, untrained agent with the selected parameters."""
        if self._factory is None:
            raise RuntimeError("create_agent requires ControlOptimizer.for_agent")
        seed = _seeds([seed])[0]
        with _random_state(seed):
            return self._factory(copy.deepcopy(self.best_params), seed)

    def save(self, path):
        """Save parameters and per-case metrics as portable, strict JSON."""

        def safe(value):
            """Convert nested results to JSON-compatible values, replacing nonfinite
            numbers.
            """
            if isinstance(value, Mapping):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        payload = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_trial_number": self.best_trial_number,
            "protocol": self.study.user_attrs.get("control_protocol"),
            "execution": self.study.user_attrs.get("execution"),
            "trials": [
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    "params": t.params,
                    "details": t.user_attrs,
                    "duration_seconds": (
                        t.duration.total_seconds() if t.duration else None
                    ),
                }
                for t in self.study.trials
            ],
        }
        Path(path).write_text(
            json.dumps(safe(payload), indent=2, allow_nan=False) + "\n"
        )

    def plot_history(self):
        """Plot feasible trial values and the running best, with actual trial numbers."""
        import matplotlib.pyplot as plt

        trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        x = [t.number for t in trials]
        y = [float(cast(float, t.value)) for t in trials]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.scatter(x, y, alpha=0.65, label="Feasible trial")
        ax.step(x, np.minimum.accumulate(y), where="post", label="Best so far")
        ax.set(
            xlabel="Trial",
            ylabel="Objective (lower is better)",
            title="Controller tuning",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        return fig


def _validate_search_space(search_space):
    """Reject empty search spaces, invalid names and unsupported Optuna distributions."""
    if not search_space or any(
        not isinstance(k, str)
        or not k
        or not isinstance(
            v, (FloatDistribution, IntDistribution, CategoricalDistribution)
        )
        for k, v in search_space.items()
    ):
        raise ValueError(
            "search_space needs named Float, Int or Categorical distributions"
        )
    for distribution in search_space.values():
        if isinstance(distribution, CategoricalDistribution) and any(
            not isinstance(choice, (str, bool, int, float, type(None)))
            or isinstance(choice, float)
            and not math.isfinite(choice)
            for choice in distribution.choices
        ):
            raise ValueError("Categorical choices must be finite JSON scalar values")


def _validate_run_options(n_trials, timeout, target, patience, show_progress_bar):
    """Validate trial budgets, timeout, target, patience and progress options."""
    if not isinstance(show_progress_bar, bool):
        raise ValueError("show_progress_bar must be a boolean")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if target is not None and not np.isfinite(target):
        raise ValueError("target must be finite")
    if patience is not None and (
        isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
    ):
        raise ValueError("patience must be a positive integer")


class ControlOptimizer(HyperParamOptimizationOptuna):
    """Minimize a scalar metric across fixed scenarios and random seeds.

    ``objective(params, seed)`` returns a number or a metric dictionary. A
    mapping ``{scenario_name: objective}`` evaluates every scenario for each
    seed. ``constraints`` maps metric names to inclusive upper bounds, checked
    on EVERY case, not their mean. Nonfinite objectives, unmet bounds and
    TrialRejected/FloatingPointError/OverflowError prune a candidate; programming
    errors propagate. Missing metric names are errors, missing settling (None)
    is infeasible. Successful trials always finish all cases at full horizon.

    Speed comes from seeded TPE, log search ranges, enqueueing known settings,
    median pruning between complete cases, and early target/patience stopping.
    There is no assumption that CPI can reach zero. Trials are serial because
    many controller constructors seed global Torch/NumPy RNGs. For independent
    process workers use shared Optuna storage (PostgreSQL for concurrent use).
    ``experiment_id`` is required with storage: change it when the plant,
    training budget, reference, units or callback behavior changes.
    """

    def __init__(
        self,
        search_space: Mapping[str, BaseDistribution],
        objective: Objective | Mapping[str, Objective],
        *,
        metric="cpi",
        constraints=None,
        seeds=(0, 1, 2),
        aggregation="mean",
        seed=0,
        startup_trials=5,
        sampler=None,
        pruner=None,
        storage=None,
        study_name=None,
        load_if_exists=False,
        experiment_id=None,
    ):
        self.search_space = dict(search_space)
        _validate_search_space(self.search_space)
        self.objectives = _objectives(objective)
        self.seeds = _seeds(seeds)
        if aggregation not in ("mean", "worst"):
            raise ValueError("aggregation must be 'mean' or 'worst'")
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a nonempty string")
        self.metric, self.aggregation = metric, aggregation
        self.constraints = dict(constraints or {})
        if any(not np.isfinite(v) for v in self.constraints.values()):
            raise ValueError("Constraint upper bounds must be finite")
        seed = _seeds([seed])[0]
        if not isinstance(startup_trials, int) or startup_trials < 1:
            raise ValueError("startup_trials must be a positive integer")
        if storage is not None and (not experiment_id or not study_name):
            raise ValueError("Persistent searches require study_name and experiment_id")
        self._factory: Callable | None = None
        super().__init__(
            direction="minimize",
            sampler=(
                sampler
                if sampler is not None
                else optuna.samplers.TPESampler(
                    seed=seed, n_startup_trials=startup_trials
                )
            ),
            pruner=(
                pruner
                if pruner is not None
                else optuna.pruners.MedianPruner(
                    n_startup_trials=startup_trials, n_min_trials=3
                )
            ),
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )
        protocol = {
            "version": 1,
            "experiment_id": experiment_id,
            "space": {k: distribution_to_json(v) for k, v in self.search_space.items()},
            "metric": metric,
            "constraints": self.constraints,
            "seeds": list(self.seeds),
            "scenarios": list(self.objectives),
            "aggregation": aggregation,
        }
        previous = self.study.user_attrs.get("control_protocol")
        if previous is not None and previous != protocol:
            raise ValueError("Stored search protocol differs; use a new study_name")
        if previous is None and self.study.trials:
            raise ValueError("Cannot reuse a study without a control search protocol")
        self.study.set_user_attr("control_protocol", protocol)

    @classmethod
    def for_agent(
        cls,
        agent_class,
        search_space,
        evaluate,
        *,
        agent_kwargs,
        seed_path=None,
        **options,
    ):
        """Build agents from nested constructor parameters for every evaluation.

        ``agent_kwargs(seed)`` must return fresh resources if construction needs
        an environment. A plain mapping is deep-copied. ``seed_path`` explicitly
        overrides e.g. ``config.seed``; it may not itself be optimized.
        Caller evaluation functions own environment cleanup, including failures.
        """
        if seed_path is not None and any(
            p == seed_path
            or p.startswith(seed_path + ".")
            or seed_path.startswith(p + ".")
            for p in search_space
        ):
            raise ValueError("seed_path cannot overlap optimized parameters")

        template = (
            agent_kwargs if callable(agent_kwargs) else copy.deepcopy(agent_kwargs)
        )

        constructor_parameters = inspect.signature(agent_class).parameters

        def factory(params, seed):
            """Construct a fresh agent with copied settings, sampled parameters and
            seed.
            """
            kwargs = (
                dict(template(seed)) if callable(template) else copy.deepcopy(template)
            )
            updates = dict(params)
            if seed_path is not None:
                updates[seed_path] = seed
            for path in updates:
                root = path.split(".", 1)[0]
                if root not in kwargs and root in constructor_parameters:
                    parameter = constructor_parameters[root]
                    if parameter.kind in (
                        parameter.POSITIONAL_OR_KEYWORD,
                        parameter.KEYWORD_ONLY,
                    ):
                        # A searched scalar can directly supply a required argument.
                        # Nested paths still require a concrete constructor config.
                        kwargs[root] = (
                            None
                            if parameter.default is parameter.empty
                            else copy.deepcopy(parameter.default)
                        )
            return agent_class(**_replace_paths(kwargs, updates))

        objectives = {}
        for name, evaluator in _objectives(evaluate).items():

            def objective(params, seed, evaluator=evaluator):
                """Evaluate a newly constructed agent on the bound scenario and seed."""
                return evaluator(factory(params, seed), seed)

            objectives[name] = objective
        optimizer = cls(search_space, objectives, **options)
        optimizer._factory = factory
        return optimizer

    def _suggest(self, trial):
        """Sample each configured distribution through the active Optuna trial."""
        result = {}
        for name, distribution in self.search_space.items():
            if isinstance(distribution, FloatDistribution):
                result[name] = trial.suggest_float(
                    name,
                    distribution.low,
                    distribution.high,
                    log=distribution.log,
                    step=distribution.step,
                )
            elif isinstance(distribution, IntDistribution):
                result[name] = trial.suggest_int(
                    name,
                    distribution.low,
                    distribution.high,
                    log=distribution.log,
                    step=distribution.step,
                )
            else:
                assert isinstance(distribution, CategoricalDistribution)
                result[name] = trial.suggest_categorical(name, distribution.choices)
        return result

    def _measure(self, objective, params, seed):
        """Evaluate one seeded case and return metrics plus any feasibility violation."""
        with _random_state(seed):
            raw = objective(copy.deepcopy(params), seed)
        metrics = dict(raw) if isinstance(raw, Mapping) else {self.metric: float(raw)}
        # Check requested names before handling invalid trajectories: typos must fail.
        for key in (self.metric, *self.constraints):
            if key not in metrics:
                raise KeyError(f"Evaluation did not return metric {key!r}")
        metrics = {k: None if v is None else float(v) for k, v in metrics.items()}
        value = metrics[self.metric]
        reason = None
        if value is None or not np.isfinite(value):
            reason = f"Nonfinite objective: {self.metric}"
        for key, bound in self.constraints.items():
            measured = metrics[key]
            if measured is None or not np.isfinite(measured) or measured > bound:
                reason = (
                    f"Constraint {key}: no finite value (required <= {bound})"
                    if measured is None or not np.isfinite(measured)
                    else f"Constraint {key}: {measured} exceeds {bound}"
                )
                break
        return metrics, reason

    def _aggregate(self, values):
        """Reduce case scores using the configured mean or worst-case objective."""
        return float(np.mean(values) if self.aggregation == "mean" else max(values))

    def _objective(self, trial):
        """Score all scenario/seed cases, retaining diagnostics for rejected trials."""
        params = self._suggest(trial)
        rows = []
        values: list[float] = []
        count = len(self.seeds) * len(self.objectives)
        try:
            for seed in self.seeds:
                for name, objective in self.objectives.items():
                    metrics, reason = self._measure(objective, params, seed)
                    rows.append(
                        {
                            "scenario": name,
                            "seed": seed,
                            "metrics": metrics,
                            "reason": reason,
                        }
                    )
                    trial.set_user_attr("evaluations", rows)
                    value = metrics[self.metric]
                    if reason is not None:
                        if value is not None and np.isfinite(value):
                            trial.report(self._aggregate([*values, value]), len(values))
                        raise TrialRejected(reason)
                    values.append(cast(float, value))
                    trial.report(self._aggregate(values), len(values) - 1)
                    if len(values) < count and not trial.user_attrs.get(
                        "initial_params"
                    ):
                        if trial.should_prune():
                            raise TrialRejected(
                                "Unpromising result after complete evaluation cases"
                            )
        except (
            TrialRejected,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as exc:
            trial.set_user_attr("rejection_reason", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        return self._aggregate(values)

    def _run_trials(self, n_trials, *, timeout, callbacks):
        """Serial driver; callbacks may request a stop by returning True."""

        def notify(study, trial):
            """Invoke completion callbacks and stop Optuna when a callback requests it."""
            stopped = False
            for callback in callbacks:
                stopped = bool(callback(study, trial)) or stopped
            if stopped:
                study.stop()

        self.run_optimization(
            self._objective, n_trials, timeout=timeout, callbacks=[notify]
        )

    def _enqueue_initial_params(self, initial_params):
        """Validate a partial warm start and queue it ahead of sampled candidates."""
        if initial_params is not None:
            if not set(initial_params) <= set(self.search_space):
                raise ValueError(
                    "initial_params contains parameters outside search_space"
                )
            for name, value in initial_params.items():
                distribution = self.search_space[name]
                # Optuna otherwise only warns and evaluates out-of-range queued values.
                value = initial_params[name]
                if isinstance(distribution, CategoricalDistribution):
                    valid = value in distribution.choices
                else:
                    assert isinstance(
                        distribution, (FloatDistribution, IntDistribution)
                    )
                    valid = isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    )
                    valid = (
                        valid
                        and math.isfinite(value)
                        and distribution.low <= value <= distribution.high
                    )
                    if isinstance(distribution, IntDistribution):
                        valid = valid and isinstance(value, int)
                    if valid and distribution.step is not None:
                        steps = (value - distribution.low) / distribution.step
                        valid = math.isclose(
                            steps, round(steps), rel_tol=0.0, abs_tol=1e-8
                        )
                if not valid:
                    raise ValueError(
                        f"Initial parameter {name!r} lies outside its distribution"
                    )
            self.study.enqueue_trial(
                dict(initial_params),
                user_attrs={"initial_params": True},
                skip_if_exists=True,
            )

    def optimize(
        self,
        n_trials=30,
        *,
        initial_params=None,
        timeout=None,
        target=None,
        patience=None,
        show_progress_bar=True,
    ) -> OptimizationResult:
        """Run up to n_trials additional candidates, returning the best feasible one.

        ``initial_params`` queues a full or partial parameter dictionary first;
        unspecified parameters are sampled. It is evaluated without statistical
        pruning (physical constraints still apply). Timeout is checked between
        trials, not a per-rollout interrupt. Target applies to
        the aggregate full-case score, patience to trials since improvement.
        By default a notebook/terminal bar shows completed trials, elapsed time,
        estimated time remaining and the best feasible metric. Set
        ``show_progress_bar=False`` for noninteractive batch runs.
        """
        _validate_run_options(n_trials, timeout, target, patience, show_progress_bar)
        self._enqueue_initial_params(initial_params)
        complete = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        best = min((cast(float, t.value) for t in complete), default=float("inf"))
        stale = 0
        stop_reason = None

        def stop(study, trial):
            """Track improvement and request a stop when the target or patience is
            reached.
            """
            nonlocal best, stale, stop_reason
            if trial.state == TrialState.COMPLETE and trial.value < best:
                best, stale = trial.value, 0
            else:
                stale += 1
            if target is not None and best <= target:
                stop_reason = "stopped (target reached)"
                return True
            elif patience is not None and np.isfinite(best) and stale >= patience:
                stop_reason = "stopped (patience)"
                return True
            return False

        if target is None or best > target:
            from ._progress import search_progress

            initial_best = min(
                complete, key=lambda trial: cast(float, trial.value), default=None
            )
            with search_progress(
                total=n_trials,
                metric=self.metric,
                best_trial=initial_best,
                enabled=show_progress_bar,
            ) as progress:
                self._run_trials(
                    n_trials,
                    timeout=timeout,
                    callbacks=[stop, progress.update],
                )
                if stop_reason is not None:
                    progress.reason = stop_reason
                elif progress.completed < n_trials:
                    progress.reason = (
                        "stopped (timeout)" if timeout is not None else "stopped"
                    )

        if not any(t.state == TrialState.COMPLETE for t in self.study.trials):
            raise RuntimeError(self._no_feasible_message())
        best_trial = self.study.best_trial
        return OptimizationResult(
            dict(best_trial.params),
            float(cast(float, best_trial.value)),
            best_trial.number,
            self.study,
            self._factory,
            lambda *, seeds: self.validate(seeds=seeds, params=best_trial.params),
        )

    def diagnostics(self):
        """Summarize observed metrics and rejections without another simulation.

        Constraint counts refer to evaluated scenario/seed cases, not trials.
        Bounds are checked independently, so their individual minima need not
        belong to the same candidate. ``lowest_fully_evaluated`` requires all
        configured cases and a finite objective; it may still be infeasible
        and is never substituted for a successful optimization result.
        """
        trials = self.study.trials
        rows = [
            row for trial in trials for row in trial.user_attrs.get("evaluations", [])
        ]
        constraints = {}
        for name, bound in self.constraints.items():
            observed = [row["metrics"][name] for row in rows if name in row["metrics"]]
            finite = [
                float(value)
                for value in observed
                if value is not None and np.isfinite(value)
            ]
            missing = len(observed) - len(finite)
            constraints[name] = {
                "upper_bound": bound,
                "evaluated_cases": len(observed),
                "violating_cases": missing + sum(value > bound for value in finite),
                "nonfinite_cases": missing,
                "lowest_observed": min(finite) if finite else None,
            }
        expected = {(name, seed) for name in self.objectives for seed in self.seeds}
        fully_evaluated = []
        for trial in trials:
            cases = trial.user_attrs.get("evaluations", [])
            if (
                len(cases) != len(expected)
                or {(row["scenario"], row["seed"]) for row in cases} != expected
            ):
                continue
            values = [row["metrics"].get(self.metric) for row in cases]
            if any(value is None or not np.isfinite(value) for value in values):
                continue
            fully_evaluated.append(
                {
                    "trial_number": trial.number,
                    "params": dict(trial.params),
                    "value": self._aggregate(values),
                    "feasible": trial.state == TrialState.COMPLETE,
                }
            )
        return {
            "metric": self.metric,
            "trials": len(trials),
            "trial_states": dict(Counter(trial.state.name for trial in trials)),
            "evaluated_cases": len(rows),
            "constraints": constraints,
            "lowest_fully_evaluated": (
                min(fully_evaluated, key=lambda row: row["value"])
                if fully_evaluated
                else None
            ),
            "rejection_reasons": dict(
                Counter(
                    trial.user_attrs["rejection_reason"]
                    for trial in trials
                    if "rejection_reason" in trial.user_attrs
                )
            ),
        }

    def _no_feasible_message(self):
        """Explain rejected searches using recorded constraints and observed scores."""
        report = self.diagnostics()
        states = ", ".join(
            f"{name}={count}" for name, count in report["trial_states"].items()
        )
        lines = [f"No feasible trial completed. Trials: {report['trials']} ({states})."]
        for name, summary in report["constraints"].items():
            if not summary["violating_cases"]:
                continue
            best = summary["lowest_observed"]
            observed = (
                f"lowest observed {best:.6g}"
                if best is not None
                else "no finite value recorded"
            )
            lines.append(
                f"- {name} <= {summary['upper_bound']:.6g}: "
                f"violated in {summary['violating_cases']}/{summary['evaluated_cases']} evaluated cases; {observed}."
            )
        measured = report["lowest_fully_evaluated"]
        if measured is not None:
            lines.append(
                f"Lowest {self.metric} across fully evaluated trials: {measured['value']:.6g} "
                f"(trial #{measured['trial_number']}, still infeasible)."
            )
        others = {
            reason: count
            for reason, count in report["rejection_reasons"].items()
            if not reason.startswith("Constraint ")
        }
        for reason, count in sorted(others.items(), key=lambda item: -item[1])[:3]:
            lines.append(f"- {count} trial(s): {reason}")
        if (
            self.metric in self.constraints
            and report["constraints"][self.metric]["violating_cases"]
        ):
            lines.append(
                f"constraints[{self.metric!r}] is a hard ceiling, not a search target. "
                "To keep the best feasible result even when a desired score is not reached, "
                "use optimize(target=...) without that objective ceiling. Other constraints still apply."
            )
        lines.append(
            "Call diagnostics() for details; no constraint was relaxed and no rejected candidate was selected."
        )
        return "\n".join(lines)

    def validate(self, *, seeds: Sequence[int], params=None, objective=None):
        """Evaluate selected settings on held-out seeds/scenarios without tuning.

        Returns per-case metrics, feasibility and the same aggregate objective.
        Validation never inserts trials into the search or selects a new winner.
        """
        seeds = _seeds(seeds)
        params = self.get_best_param() if params is None else dict(params)
        objectives = self.objectives if objective is None else _objectives(objective)
        rows, values, feasible = [], [], True
        for seed in seeds:
            for name, evaluate in objectives.items():
                try:
                    metrics, reason = self._measure(evaluate, params, seed)
                except (
                    TrialRejected,
                    FloatingPointError,
                    OverflowError,
                    np.linalg.LinAlgError,
                ) as exc:
                    metrics, reason = {}, str(exc)
                rows.append(
                    {
                        "scenario": name,
                        "seed": seed,
                        "metrics": metrics,
                        "reason": reason,
                    }
                )
                feasible &= reason is None
                value = metrics.get(self.metric)
                values.append(
                    value if value is not None and np.isfinite(value) else float("inf")
                )
        return {
            "value": self._aggregate(values),
            "feasible": feasible,
            "evaluations": rows,
        }
