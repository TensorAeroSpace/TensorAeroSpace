"""Process-isolated, deterministic batches for declarative controller experiments."""

from __future__ import annotations

import copy
import multiprocessing
import os
import pickle
import time
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import optuna
from optuna.trial import TrialState

from .control import ControlOptimizer


@contextmanager
def torch_thread_limit(threads):
    """Bound small CPU networks without changing the caller's lasting settings."""
    import torch

    previous = torch.get_num_threads()
    changed = threads is not None and threads != previous
    try:
        if changed:
            torch.set_num_threads(threads)
        yield
    finally:
        if changed:
            torch.set_num_threads(previous)


class _RecordedTrial(optuna.trial.FixedTrial):
    """Reuse the serial evaluation/constraint code without a shared study."""

    def __init__(self, params, user_attrs):
        super().__init__(params)
        self.intermediate_values = {}
        for key, value in user_attrs.items():
            self.set_user_attr(key, value)

    def report(self, value, step):
        self.intermediate_values[step] = value


@dataclass
class _Evaluation:
    state: TrialState
    value: float | None
    user_attrs: dict
    intermediate_values: dict


_worker_optimizer = None


def _initialize_worker(tuner, verbosity):
    global _worker_optimizer
    optuna.logging.set_verbosity(max(verbosity, optuna.logging.WARNING))
    if tuner.torch_threads is not None:
        import torch

        torch.set_num_threads(tuner.torch_threads)

    def objective(params, seed):
        return tuner.simulate(params, seed=seed).metrics

    _worker_optimizer = ControlOptimizer(
        tuner.search_space,
        objective,
        metric=tuner.metric,
        constraints=tuner.constraints,
        seeds=tuner.seeds,
        seed=tuner.seed,
        pruner=optuna.pruners.NopPruner(),
    )


def _evaluate_candidate(params, user_attrs):
    assert _worker_optimizer is not None
    trial = _RecordedTrial(params, user_attrs)
    trial.set_user_attr("worker_pid", os.getpid())
    started = time.monotonic()
    try:
        value = _worker_optimizer._objective(trial)
        state = TrialState.COMPLETE
    except optuna.TrialPruned:
        value, state = None, TrialState.PRUNED
    trial.set_user_attr("evaluation_seconds", time.monotonic() - started)
    return _Evaluation(state, value, trial.user_attrs, trial.intermediate_values)


class ParallelControllerOptimizer(ControlOptimizer):
    """Private driver for picklable ControllerTuner experiments, with no pruning.

    Ask/tell and samplers live in the parent. Workers reuse the exact serial
    evaluator, including all seeds, constraints and physical failure handling.
    Results enter the study in trial-number order, so scheduling cannot change
    future proposals. A warm-start candidate finishes before the first batch.
    """

    def __init__(self, *args, tuner, n_jobs, **kwargs):
        # The live tuner will point back to this optimizer (and its closures).
        # Workers only need a frozen experiment, never its old study or history.
        self._tuner = copy.copy(tuner)
        self._tuner.optimizer = None
        try:
            pickle.dumps(self._tuner)
        except (TypeError, AttributeError, pickle.PicklingError) as exc:
            raise ValueError(
                "Parallel search requires picklable environment/controller settings; "
                "use importable classes/functions instead of notebook-local functions "
                "or lambdas, or n_jobs=1"
            ) from exc
        self._n_jobs = n_jobs
        super().__init__(*args, **kwargs)
        if not isinstance(self.study.pruner, optuna.pruners.NopPruner):
            raise ValueError("Parallel controller batches require NopPruner")

    def _run_trials(self, n_trials, *, timeout, callbacks):
        started = time.monotonic()
        submitted, stopped = 0, False
        pending: list[tuple[optuna.Trial, Future[_Evaluation] | None]] = []
        # Spawn is safe after Torch/CUDA/notebook initialization; fork is not.
        with ProcessPoolExecutor(
            max_workers=min(self._n_jobs, n_trials),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(self._tuner, optuna.logging.get_verbosity()),
        ) as pool:
            try:
                while submitted < n_trials and not stopped:
                    if timeout is not None and time.monotonic() - started >= timeout:
                        break
                    pending = []
                    for _ in range(min(self._n_jobs, n_trials - submitted)):
                        trial = self.study.ask()
                        pending.append((trial, None))
                        params = self._suggest(trial)
                        submitted_future = pool.submit(
                            _evaluate_candidate, params, trial.user_attrs
                        )
                        pending[-1] = (trial, submitted_future)
                        submitted += 1
                        if trial.user_attrs.get("initial_params"):
                            break
                    for trial, future in pending:
                        assert future is not None
                        stopped = (
                            self._record_evaluation(trial, future, callbacks) or stopped
                        )
            except BaseException:
                # Do not leave phantom RUNNING trials or hide programming errors.
                self._abort_batch(pending)
                raise

    def _record_evaluation(self, trial, future, callbacks):
        stopped = False
        evaluation = future.result()
        for key, value in evaluation.user_attrs.items():
            trial.set_user_attr(key, value)
        for step, value in evaluation.intermediate_values.items():
            trial.report(value, step)
        frozen = self.study.tell(trial, evaluation.value, state=evaluation.state)
        for callback in callbacks:
            stopped = bool(callback(self.study, frozen)) or stopped
        return stopped

    def _abort_batch(self, pending):
        for trial, future in pending:
            if future is not None:
                future.cancel()
            frozen = self.study.trials[trial.number]
            if frozen.state == TrialState.RUNNING:
                trial.set_user_attr("execution_error", "Parallel batch aborted")
                self.study.tell(trial, state=TrialState.FAIL)
