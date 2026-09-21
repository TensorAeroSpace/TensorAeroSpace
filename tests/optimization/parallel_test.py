"""Real process isolation, equivalent rollouts and bounded search termination."""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from optuna.trial import TrialState

from tensoraerospace.optimization import ControllerTuner, Float, Step, _parallel


def settings(**updates):
    value = dict(
        env="NonlinearB737-v0",
        env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
        reference={"theta": Step(0.1, at=0.1, unit="deg")},
        controller="ihdp",
        duration=0.3,
        method="random",
        seed=42,
    )
    value.update(updates)
    return value


def trial_signature(trial):
    return (
        trial.params,
        trial.state,
        trial.value,
        trial.user_attrs.get("evaluations"),
        trial.user_attrs.get("rejection_reason"),
        trial.intermediate_values,
    )


def test_parallel_random_matches_serial_every_case_and_replays_winner():
    tuner = ControllerTuner(**settings(seeds=(0, 13)))
    serial = tuner.optimize(5, show_progress_bar=False)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    parallel = tuner.optimize(5, n_jobs=2, show_progress_bar=False)
    assert [trial_signature(t) for t in parallel.study.trials] == [
        trial_signature(t) for t in serial.study.trials
    ]
    assert parallel.best_params == serial.best_params
    np.testing.assert_array_equal(parallel.best_run.output, serial.best_run.output)
    np.testing.assert_array_equal(parallel.best_run.actions, serial.best_run.actions)
    assert parallel.best_run.time[-1] == 0.3
    pids = {t.user_attrs["worker_pid"] for t in parallel.study.trials}
    assert os.getpid() not in pids
    assert len(pids) == 2
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("method", ["tpe", "genetic"])
def test_batched_adaptive_sampler_is_reproducible(method):
    tuner = ControllerTuner(
        **settings(
            controller="aa_indi",
            method=method,
            method_options=(
                {"population_size": 2}
                if method == "genetic"
                else {"n_startup_trials": 2}
            ),
        )
    )
    first = tuner.optimize(6, n_jobs=2, show_progress_bar=False)
    second = tuner.optimize(6, n_jobs=2, show_progress_bar=False)
    assert [trial_signature(t) for t in first.study.trials] == [
        trial_signature(t) for t in second.study.trials
    ]


@pytest.mark.parametrize(
    "controller,parameter,value,options",
    [
        ("aidi", "rate_gain", 3.0, {}),
        ("iadp", "gamma", 0.95, {}),
        ("imgdhp", "actor_lr", 0.001, {}),
        ("et_dhp", "actor_lr", 0.001, {"model_samples": 16, "model_epochs": 2}),
        ("mpc", "lr", 0.03, {"iters": 2}),
        ("hdp", "actor_lr", 0.001, {}),
    ],
)
def test_other_profiles_use_same_native_evaluation_in_worker(
    controller, parameter, value, options
):
    kwargs = settings(
        controller=controller,
        controller_options=options,
        search_space={parameter: Float(value, value)},
    )
    if controller == "hdp":
        kwargs.update(
            env="ImprovedB747-v0",
            env_kwargs={"initial_state": [0.0] * 4, "dt": 0.02},
        )
    tuner = ControllerTuner(**kwargs)
    expected = tuner.simulate({parameter: value})
    result = tuner.optimize(1, n_jobs=2, show_progress_bar=False)
    assert result.study.trials[0].user_attrs["worker_pid"] != os.getpid()
    assert result.best_value == expected.metrics["cpi"]
    np.testing.assert_array_equal(result.best_run.output, expected.output)


def test_hard_constraints_preserve_rejected_metrics_and_diagnostics():
    tuner = ControllerTuner(**settings(constraints={"cpi": 0.0}))
    with pytest.raises(RuntimeError, match="No feasible trial completed"):
        tuner.optimize(3, n_jobs=2, show_progress_bar=False)
    trials = tuner.optimizer.study.trials
    assert len(trials) == 3
    assert all(t.state == TrialState.PRUNED for t in trials)
    assert all(t.user_attrs["evaluations"][0]["metrics"]["cpi"] > 0 for t in trials)
    assert all(t.intermediate_values for t in trials)
    assert tuner.diagnostics()["constraints"]["cpi"]["violating_cases"] == 3


def test_target_finishes_current_batch_and_stops_before_next():
    tuner = ControllerTuner(**settings())
    result = tuner.optimize(9, n_jobs=2, target=1e6, show_progress_bar=False)
    assert len(result.study.trials) == 2
    assert all(t.state == TrialState.COMPLETE for t in result.study.trials)


def test_warm_start_finishes_first_before_other_candidates():
    tuner = ControllerTuner(**settings())
    result = tuner.optimize(
        9,
        n_jobs=2,
        target=1e6,
        initial_params={"actor_lr": 0.01},
        show_progress_bar=False,
    )
    assert len(result.study.trials) == 1
    assert result.study.trials[0].user_attrs["initial_params"]
    assert result.best_params["actor_lr"] == 0.01


def test_patience_finishes_current_batch_with_no_running_trials():
    tuner = ControllerTuner(**settings(search_space={"actor_lr": Float(0.01, 0.01)}))
    result = tuner.optimize(9, n_jobs=2, patience=1, show_progress_bar=False)
    assert len(result.study.trials) == 2
    assert all(t.state == TrialState.COMPLETE for t in result.study.trials)


def test_timeout_is_checked_between_complete_batches(monkeypatch):
    clock = iter([0.0, 0.0, 20.0])
    monkeypatch.setattr(
        _parallel, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    tuner = ControllerTuner(**settings())
    result = tuner.optimize(9, n_jobs=2, timeout=10.0, show_progress_bar=False)
    assert len(result.study.trials) == 2


def test_worker_programming_errors_propagate_and_leave_no_running_trials(monkeypatch):
    # Importable callable with the wrong signature fails inside the actual worker.
    monkeypatch.setattr(_parallel, "_evaluate_candidate", os.getpid)
    tuner = ControllerTuner(**settings())
    with pytest.raises(TypeError):
        tuner.optimize(3, n_jobs=2, show_progress_bar=False)
    assert all(t.state == TrialState.FAIL for t in tuner.optimizer.study.trials)
    assert len(tuner.optimizer.study.trials) == 2


@pytest.mark.parametrize("n_jobs", [0, -1, True, 1.5])
def test_bad_worker_count_fails_before_search(n_jobs):
    tuner = ControllerTuner(**settings())
    with pytest.raises(ValueError, match="n_jobs"):
        tuner.optimize(2, n_jobs=n_jobs)
    assert tuner.optimizer is None


def test_annealing_keeps_serial_metropolis_semantics():
    tuner = ControllerTuner(**settings(method="annealing"))
    with pytest.raises(ValueError, match="Annealing requires n_jobs=1"):
        tuner.optimize(2, n_jobs=2)
    assert tuner.optimizer is None


def test_unpicklable_settings_explain_serial_fallback():
    tuner = ControllerTuner(**settings())
    tuner.environment.kwargs["reward_func"] = lambda *args: 0.0
    with pytest.raises(ValueError, match="picklable.*n_jobs=1"):
        tuner.optimize(2, n_jobs=2)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_torch_thread_count(threads):
    with pytest.raises(ValueError, match="torch_threads"):
        ControllerTuner(**settings(torch_threads=threads))


def test_torch_setting_restored_even_after_failure(monkeypatch):
    import tensoraerospace.optimization._controllers as controllers

    previous = torch.get_num_threads()
    tuner = ControllerTuner(**settings(torch_threads=2 if previous == 1 else 1))

    def fail(*args):
        assert torch.get_num_threads() == tuner.torch_threads
        raise RuntimeError("deliberate controller failure")

    monkeypatch.setattr(controllers, "build_controller", fail)
    with pytest.raises(RuntimeError, match="deliberate controller failure"):
        tuner.simulate()
    assert torch.get_num_threads() == previous


def test_parallel_progress_counts_processed_trials_and_best(monkeypatch):
    import io

    from tqdm.std import tqdm

    from tensoraerospace.optimization import _progress

    stream = io.StringIO()
    bars = []

    def progress(**kwargs):
        bar = tqdm(**kwargs, file=stream, mininterval=0)
        bars.append(bar)
        return bar

    monkeypatch.setattr(_progress, "tqdm", progress)
    result = ControllerTuner(**settings()).optimize(9, n_jobs=2, target=1e6)
    assert bars[0].n == 2
    assert bars[0].disable
    assert f"best cpi={result.best_value:.6g}" in stream.getvalue()
    assert "target reached" in stream.getvalue()
