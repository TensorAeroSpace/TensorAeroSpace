"""Continued searches retain sampler state and comparable physical experiments."""

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
from optuna.trial import TrialState

from tensoraerospace.optimization import ControllerTuner, Float, Step
from tensoraerospace.optimization._continuation import same_setting, write_checkpoint


def settings(**updates):
    value = dict(
        env="NonlinearB737-v0",
        env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
        reference={"theta": Step(0.1, at=0.1, unit="deg")},
        controller="ihdp",
        duration=0.3,
        seed=42,
    )
    value.update(updates)
    return value


def signature(study):
    return [
        (
            trial.number,
            trial.params,
            trial.value,
            trial.state,
            trial.user_attrs.get("evaluations"),
            trial.user_attrs.get("rejection_reason"),
            trial.intermediate_values,
        )
        for trial in study.trials
    ]


@pytest.mark.parametrize(
    "method,n_jobs", [("tpe", 2), ("genetic", 2), ("annealing", 1), ("random", 1)]
)
def test_resume_and_checkpoint_match_uninterrupted_search(method, n_jobs, tmp_path):
    options = {"population_size": 2} if method == "genetic" else {}
    config = settings(method=method, method_options=options)
    tuner = ControllerTuner(**config)
    before = tuner.optimize(4, n_jobs=n_jobs, show_progress_bar=False)
    prior_trials = signature(before.study)
    study, sampler = before.study, before.study.sampler
    temperature = getattr(sampler, "temperature", None)
    checkpoint = tmp_path / "search.pkl"
    before.save_checkpoint(checkpoint)

    continued = before.resume(4, show_progress_bar=False)
    assert continued.study is study
    assert continued.study.sampler is sampler
    assert signature(continued.study)[:4] == prior_trials
    assert continued.best_value <= before.best_value
    if temperature is not None:
        assert sampler.temperature == pytest.approx(temperature * sampler.cooling**4)

    restored = ControllerTuner.load_checkpoint(checkpoint)
    assert restored._search_n_jobs == n_jobs
    assert signature(restored.optimizer.study) == prior_trials
    replay = restored.resume(4, show_progress_bar=False)
    continuous = ControllerTuner(**config).optimize(
        8, n_jobs=n_jobs, show_progress_bar=False
    )
    assert (
        signature(continued.study)
        == signature(replay.study)
        == signature(continuous.study)
    )
    np.testing.assert_array_equal(continued.best_run.output, replay.best_run.output)
    np.testing.assert_array_equal(continued.best_run.actions, replay.best_run.actions)


def test_resume_can_change_worker_count_without_replacing_study_or_sampler():
    tuner = ControllerTuner(**settings(method="random"))
    initial = tuner.optimize(1, show_progress_bar=False)
    study, sampler = initial.study, initial.study.sampler
    parallel = tuner.resume(2, n_jobs=2, show_progress_bar=False)
    serial = tuner.resume(1, n_jobs=1, show_progress_bar=False)
    assert serial.study is parallel.study is study
    assert serial.study.sampler is sampler
    assert len(study.trials) == 4
    assert all("worker_pid" in trial.user_attrs for trial in study.trials[1:3])
    assert "worker_pid" not in study.trials[-1].user_attrs
    reference = ControllerTuner(**settings(method="random")).optimize(
        4, show_progress_bar=False
    )
    assert signature(study) == signature(reference.study)


def test_infeasible_search_can_be_saved_and_continued_without_relaxing_constraints(
    tmp_path,
):
    tuner = ControllerTuner(**settings(method="random", constraints={"cpi": 0.0}))
    with pytest.raises(RuntimeError, match="No feasible"):
        tuner.optimize(2, show_progress_bar=False)
    original = signature(tuner.optimizer.study)
    tuner.save_checkpoint(tmp_path / "infeasible.pkl")
    restored = ControllerTuner.load_checkpoint(tmp_path / "infeasible.pkl")
    with pytest.raises(RuntimeError, match="No feasible"):
        restored.resume(3, show_progress_bar=False)
    assert len(restored.optimizer.study.trials) == 5
    assert signature(restored.optimizer.study)[:2] == original
    assert all(t.state == TrialState.PRUNED for t in restored.optimizer.study.trials)
    assert restored.constraints == {"cpi": 0.0}
    assert restored.diagnostics()["constraints"]["cpi"]["violating_cases"] == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference", {"theta": Step(0.2, at=0.1, unit="deg")}),
        ("controller", "imgdhp"),
        ("controller_options", {"actor_lr": 0.002}),
        ("search_space", {"actor_lr": Float(0.001, 0.002)}),
        ("metric", "ise"),
        ("normalize", False),
        ("constraints", {"cpi": 100.0}),
        ("state_limits", {"theta": (-0.2, 0.2)}),
        ("duration", 0.4),
        ("training_episodes", 1),
        ("seeds", (7,)),
        ("seed", 10),
        ("torch_threads", None),
        ("method", "annealing"),
        ("method_options", {"n_startup_trials": 9}),
    ],
)
def test_changed_experiment_is_rejected_before_adding_trials(field, value, tmp_path):
    tuner = ControllerTuner(**settings())
    result = tuner.optimize(1, show_progress_bar=False)
    setattr(tuner, field, value)
    with pytest.raises(ValueError, match=f"settings changed:.*{field}"):
        tuner.resume(1, show_progress_bar=False)
    with pytest.raises(ValueError, match="settings changed"):
        tuner.save_checkpoint(tmp_path / "changed.pkl")
    assert len(result.study.trials) == 1
    assert not (tmp_path / "changed.pkl").exists()


def test_inplace_environment_and_time_changes_are_detected():
    tuner = ControllerTuner(**settings())
    tuner.optimize(1, show_progress_bar=False)
    tuner.environment.kwargs["dt"] = 0.01
    with pytest.raises(ValueError, match="settings changed: environment"):
        tuner.resume(1)
    tuner.environment.kwargs["dt"] = 0.02
    tuner.time[1] += 0.001
    with pytest.raises(ValueError, match="settings changed: time"):
        tuner.resume(1)


def test_named_channel_order_is_part_of_the_experiment():
    tuner = ControllerTuner(
        **settings(
            controller="aa_indi",
            reference={
                "theta": Step(0.1, at=0.1, unit="deg"),
                "phi": Step(0.1, at=0.1, unit="deg"),
            },
        )
    )
    tuner.optimize(1, show_progress_bar=False)
    tuner.reference = dict(reversed(list(tuner.reference.items())))
    with pytest.raises(ValueError, match="settings changed: reference"):
        tuner.resume(1)


def test_optimize_starts_new_study_and_old_result_cannot_resume_it(tmp_path):
    tuner = ControllerTuner(**settings(method="random"))
    old = tuner.optimize(1, show_progress_bar=False)
    current = tuner.optimize(1, show_progress_bar=False)
    assert old.study is not current.study
    assert len(old.study.trials) == len(current.study.trials) == 1
    with pytest.raises(RuntimeError, match="older study"):
        old.resume(1)
    with pytest.raises(RuntimeError, match="older study"):
        old.save_checkpoint(tmp_path / "wrong.pkl")


def test_resume_requires_existing_study_and_no_running_candidates(tmp_path):
    tuner = ControllerTuner(**settings())
    with pytest.raises(RuntimeError, match="Call optimize"):
        tuner.resume(1)
    with pytest.raises(RuntimeError, match="Call optimize"):
        tuner.save_checkpoint(tmp_path / "missing.pkl")
    result = tuner.optimize(1, show_progress_bar=False)
    pending = result.study.ask()
    with pytest.raises(RuntimeError, match="active search"):
        tuner.resume(1)
    with pytest.raises(RuntimeError, match="active search"):
        tuner.save_checkpoint(tmp_path / "running.pkl")
    result.study.tell(pending, state=TrialState.FAIL)


def test_reached_target_does_not_spend_another_trial_budget():
    tuner = ControllerTuner(**settings())
    first = tuner.optimize(1, show_progress_bar=False)
    after = tuner.resume(5, target=first.best_value, show_progress_bar=False)
    assert after.best_params == first.best_params
    assert len(after.study.trials) == 1


def test_unknown_checkpoint_format_and_failed_save_preserve_previous_file(tmp_path):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps({"format": "another format"}))
    with pytest.raises(ValueError, match="supported ControllerTuner checkpoint"):
        ControllerTuner.load_checkpoint(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="picklable"):
        write_checkpoint(path, {"callback": lambda: 0})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_nested_arrays_and_dataclasses_compare_without_ambiguous_truth_value():
    @dataclass
    class Config:
        vector: np.ndarray

    first = {"model": Config(np.array([1.0, 2.0])), "labels": np.array(["q", "theta"])}
    second = {"model": Config(np.array([1.0, 2.0])), "labels": np.array(["q", "theta"])}
    assert same_setting(first, second)
    second["model"].vector[1] = 3.0
    assert not same_setting(first, second)
    assert not same_setting(np.ones(2), np.ones((2, 1)))
    assert not same_setting([1, 2], (1, 2))
