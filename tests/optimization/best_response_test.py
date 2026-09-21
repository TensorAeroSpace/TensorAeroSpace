"""Best-response search retains evidence, budgets and hard physical constraints."""

import numpy as np
import pytest
from optuna.trial import TrialState

from tensoraerospace.optimization import ControllerTuner, Step


def settings(**updates):
    value = dict(
        env="NonlinearB737-v0",
        env_kwargs={"dt": 0.02, "trim_at": (10000.0, 600.0)},
        reference={"theta": Step(0.1, at=0.1, unit="deg")},
        controller="ihdp",
        duration=0.3,
        method="random",
        seed=42,
    )
    value.update(updates)
    return value


def signature(study):
    return [
        (t.params, t.value, t.state, t.user_attrs.get("evaluations"))
        for t in study.trials
    ]


def test_start_then_repeat_uses_additional_budget_without_discarding_trials():
    tuner = ControllerTuner(**settings())
    first = tuner.find_best_response(2, show_progress_bar=False)
    history = signature(first.study)
    sampler = first.study.sampler
    second = tuner.find_best_response(3, show_progress_bar=False)
    assert first.study is second.study
    assert sampler is second.study.sampler
    assert len(second.study.trials) == 5
    assert signature(second.study)[:2] == history
    assert second.best_value <= first.best_value
    whole = ControllerTuner(**settings()).optimize(5, show_progress_bar=False)
    assert signature(second.study) == signature(whole.study)
    np.testing.assert_array_equal(second.best_run.output, whole.best_run.output)


def test_prior_target_does_not_prevent_further_best_response_search():
    tuner = ControllerTuner(**settings())
    first = tuner.optimize(20, target=1e6, show_progress_bar=False)
    assert len(first.study.trials) == 1
    best = tuner.find_best_response(3, show_progress_bar=False)
    assert len(best.study.trials) == 4
    assert best.best_value <= first.best_value


def test_best_response_keeps_custom_objective_parallelism_and_checkpoint(tmp_path):
    tuner = ControllerTuner(**settings(metric="iae"))
    first = tuner.find_best_response(2, n_jobs=2, show_progress_bar=False)
    assert first.best_value == first.best_run.metrics["iae"]
    first.save_checkpoint(tmp_path / "best.pkl")
    restored = ControllerTuner.load_checkpoint(tmp_path / "best.pkl")
    second = restored.find_best_response(2, show_progress_bar=False)
    assert len(second.study.trials) == 4
    assert restored._search_n_jobs == 2
    assert second.best_value <= first.best_value
    assert all("worker_pid" in t.user_attrs for t in second.study.trials)


def test_best_response_does_not_relax_hard_constraints_after_all_trials_fail():
    tuner = ControllerTuner(**settings(constraints={"cpi": 0.0}))
    for budget in (2, 3):
        with pytest.raises(RuntimeError, match="No feasible"):
            tuner.find_best_response(budget, show_progress_bar=False)
    assert len(tuner.optimizer.study.trials) == 5
    assert tuner.constraints == {"cpi": 0.0}
    assert all(t.state == TrialState.PRUNED for t in tuner.optimizer.study.trials)
    assert tuner.diagnostics()["constraints"]["cpi"]["violating_cases"] == 5


def test_changed_reference_is_not_silently_restarted_or_mixed_with_old_results():
    tuner = ControllerTuner(**settings())
    first = tuner.find_best_response(1, show_progress_bar=False)
    tuner.reference = {"theta": Step(0.2, at=0.1, unit="deg")}
    with pytest.raises(ValueError, match="settings changed: reference"):
        tuner.find_best_response(2, show_progress_bar=False)
    assert len(first.study.trials) == 1
