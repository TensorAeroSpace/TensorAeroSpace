"""Search correctness: actual Optuna studies, independent candidates and failures."""

import json
import random
from dataclasses import dataclass, field

import numpy as np
import optuna
import pytest
import torch

from tensoraerospace.benchmark import ControlBenchmark
from tensoraerospace.optimization import (
    Categorical,
    ControlOptimizer,
    Float,
    Int,
    StepResponseMetric,
    TrialRejected,
)
from tensoraerospace.optimization.agent import OptimizableAgent, _replace_paths


def test_tpe_improves_known_seeded_objective_and_warm_start():
    def objective(p, seed):
        return {"cpi": (p["x"] - 0.37) ** 2 + seed * 0.001}

    optimizer = ControlOptimizer({"x": Float(0, 1)}, objective, seeds=[0, 1], seed=17)
    result = optimizer.optimize(30, initial_params={"x": 0.95})
    assert result.best_value < 0.003
    assert result.best_value < optimizer.study.trials[0].value
    assert result.study.trials[0].user_attrs["initial_params"]
    assert len(result.study.best_trial.user_attrs["evaluations"]) == 2


def test_constraints_checked_per_case_and_no_missing_settling_as_zero():
    def objective(p, seed):
        return {"cpi": p["x"], "settling": None if p["x"] == 0 else 2 * seed}

    optimizer = ControlOptimizer(
        {"x": Int(0, 2)}, objective, seeds=[0, 1], constraints={"settling": 1}
    )
    with pytest.raises(RuntimeError, match="No feasible"):
        optimizer.optimize(3, initial_params={"x": 0})
    assert all(
        t.state == optuna.trial.TrialState.PRUNED for t in optimizer.study.trials
    )
    assert (
        "Constraint settling"
        in optimizer.study.trials[0].user_attrs["rejection_reason"]
    )


def test_complete_scenarios_target_patience_and_validation_is_not_training(tmp_path):
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        {"positive": lambda p, s: 2.0, "negative": lambda p, s: 4.0},
        seeds=[0, 1],
        aggregation="worst",
    )
    result = optimizer.optimize(20, target=4)
    assert len(result.study.trials) == 1
    assert result.best_value == 4
    assert len(result.study.best_trial.user_attrs["evaluations"]) == 4
    validation = optimizer.validate(seeds=[9, 10])
    assert validation["feasible"] and validation["value"] == 4
    assert len(result.study.trials) == 1
    result.save(tmp_path / "result.json")
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["best_value"] == 4
    result.plot_history()
    optimizer.optimize(10, patience=2)
    assert len(result.study.trials) == 3


def test_pruning_skips_expensive_remaining_cases():
    seen = []
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: seen.append(s) or 2.0,
        seeds=[0, 1, 2],
        pruner=optuna.pruners.ThresholdPruner(upper=1),
    )
    with pytest.raises(RuntimeError, match="No feasible"):
        optimizer.optimize(1)
    assert seen == [0]
    seen.clear()
    result = optimizer.optimize(1, initial_params={"x": 0.5})
    assert result.best_value == 2 and seen == [0, 1, 2]


@pytest.mark.parametrize(
    "failure", [float("nan"), float("inf"), None, TrialRejected("envelope")]
)
def test_bad_trajectories_never_win(failure):
    def objective(p, seed):
        if isinstance(failure, Exception):
            raise failure
        return {"cpi": failure}

    opt = ControlOptimizer({"x": Float(0, 1)}, objective, seeds=[0])
    with pytest.raises(RuntimeError, match="No feasible"):
        opt.optimize(1)
    assert not opt.validate(seeds=[1], params={"x": 0.5})["feasible"]


def test_programming_error_not_hidden_as_bad_parameters():
    opt = ControlOptimizer({"x": Float(0, 1)}, lambda p, s: {"cp_typo": 0.0})
    with pytest.raises(KeyError, match="cpi"):
        opt.optimize(1)
    assert opt.study.trials[0].state == optuna.trial.TrialState.FAIL


def test_storage_resume_checks_protocol(tmp_path):
    options = dict(
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        study_name="test",
        experiment_id="plant-v1",
        seeds=[0],
    )

    def objective(p, s):
        return p["x"] ** 2

    opt = ControlOptimizer({"x": Float(0, 1)}, objective, **options)
    first = opt.optimize(1, initial_params={"x": 0.25})
    second = ControlOptimizer(
        {"x": Float(0, 1)}, objective, load_if_exists=True, **options
    )
    result = second.optimize(1)
    assert len(result.study.trials) == 2
    assert result.best_value <= first.best_value
    with pytest.raises(ValueError, match="protocol differs"):
        ControlOptimizer({"x": Float(0, 2)}, objective, load_if_exists=True, **options)
    with pytest.raises(ValueError, match="experiment_id"):
        ControlOptimizer({"x": Float(0, 1)}, objective, storage=options["storage"])


@dataclass
class Config:
    gain: float = 1.0
    seed: int = 0
    array: np.ndarray = field(default_factory=lambda: np.ones((2, 2)))
    limits: tuple = (1.0, 2.0)

    def __post_init__(self):
        if self.gain < 0:
            raise ValueError("negative gain")


class Agent(OptimizableAgent):
    def __init__(self, config, settings, resource=None):
        self.cfg, self.settings, self.resource = config, settings, resource
        self.history = []


def test_nested_constructor_agent_method_and_independence():
    cfg = Config()
    settings = {"lr": 0.1}
    instances = []

    def evaluate(agent, seed):
        assert agent.cfg.seed == seed
        assert agent.settings["lr"] == 0.02
        assert agent.cfg.array[1, 0] == 3
        assert agent.cfg.limits[0] == 4
        assert not agent.history
        instances.append(agent)
        agent.history.append(seed)
        return agent.cfg.gain

    result = Agent.optimize(
        {
            "config.gain": Float(0.1, 0.2),
            "settings.lr": Categorical([0.02]),
            "config.array.1.0": Int(3, 3),
            "config.limits.0": Int(4, 4),
        },
        evaluate,
        agent_kwargs={"config": cfg, "settings": settings},
        seed_path="config.seed",
        seeds=[0, 3],
        n_trials=2,
        pruner=optuna.pruners.NopPruner(),
    )
    assert len(instances) == len({id(a) for a in instances}) == 4
    assert cfg.gain == 1 and cfg.seed == 0 and np.all(cfg.array == 1)
    assert cfg.limits == (1.0, 2.0) and settings == {"lr": 0.1}
    best = result.create_agent(seed=91)
    assert best.cfg.seed == 91 and not best.history
    assert best.cfg.gain == result.best_params["config.gain"]


def test_fresh_resource_factory_is_not_deepcopied():
    class Resource:
        def __deepcopy__(self, memo):
            raise AssertionError("A live env/session must not be cloned")

    resources = []

    def kwargs(seed):
        resource = Resource()
        resources.append(resource)
        return {"config": Config(), "settings": {}, "resource": resource}

    def evaluate(agent, seed):
        assert agent.resource is resources[-1]
        return agent.cfg.gain

    Agent.optimize(
        {"config.gain": Float(0, 1)},
        evaluate,
        agent_kwargs=kwargs,
        seeds=[0],
        n_trials=2,
    )
    assert len(resources) == 2 and resources[0] is not resources[1]


def test_invalid_paths_and_dataclass_validation():
    for updates, error in [
        ({"oops": 2}, KeyError),
        ({"config.gian": 2}, KeyError),
        ({"config.gain": -1}, ValueError),
        ({"config": Config(), "config.gain": 1}, ValueError),
        ({"config.limits.4": 1}, KeyError),
    ]:
        with pytest.raises(error):
            _replace_paths({"config": Config()}, updates)
    with pytest.raises(ValueError, match="overlap"):
        ControlOptimizer.for_agent(
            Agent,
            {"config.seed": Int(0, 9)},
            lambda a, s: 0.0,
            agent_kwargs={},
            seed_path="config.seed",
        )


def test_rng_restored_on_success_and_failure():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    seen = []

    def objective(p, seed):
        seen.append((random.random(), np.random.random(), torch.rand(1).item()))
        raise TrialRejected("bad")

    optimizer = ControlOptimizer({"x": Float(0, 1)}, objective, seeds=[7])
    with pytest.raises(RuntimeError):
        optimizer.optimize(2)
    assert seen[0] == seen[1]
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])


def test_step_metric_native_cpi_timing_and_negative_steps():
    dt = 0.01
    reference = np.r_[np.zeros(2000), np.ones(2001)]
    output = np.zeros_like(reference)
    output[2000:] = 1 - np.exp(-np.arange(2001) * dt)
    scorer = StepResponseMetric(reference, dt)
    metrics = scorer(output)
    expected = ControlBenchmark().benchmarking_step_response(reference, output, 0.0, dt)
    assert metrics["cpi"] == expected["performance_index"]
    assert 2.99 <= metrics["command_settling_time"] <= 3.01
    assert metrics["relative_tail_error"] < 1e-7
    negative = StepResponseMetric(-reference, dt, normalize=True)(-output)
    assert negative["cpi"] == pytest.approx(metrics["cpi"])
    with pytest.raises(TrialRejected, match="complete"):
        scorer(output[:-1])
    output[0] = np.nan
    with pytest.raises(TrialRejected):
        scorer(output)


def test_step_metric_does_not_hide_offsets_or_channel_failures():
    ref = np.r_[np.zeros(100), np.ones(900)]
    response = ref * 0.8
    response[:100] = 0.1
    m = StepResponseMetric(ref, 0.01)(response)
    assert m["relative_tail_error"] == pytest.approx(0.2)
    assert m["pre_step_max_error"] == 0.1
    assert m["command_settling_time"] is None
    refs = np.column_stack([ref, ref * 100])
    outputs = np.column_stack([ref, response * 100])
    multi = StepResponseMetric(refs, 0.01, normalize=True)(outputs)
    assert multi["relative_tail_error"] == pytest.approx(0.2)
    assert multi["tail_max_error"] == 20
    assert multi["command_settling_time"] is None
    assert multi["cpi"] == pytest.approx(
        (multi["channel_0.performance_index"] + multi["channel_1.performance_index"])
        / 2
    )


@pytest.mark.parametrize(
    "reference", [np.zeros(5), [0, 1, 0, 1], [0, 0, 1], [0, 1, np.nan]]
)
def test_invalid_step_protocol_rejected(reference):
    with pytest.raises(ValueError):
        StepResponseMetric(reference, 0.01)


@pytest.mark.parametrize(
    "distribution,value",
    [
        (Int(1, 5), 1.5),
        (Float(0.0, 1.0, step=0.1), 0.35),
        (Float(0.01, 1.0, log=True), 0.0),
        (Categorical(["a", "b"]), "c"),
    ],
)
def test_enqueued_parameters_must_belong_to_search_grid(distribution, value):
    optimizer = ControlOptimizer({"x": distribution}, lambda p, s: 0.0)
    with pytest.raises(ValueError, match="outside"):
        optimizer.optimize(1, initial_params={"x": value})
    assert not optimizer.study.trials


def test_float_tuning_of_integer_array_never_truncates_candidate():
    base = {"weights": np.array([[1, 2], [3, 4]], dtype=int)}
    changed = _replace_paths(base, {"weights.0.1": 0.25})
    assert changed["weights"][0, 1] == 0.25
    np.testing.assert_array_equal(base["weights"], [[1, 2], [3, 4]])


def test_result_validation_uses_best_snapshot_and_does_not_mutate_live_instance():
    live = Agent(Config(), {})
    live.history.append("trained")
    kwargs = {"config": Config(), "settings": {}}
    result = live.optimize(
        {"config.gain": Float(0.2, 0.2)},
        lambda a, s: a.cfg.gain,
        agent_kwargs=kwargs,
        seeds=[0],
        n_trials=1,
    )
    kwargs["config"].gain = 91.0
    assert live.history == ["trained"] and live.cfg.gain == 1.0
    assert result.validate(seeds=[8])["value"] == 0.2
    assert len(result.study.trials) == 1
    assert result.create_agent().cfg.gain == 0.2


@pytest.mark.parametrize(
    "path", ["config.", "config..gain", ".config", "config.limits.01"]
)
def test_paths_cannot_silently_alias_a_different_parameter(path):
    with pytest.raises((ValueError, KeyError)):
        _replace_paths({"config": Config()}, {path: 1.0})


def test_agent_mixin_import_is_lightweight():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tensoraerospace.optimization.agent import OptimizableAgent; "
            "assert 'optuna' not in sys.modules; assert 'ray' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_constructor_defaults_need_no_placeholder_and_unknown_keys_still_fail():
    class WithDefaults(OptimizableAgent):
        def __init__(self, gain=2.0, seed=42):
            self.gain, self.seed = gain, seed

    def evaluate(agent, seed):
        assert agent.seed == seed
        return agent.gain

    result = WithDefaults.optimize(
        {"gain": Float(0.3, 0.3)},
        evaluate,
        agent_kwargs={},
        seed_path="seed",
        seeds=[8],
        n_trials=1,
    )
    assert result.best_value == 0.3
    with pytest.raises(KeyError, match="gian"):
        WithDefaults.optimize(
            {"gian": Float(0.3, 0.3)}, evaluate, agent_kwargs={}, n_trials=1
        )


def test_infeasible_objective_ceiling_reports_observed_score_and_preserves_bound():
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: {"cpi": 3.4, "tail": 0.01},
        seeds=[0],
        constraints={"cpi": 0.6, "tail": 0.02},
    )
    with pytest.raises(RuntimeError) as error:
        optimizer.optimize(3)
    message = str(error.value)
    assert "cpi <= 0.6" in message
    assert "3/3 evaluated cases" in message
    assert "lowest observed 3.4" in message
    assert "target=" in message
    report = optimizer.diagnostics()
    assert report["trial_states"] == {"PRUNED": 3}
    assert report["constraints"]["cpi"]["upper_bound"] == 0.6
    assert report["constraints"]["tail"]["violating_cases"] == 0
    assert report["lowest_fully_evaluated"]["value"] == 3.4
    assert not report["lowest_fully_evaluated"]["feasible"]
    assert optimizer.constraints["cpi"] == 0.6
    json.dumps(report, allow_nan=False)


def test_diagnostics_include_all_violations_and_missing_settling():
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: {"cpi": 3.0, "tail": 0.5, "settling": None},
        seeds=[0],
        constraints={"cpi": 0.6, "tail": 0.02, "settling": 12.0},
    )
    with pytest.raises(RuntimeError) as error:
        optimizer.optimize(1)
    report = optimizer.diagnostics()
    assert all(row["violating_cases"] == 1 for row in report["constraints"].values())
    assert report["constraints"]["settling"]["lowest_observed"] is None
    assert report["constraints"]["settling"]["nonfinite_cases"] == 1
    assert "settling <= 12" in str(error.value)
    assert "no finite value recorded" in str(error.value)
    assert "tail <= 0.02" in str(error.value)


def test_diagnostics_do_not_call_a_partial_seed_score_fully_evaluated():
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: {"cpi": 0.7 if s == 0 else 100.0},
        seeds=[0, 1],
        constraints={"cpi": 0.6},
    )
    with pytest.raises(RuntimeError) as error:
        optimizer.optimize(1)
    report = optimizer.diagnostics()
    assert report["evaluated_cases"] == 1
    assert report["constraints"]["cpi"]["lowest_observed"] == 0.7
    assert report["lowest_fully_evaluated"] is None
    assert "Lowest cpi across fully evaluated trials" not in str(error.value)


def test_diagnostics_count_cases_and_match_multi_seed_aggregation():
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        {"a": lambda p, s: {"cpi": 1.0 + s}, "b": lambda p, s: {"cpi": 2.0 + s}},
        seeds=[0, 1],
        constraints={"cpi": 2.5},
        pruner=optuna.pruners.NopPruner(),
    )
    with pytest.raises(RuntimeError):
        optimizer.optimize(1)
    report = optimizer.diagnostics()
    assert report["constraints"]["cpi"]["evaluated_cases"] == 4
    assert report["constraints"]["cpi"]["violating_cases"] == 1
    assert report["lowest_fully_evaluated"]["value"] == 2.0
    assert not report["lowest_fully_evaluated"]["feasible"]


def test_diagnostics_distinguish_physical_failure_from_metric_rejection():
    def objective(params, seed):
        raise TrialRejected("Boeing left the flight envelope")

    optimizer = ControlOptimizer(
        {"x": Float(0, 1)}, objective, seeds=[0], constraints={"cpi": 0.6}
    )
    with pytest.raises(RuntimeError) as error:
        optimizer.optimize(2)
    report = optimizer.diagnostics()
    assert report["lowest_fully_evaluated"] is None
    assert report["constraints"]["cpi"]["evaluated_cases"] == 0
    assert report["rejection_reasons"]["Boeing left the flight envelope"] == 2
    assert "2 trial(s): Boeing left the flight envelope" in str(error.value)
    assert "lowest observed" not in str(error.value)


def test_unmet_stop_target_still_returns_best_feasible_result():
    optimizer = ControlOptimizer(
        {"x": Float(0, 1)},
        lambda p, s: {"cpi": 3.4},
        seeds=[0],
        constraints={"cpi": 3.5},
    )
    result = optimizer.optimize(3, target=0.6)
    assert len(result.study.trials) == 3
    assert result.best_value == 3.4
    assert optimizer.diagnostics()["lowest_fully_evaluated"]["feasible"]
