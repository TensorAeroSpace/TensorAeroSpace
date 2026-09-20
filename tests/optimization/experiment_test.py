"""Executable contracts for the callback-free physical step experiment API."""

from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from tensoraerospace.optimization import (
    AnnealingSampler,
    Categorical,
    ControllerTuner,
    Float,
    Int,
    Step,
    StepResponseMetric,
)
from tensoraerospace.optimization.samplers import make_sampler


def linear_settings(**updates):
    settings = dict(
        env="LinearLongitudinalF16-v0",
        env_kwargs=dict(initial_state=[0.0, 0.0], state_space=["alpha", "q"], dt=0.02),
        reference={"alpha": Step(0.1, at=0.1, unit="deg")},
        duration=0.3,
        controller="iadp",
    )
    settings.update(updates)
    return settings


def boeing_settings(**updates):
    settings = dict(
        env="NonlinearB737-v0",
        env_kwargs={"dt": 0.02, "trim_at": (10000.0, 600.0)},
        reference={"theta": Step(0.1, at=0.1, unit="deg")},
        controller="aa_indi",
        duration=0.3,
    )
    settings.update(updates)
    return settings


@pytest.mark.parametrize(
    "controller", ["iadp", "imgdhp", "et_dhp", "ihdp", "mpc", "aa_indi", "aidi", "hdp"]
)
def test_native_controller_full_online_episode(controller):
    if controller in ("aa_indi", "aidi"):
        settings = boeing_settings(controller=controller)
    elif controller == "hdp":
        settings = linear_settings(
            controller=controller,
            env="ImprovedB747-v0",
            env_kwargs={"initial_state": [0.0, 0.0, 0.0, 0.0], "dt": 0.02},
            reference={"theta": Step(0.1, at=0.1, unit="deg")},
        )
    else:
        settings = linear_settings(controller=controller)
    settings["controller_options"] = (
        {"model_epochs": 2, "model_samples": 16}
        if controller == "et_dhp"
        else {"iters": 2} if controller == "mpc" else {}
    )
    tuner = ControllerTuner(**settings)
    result = tuner.simulate(seed=13)
    assert result.output.shape == (16, 1)
    assert result.actions.shape[0] == 15
    assert result.time[-1] == pytest.approx(0.3)
    assert np.isfinite(result.output).all()
    assert result.metrics["cpi"] == pytest.approx(
        StepResponseMetric(result.reference, 0.02, normalize=True)(result.output)["cpi"]
    )
    assert result.metrics[f"{result.states[0]}.cpi"] == result.metrics["cpi"]


def test_multiple_named_steps_have_independent_timing_units_and_scores():
    tuner = ControllerTuner(
        **boeing_settings(
            reference={
                "theta": Step(0.1, at=0.1, unit="deg"),
                "phi": Step(-0.002, at=0.2, unit="rad"),
            }
        )
    )
    result = tuner.simulate()
    assert result.states == ("theta", "phi")
    assert result.reference[4, 0] == result.reference[0, 0]
    assert result.reference[5, 0] - result.reference[0, 0] == pytest.approx(
        np.deg2rad(0.1)
    )
    assert result.reference[9, 1] == result.reference[0, 1]
    assert result.reference[10, 1] - result.reference[0, 1] == pytest.approx(-0.002)
    assert result.metrics["cpi"] == pytest.approx(
        (result.metrics["theta.cpi"] + result.metrics["phi.cpi"]) / 2
    )
    np.testing.assert_allclose(result.reference[0], result.output[0])


@pytest.mark.parametrize(
    "method,sampler_type",
    [
        ("annealing", AnnealingSampler),
        ("genetic", optuna.samplers.NSGAIISampler),
        ("tpe", optuna.samplers.TPESampler),
        ("random", optuna.samplers.RandomSampler),
    ],
)
def test_select_search_method_and_automatically_replay(method, sampler_type):
    tuner = ControllerTuner(
        **boeing_settings(
            method=method,
            method_options={"population_size": 2} if method == "genetic" else {},
        )
    )
    result = tuner.optimize(
        n_trials=4, initial_params={"rate_gain": 3.0, "cutoff_hz": 5.0}
    )
    assert isinstance(result.study.sampler, sampler_type)
    assert len(result.study.trials) == 4
    assert result.best_value == pytest.approx(min(t.value for t in result.study.trials))
    assert result.best_run.metrics["cpi"] == pytest.approx(result.best_value)
    np.testing.assert_array_equal(result.simulate().output, result.best_run.output)


def test_training_reuses_agent_but_resets_plant_and_keeps_adaptation(monkeypatch):
    import tensoraerospace.optimization._controllers as controllers

    records = []
    native = controllers.build_controller

    def record(*args, **kwargs):
        runner = native(*args, **kwargs)
        original = runner.learn

        def learn(*args):
            original(*args)
            records.append(runner.agent.identifier.derivatives.copy())

        runner.learn = learn
        return runner

    monkeypatch.setattr(controllers, "build_controller", record)
    tuner = ControllerTuner(**boeing_settings(training_episodes=1))
    result = tuner.simulate()
    assert len(records) == 30
    assert result.output.shape == (16, 1)
    assert np.linalg.norm(records[-1] - records[0]) > 0


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"reference": {"typo": Step(1.0, at=0.1)}}, "Unknown state"),
        ({"reference": {"alpha": Step(1.0, at=0.105)}}, "align"),
        ({"controller": "aa_indi"}, "requires nonlinear"),
        ({"controller": "hdp"}, "requires Improved"),
        ({"controller": "ihdp", "training_episodes": 1}, "one online episode"),
        ({"controller_options": {"typo": 1}}, "Unknown"),
        ({"method": "typo"}, "method must"),
        ({"search_space": {"typo": Float(0.1, 1.0)}}, "Unknown tuning"),
    ],
)
def test_invalid_configuration_fails_before_search(updates, message):
    with pytest.raises(ValueError, match=message):
        ControllerTuner(**linear_settings(**updates))


def test_full_episode_and_unstable_candidates_do_not_win():
    tuner = ControllerTuner(**boeing_settings(constraints={"theta.cpi": 0.0}))
    with pytest.raises(RuntimeError, match="feasible"):
        tuner.optimize(n_trials=2)
    assert all(
        t.state == optuna.trial.TrialState.PRUNED for t in tuner.optimizer.study.trials
    )


def test_linear_f16_dt_changes_physics_and_survives_reset():
    slow = ControllerTuner(**linear_settings())
    fast = ControllerTuner(
        **linear_settings(
            env_kwargs=dict(
                initial_state=[0.0, 0.0], state_space=["alpha", "q"], dt=0.01
            )
        )
    )
    with slow.environment.open(0) as a, fast.environment.open(0) as b:
        np.testing.assert_allclose(
            a.model.filt_A, b.model.filt_A @ b.model.filt_A, atol=1e-12
        )
        a.reset(0)
        assert a.model.discretisation_time == 0.02


@pytest.mark.parametrize(
    "env,initial,names",
    [
        ("NonlinearLongitudinalF16-v0", [0.0, 0.0, 0.0, 0.0], ["alpha", "wz"]),
        ("NonlinearAngularF16-v0", [0.0] * 14, None),
    ],
)
def test_nonlinear_f16_nominal_prior_preserves_live_model(env, initial, names):
    kwargs = {"initial_state": initial, "dt": 0.01, "integrator": "rk4"}
    if names:
        kwargs["state_space"] = names
    tuner = ControllerTuner(
        **linear_settings(
            env=env,
            env_kwargs=kwargs,
            controller="mpc",
            controller_options={"iters": 2},
        )
    )
    with tuner.environment.open(0) as physical:
        before = physical.model.current_state.copy()
        counter = physical.model.time_step
        A, B = physical.nominal_discrete(list(range(len(physical.bias))))
        np.testing.assert_array_equal(physical.model.current_state, before)
        assert counter == physical.model.time_step
        assert A.shape == (len(physical.names),) * 2
        assert B.shape == (len(physical.names), len(physical.bias))
        assert np.isfinite(A).all() and np.isfinite(B).all()


def test_boeing_nominal_prior_does_not_read_fault_effectiveness():
    from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness

    healthy = ControllerTuner(**boeing_settings())
    failed = ControllerTuner(
        **boeing_settings(
            env_kwargs={
                "trim_at": (10000.0, 600.0),
                "dt": 0.02,
                "elevator_fault": ElevatorEffectiveness(time=0.0, effectiveness=0.8),
            }
        )
    )
    with healthy.environment.open(0) as a, failed.environment.open(0) as b:
        for expected, actual in zip(a.nominal_discrete([0]), b.nominal_discrete([0])):
            np.testing.assert_array_equal(expected, actual)


def test_annealing_acceptance_rejection_and_cooling():
    sampler = AnnealingSampler({"x": Float(0.0, 1.0)}, temperature=1e-4, cooling=0.5)
    trial = SimpleNamespace(params={"x": 0.2})
    sampler.after_trial(None, trial, optuna.trial.TrialState.COMPLETE, [1.0])
    assert sampler.current == {"x": 0.2}
    sampler.after_trial(
        None,
        SimpleNamespace(params={"x": 0.9}),
        optuna.trial.TrialState.COMPLETE,
        [100.0],
    )
    assert sampler.current == {"x": 0.2}
    sampler.after_trial(
        None, SimpleNamespace(params={"x": 0.1}), optuna.trial.TrialState.PRUNED, [0.0]
    )
    assert sampler.current == {"x": 0.2}
    sampler.after_trial(
        None,
        SimpleNamespace(params={"x": 0.3}),
        optuna.trial.TrialState.COMPLETE,
        [0.5],
    )
    assert sampler.current == {"x": 0.3}
    assert sampler.temperature == pytest.approx(1e-4 * 0.5**4)


def test_annealing_mixed_space_is_bounded_reproducible_and_mutates():
    space = {
        "gain": Float(1e-5, 1e-1, log=True),
        "window": Int(2, 12, step=2),
        "kind": Categorical(["a", "b"]),
    }

    def search():
        study = optuna.create_study(sampler=make_sampler("annealing", space, seed=7))

        def objective(trial):
            gain = trial.suggest_float("gain", 1e-5, 1e-1, log=True)
            window = trial.suggest_int("window", 2, 12, step=2)
            trial.suggest_categorical("kind", ["a", "b"])
            return gain + window / 100

        study.optimize(objective, n_trials=20)
        return [t.params for t in study.trials]

    a = search()
    assert a == search()
    assert len({p["gain"] for p in a}) > 1
    for p in a:
        assert 1e-5 <= p["gain"] <= 1e-1
        assert p["window"] in range(2, 13, 2)
        assert p["kind"] in ("a", "b")


def test_unknown_metric_is_rejected_before_any_trial():
    with pytest.raises(ValueError, match="Unknown metric"):
        ControllerTuner(**boeing_settings(metric="cp1"))


def test_named_rate_step_converts_degrees_per_second():
    tuner = ControllerTuner(
        **boeing_settings(reference={"q": Step(1.0, at=0.1, unit="deg")})
    )
    result = tuner.simulate()
    assert result.units == ("rad/s",)
    assert result.reference[-1, 0] - result.reference[0, 0] == pytest.approx(
        np.deg2rad(1.0)
    )


def test_no_control_effect_prior_has_actionable_preflight_error():
    with pytest.raises(ValueError, match="include actuator states"):
        ControllerTuner(
            **linear_settings(
                env="NonlinearLongitudinalF16-v0",
                env_kwargs={
                    "initial_state": [0.0, 0.0],
                    "state_space": ["alpha", "wz"],
                    "dt": 0.01,
                },
            )
        )


def test_failed_rollout_closes_environment(monkeypatch):
    from tensoraerospace.envs.b737_nonlinear import NonlinearB737Env
    from tensoraerospace.optimization import TrialRejected

    tuner = ControllerTuner(**boeing_settings())
    closed = []
    monkeypatch.setattr(NonlinearB737Env, "close", lambda self: closed.append(True))

    def fail(*args):
        raise TrialRejected("forced numerical failure")

    monkeypatch.setattr(tuner, "_check_state", fail)
    with pytest.raises(TrialRejected):
        tuner.simulate()
    assert closed == [True]


def test_last_ihdp_update_failure_is_detected(monkeypatch):
    import torch

    import tensoraerospace.optimization._controllers as controllers
    from tensoraerospace.optimization import TrialRejected

    native = controllers.build_controller

    def sabotage(*args, **kwargs):
        runner = native(*args, **kwargs)
        predict = runner.predict

        def poisoned(physical, reference, k):
            result = predict(physical, reference, k)
            if k == 14:
                with torch.no_grad():
                    next(runner.agent.critic.model.parameters()).fill_(float("nan"))
            return result

        runner.predict = poisoned
        return runner

    monkeypatch.setattr(controllers, "build_controller", sabotage)
    tuner = ControllerTuner(**linear_settings(controller="ihdp"))
    with pytest.raises(TrialRejected, match="network weights"):
        tuner.simulate()


def test_tuner_diagnostics_are_available_after_no_feasible_result():
    tuner = ControllerTuner(**boeing_settings(constraints={"cpi": 0.6}))
    with pytest.raises(RuntimeError, match=r"Call optimize\(\)"):
        tuner.diagnostics()
    with pytest.raises(RuntimeError, match="cpi <= 0.6"):
        tuner.optimize(1, initial_params={"rate_gain": 3.0, "cutoff_hz": 5.0})
    report = tuner.diagnostics()
    assert report["constraints"]["cpi"]["violating_cases"] == 1
    assert report["lowest_fully_evaluated"]["value"] > 0.6
    assert not report["lowest_fully_evaluated"]["feasible"]


def test_tuner_forwards_progress_option(monkeypatch):
    from tensoraerospace.optimization.control import ControlOptimizer

    observed = []
    native = ControlOptimizer.optimize

    def record(self, *args, **kwargs):
        observed.append(kwargs["show_progress_bar"])
        return native(self, *args, **kwargs)

    monkeypatch.setattr(ControlOptimizer, "optimize", record)
    tuner = ControllerTuner(**boeing_settings())
    tuner.optimize(1, show_progress_bar=False)
    assert observed == [False]
