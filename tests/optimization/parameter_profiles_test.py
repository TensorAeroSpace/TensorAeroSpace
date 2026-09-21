"""Native profile parameters must affect the actual controller, not just Optuna."""

import numpy as np
import pytest

from tensoraerospace.optimization import (
    Categorical,
    ControllerTuner,
    ControlOptimizer,
    Float,
    Int,
    Step,
)
from tensoraerospace.optimization._controllers import build_controller
from tensoraerospace.optimization._parameters import native_parameters


def settings(controller, **updates):
    value = dict(
        env="NonlinearB737-v0",
        env_kwargs={"dt": 0.02, "trim_at": (10000.0, 600.0)},
        reference={"theta": Step(0.1, at=0.1, unit="deg")},
        duration=0.3,
        controller=controller,
    )
    value.update(updates)
    return value


def build(tuner, params=None):
    with tuner.environment.open(0) as physical:
        return build_controller(
            tuner, physical, tuner._reference_for(physical), params or {}, 0
        ).agent


@pytest.mark.parametrize(
    "controller,options,path,expected",
    [
        (
            "aa_indi",
            {"config.sigma0": 5.0, "config.observer.hosm_gains.0": 8.0},
            "cfg.sigma0",
            5.0,
        ),
        ("aidi", {"config.rls_memory_length": 17}, "cfg.rls_memory_length", 17),
        (
            "iadp",
            {"config.policy_eval_iterations": 3, "config.Q.0.0": 4.0},
            "cfg.policy_eval_iterations",
            3,
        ),
        (
            "imgdhp",
            {
                "config.beta_lambda": 0.02,
                "config.actor_hidden": (7, 5),
                "config.track_Q.0": 2.0,
            },
            "cfg.beta_lambda",
            0.02,
        ),
        (
            "et_dhp",
            {
                "config.trigger_floor": 0.003,
                "config.model_hidden": (7,),
                "model_epochs": 1,
                "model_samples": 8,
            },
            "cfg.trigger_floor",
            0.003,
        ),
        (
            "mpc",
            {"weights.S_diag.0": 0.4, "iters": 2, "optimizer": "sgd"},
            "optimizer",
            "sgd",
        ),
    ],
)
def test_native_parameters_reach_real_controller(controller, options, path, expected):
    tuner = ControllerTuner(**settings(controller, controller_options=options))
    agent = build(tuner)
    value = agent
    for key in path.split("."):
        value = getattr(value, key)
    assert value == expected
    roots = {
        "config": getattr(agent, "cfg", None),
        "weights": getattr(agent, "weights", None),
    }
    for parameter in native_parameters(controller):
        parts = parameter.split(".")
        obj = roots[parts.pop(0)]
        # MPC stores converted Q/R/S, not a copy of its dataclass.
        if obj is None:
            continue
        for field in parts:
            obj = getattr(obj, field)


def test_ihdp_architecture_cost_schedule_excitation_and_identifier_window():
    tuner = ControllerTuner(
        **settings(
            "ihdp",
            controller_options={
                "actor_hidden": (12, 6),
                "critic_hidden": (9,),
                "actor_settings.activations.0": "relu",
                "actor_settings.learning_rate_min": 1e-7,
                "actor_settings.learning_rate_decay": 0.9,
                "critic_settings.learning_rate_decay": 0.8,
                "critic_settings.Q_weights.0": 4.0,
                "critic_settings.gamma": 0.87,
                "actor_settings.pulse_length_3211": 11,
                "actor_settings.amplitude_3211": 0.02,
                "incremental_settings.window_size": 27,
            },
        )
    )
    agent = build(tuner)
    assert agent.actor.layers == (12, 6, 1)
    assert agent.actor.activations == ("relu", "tanh", "tanh")
    assert agent.critic.layers == (9, 1)
    np.testing.assert_allclose(agent.critic.Q, [[4.0]])
    assert agent.critic.gamma == 0.87
    assert agent.actor.learning_rate_decay == 0.9
    assert agent.critic.learning_rate_decay == 0.8
    assert agent.actor.pulse_length_3211 == 11
    np.testing.assert_allclose(agent.actor.amplitude_3211, [[0.02]])
    assert agent.incremental_model.L == 27
    for parameter in native_parameters("ihdp"):
        root, field = parameter.split(".")
        assert field in getattr(agent, root)


def test_low_ihdp_learning_rate_never_increases_and_custom_decay_is_used():
    tuner = ControllerTuner(
        **settings(
            "ihdp",
            controller_options={
                "actor_lr": 1e-5,
                "critic_lr": 1e-8,
                "warmup_steps": 0,
                "actor_settings.learning_rate_decay": 0.5,
                "critic_settings.learning_rate_decay": 0.5,
                # Native defaults have floors higher than these chosen initial rates.
            },
        )
    )
    with tuner.environment.open(0) as physical:
        ref = tuner._reference_for(physical)
        runner = build_controller(tuner, physical, ref, {}, 0)
        for k in range(3):
            action = runner.predict(physical, ref, k)
            physical.step(action)
            runner.learn(physical, ref, k)
            assert runner.agent.actor.learning_rate <= 1e-5
            assert runner.agent.critic.learning_rate <= 1e-8
    tuner = ControllerTuner(
        **settings(
            "ihdp",
            controller_options={
                "actor_lr": 0.01,
                "critic_lr": 0.02,
                "warmup_steps": 0,
                "actor_settings.learning_rate_min": 0.0,
                "actor_settings.learning_rate_decay": 0.5,
                "critic_settings.learning_rate_min": 0.0,
                "critic_settings.learning_rate_decay": 0.25,
            },
        )
    )
    with tuner.environment.open(0) as physical:
        ref = tuner._reference_for(physical)
        runner = build_controller(tuner, physical, ref, {}, 0)
        for k in range(2):
            physical.step(runner.predict(physical, ref, k))
        assert runner.agent.actor.learning_rate == pytest.approx(0.005)
        assert runner.agent.critic.learning_rate == pytest.approx(0.005)


def test_custom_search_can_optimize_native_paths_and_fixed_options():
    tuner = ControllerTuner(
        **settings(
            "ihdp",
            search_space={
                "actor_settings.layers.0": Int(4, 8, step=4),
                "critic_settings.gamma": Float(0.8, 0.9),
                "critic_settings.Q_weights.0": Float(0.5, 2.0),
                "warmup_steps": Int(0, 2),
                "actor_settings.activations.0": Categorical(["tanh", "relu"]),
            },
        )
    )
    result = tuner.optimize(
        3, initial_params={"critic_settings.gamma": 0.85}, show_progress_bar=False
    )
    assert result.study.trials[0].params["critic_settings.gamma"] == 0.85
    assert result.best_run.metrics["cpi"] == pytest.approx(result.best_value)
    assert set(result.best_params) == set(tuner.search_space)


def test_fixed_native_value_excludes_its_alias_from_default_search():
    tuner = ControllerTuner(
        **settings(
            "ihdp",
            controller_options={"actor_settings.learning_rate": 0.001, "gamma": 0.9},
        )
    )
    assert "actor_lr" not in tuner.search_space
    assert "gamma" not in tuner.search_space
    assert "critic_lr" in tuner.search_space


@pytest.mark.parametrize(
    "space",
    [
        {
            "actor_lr": Float(0.001, 0.01),
            "actor_settings.learning_rate": Float(0.001, 0.01),
        },
        {"hidden_size": Int(4, 8), "actor_settings.layers.0": Int(4, 8)},
    ],
)
def test_duplicate_aliases_fail_instead_of_silently_ignoring_a_trial_parameter(space):
    with pytest.raises(ValueError, match="overlap"):
        ControllerTuner(**settings("ihdp", search_space=space))


@pytest.mark.parametrize(
    "key",
    [
        "config.dt",
        "actor_settings.NN_initial",
        "actor_settings.typo",
        "incremental_settings.dt",
    ],
)
def test_protocol_parameters_and_unknown_fields_cannot_be_searched(key):
    with pytest.raises(ValueError, match="Unknown tuning"):
        ControllerTuner(**settings("ihdp", search_space={key: Float(0.1, 1.0)}))


def test_partial_initial_values_are_completed_by_sampler_and_unknown_keys_fail():
    opt = ControlOptimizer(
        {"x": Float(0, 1), "y": Float(0, 1)}, lambda p, s: p["x"] + p["y"], seeds=[0]
    )
    result = opt.optimize(1, initial_params={"x": 0.3}, show_progress_bar=False)
    assert result.best_params["x"] == 0.3
    assert 0 <= result.best_params["y"] <= 1
    with pytest.raises(ValueError, match="outside search_space"):
        opt.optimize(1, initial_params={"typo": 0.3})


def test_ihdp_profile_exposes_more_than_learning_rates():
    profile = ControllerTuner.profile("ihdp")
    assert len(profile["search_space"]) == 8
    assert "incremental_settings.window_size" in profile["native_parameters"]
    assert "critic_settings.Q_weights" in profile["tunable_parameters"]


def test_fixed_native_vector_template_can_have_a_searched_element():
    tuner = ControllerTuner(
        **settings(
            "imgdhp",
            controller_options={
                "config.actor_hidden": (16, 8),
                "config.obs_scale": (2.0,) * 9,
            },
            search_space={"config.actor_hidden.0": Int(4, 12, step=4)},
        )
    )
    agent = build(tuner, {"config.actor_hidden.0": 12})
    assert agent.cfg.actor_hidden == (12, 8)
    assert tuner.controller_options["config.actor_hidden"] == (16, 8)
    assert agent.cfg.obs_scale == (2.0,) * 9


def test_search_alias_cannot_be_silently_overwritten_by_fixed_native_setting():
    with pytest.raises(ValueError, match="overrides searched alias"):
        ControllerTuner(
            **settings(
                "ihdp",
                controller_options={
                    "actor_settings.learning_rate": 0.001,
                },
                search_space={"actor_lr": Float(0.001, 0.01)},
            )
        )


def test_iadp_initial_value_prior_matches_selected_native_cost_and_discount():
    tuner = ControllerTuner(
        **settings(
            "iadp",
            controller_options={
                "config.Q.0.0": 3.0,
                "config.R.0.0": 0.4,
                "config.gamma": 0.9,
            },
        )
    )
    cfg = build(tuner).cfg
    E = np.column_stack((cfg.output_matrix, -np.eye(1)))
    Q = E.T @ cfg.Q @ E + np.eye(cfg.P_init.shape[0]) * 1e-8
    P, F, G, gamma = cfg.P_init, cfg.F_init, cfg.G_init, cfg.gamma
    expected = (
        Q
        + gamma * F.T @ P @ F
        - gamma**2
        * F.T
        @ P
        @ G
        @ np.linalg.solve(cfg.R + gamma * G.T @ P @ G, G.T @ P @ F)
    )
    np.testing.assert_allclose(P, expected, atol=1e-7, rtol=1e-7)


def hdp_settings(**updates):
    return settings(
        "hdp",
        env="ImprovedB747-v0",
        env_kwargs={
            "initial_state": [0.0, 0.0, 0.0, 0.0],
            "dt": 0.02,
        },
        **updates,
    )


def test_hdp_utility_weights_require_active_custom_cost_mode():
    with pytest.raises(ValueError, match="dhp_use_env_cost"):
        ControllerTuner(**hdp_settings(search_space={"dhp_w_theta": Float(1.0, 3.0)}))


def test_hdp_custom_weights_and_episode_cycles_reach_native_training(monkeypatch):
    import tensoraerospace.agent.hdp as module

    native = module.HDP
    agents = []

    class RecordingHDP(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            agents.append(self)

    monkeypatch.setattr(module, "HDP", RecordingHDP)
    tuner = ControllerTuner(
        **hdp_settings(
            controller_options={
                "dhp_use_env_cost": False,
                "dhp_critic_cycle_episodes": 1,
                "dhp_action_cycle_episodes": 1,
                "hidden_size": 8,
            },
            search_space={"dhp_w_theta": Float(1.0, 3.0)},
            training_episodes=1,
        )
    )
    run = tuner.simulate({"dhp_w_theta": 2.0})
    assert not agents[0]._dhp_use_env_cost
    assert agents[0]._dhp_w_theta == 2.0
    assert agents[0]._dhp_critic_cycle_episodes == 1
    assert agents[0]._dhp_action_cycle_episodes == 1
    assert run.output.shape == (16, 1)
    assert np.isfinite(run.metrics["cpi"])


@pytest.mark.parametrize("window", [3, 7, 25])
def test_ihdp_configurable_window_identifies_known_dynamics(window):
    from tensoraerospace.agent.ihdp.Incremental_model import IncrementalModel

    A = np.array([[0.8, 0.1], [-0.1, 0.7]])
    B = np.array([[0.2], [0.4]])
    model = IncrementalModel(
        ["x", "y"], ["u"], 70, 0.1, 10.0, 1000.0, window_size=window
    )
    rng = np.random.default_rng(8)
    x = np.zeros(2)
    for k in range(60):
        u = rng.uniform(-1.0, 1.0, 1)
        model.identify_incremental_model_LS(x, u)
        model.update_incremental_model_attributes()
        x = A @ x + B @ u
    np.testing.assert_allclose(model.F, A, atol=1e-10)
    np.testing.assert_allclose(model.G, B, atol=1e-10)


@pytest.mark.parametrize("window", [0, 2, 3.5, True])
def test_invalid_identification_window_fails_explicitly(window):
    from tensoraerospace.agent.ihdp.Incremental_model import IncrementalModel

    with pytest.raises(ValueError, match="window_size"):
        IncrementalModel(["x", "y"], ["u"], 30, window_size=window)
