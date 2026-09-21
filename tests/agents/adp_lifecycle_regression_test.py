"""Applied controls and resumed online learning must preserve the transition."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.agent.ihdp import IHDPAgent
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig


def make_imgdhp(**overrides):
    settings = dict(
        actor_hidden=(4,),
        critic_hidden=(4,),
        warmup_steps=2,
        target_update_tau=0.1,
        exploration_noise_std=0.05,
        seed=19,
    )
    settings.update(overrides)
    return IMGDHPAgent(1, 1, config=IMGDHPConfig(**settings))


def make_etdhp(**overrides):
    settings = dict(
        actor_hidden=(4,),
        critic_hidden=(4,),
        model_hidden=(4,),
        num_epochs_per_trigger=1,
        trigger_floor=0.05,
        seed=19,
    )
    settings.update(overrides)
    return ETDHPAgent(1, 1, config=ETDHPConfig(**settings))


def assert_same_network(left, right):
    for original, restored in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)


@pytest.mark.parametrize("pending_transition", [False, True])
def test_imgdhp_resumes_the_same_updates_and_exploration(tmp_path, pending_transition):
    agent = make_imgdhp()
    reference = np.zeros((1, 20))
    state = np.array([0.1])
    for step in range(5):
        action = agent.predict(state, reference, step)
        state = 0.8 * state + 0.5 * action
        agent.learn(state, reference, step)
    if pending_transition:
        action = agent.predict(state, reference, 5)
        state = 0.8 * state + 0.5 * action
    folder = agent.save(tmp_path, save_gradients=True)
    restored = IMGDHPAgent.from_pretrained(folder, load_gradients=True)
    assert restored._total_steps == agent._total_steps
    if pending_transition:
        expected = agent.learn(state, reference, 5)
        actual = restored.learn(state, reference, 5)
        assert actual == pytest.approx(expected)
    for step in range(6 if pending_transition else 5, 10):
        action = agent.predict(state, reference, step)
        resumed_action = restored.predict(state, reference, step)
        np.testing.assert_array_equal(resumed_action, action)
        state = 0.8 * state + 0.5 * action
        expected = agent.learn(state, reference, step)
        actual = restored.learn(state, reference, step)
        assert actual == pytest.approx(expected)
        np.testing.assert_array_equal(
            restored.incremental_model.theta, agent.incremental_model.theta
        )
        for name in ("actor", "critic", "target_critic"):
            assert_same_network(getattr(agent, name), getattr(restored, name))


def test_imgdhp_exploration_is_not_changed_by_other_numpy_users():
    first = make_imgdhp()
    second = make_imgdhp()
    reference = np.zeros((1, 8))
    for step in range(5):
        a = first.predict(np.zeros(1), reference, step)
        np.random.normal(size=100)
        b = second.predict(np.zeros(1), reference, step)
        np.testing.assert_array_equal(a, b)


def test_imgdhp_reset_discards_transition_but_keeps_learning_progress():
    agent = make_imgdhp()
    ref = np.zeros((1, 10))
    for step in range(4):
        agent.predict(np.array([0.1]), ref, step)
        agent.learn(np.array([0.2]), ref, step)
    theta = agent.incremental_model.theta.copy()
    covariance = agent.incremental_model.P.copy()
    updates = agent.incremental_model.num_updates
    agent.reset()
    assert agent._total_steps == 4
    agent.predict(np.array([9.0]), ref, 0)
    agent.learn(np.array([9.1]), ref, 0)
    assert agent.incremental_model.num_updates == updates
    np.testing.assert_array_equal(agent.incremental_model.theta, theta)
    np.testing.assert_array_equal(agent.incremental_model.P, covariance)


@pytest.mark.parametrize("pending_transition", [False, True])
def test_etdhp_resume_preserves_held_control_and_next_update(
    tmp_path, pending_transition
):
    agent = make_etdhp(online_model_fit=True)
    agent.predict(np.array([0.2]), None, 0)
    agent.learn(np.array([0.25]), None, 0, dt=0.02)
    if pending_transition:
        agent.predict(np.array([0.251]), None, 1)
    agent.global_env_step = 7
    agent.update_count = 3
    folder = agent.save(tmp_path, save_gradients=True)
    restored = ETDHPAgent.from_pretrained(folder, load_gradients=True)
    np.testing.assert_array_equal(restored.last_action(), agent.last_action())
    assert restored.global_env_step == 7
    assert restored.update_count == 3
    for step, next_state in ((1, 0.252), (2, 0.5)):
        if not (pending_transition and step == 1):
            measurement = np.array([0.251 if step == 1 else 0.252])
            np.testing.assert_array_equal(
                restored.predict(measurement, None, step),
                agent.predict(measurement, None, step),
            )
        expected = agent.learn(np.array([next_state]), None, step, dt=0.02)
        actual = restored.learn(np.array([next_state]), None, step, dt=0.02)
        assert actual["triggered"] == expected["triggered"] == float(step == 2)
        np.testing.assert_array_equal(restored.last_action(), agent.last_action())
        assert restored._last_time_sec == agent._last_time_sec
        for name in ("actor", "critic", "plant_model"):
            assert_same_network(getattr(agent, name), getattr(restored, name))


def test_etdhp_can_save_and_reattach_exploration_callback(tmp_path):
    def excitation(time_sec):
        return np.array([0.1 * np.sin(time_sec)])

    agent = make_etdhp(exploration_fn=excitation)
    folder = Path(agent.save(tmp_path))
    json.loads((folder / "config.json").read_text())
    plain = ETDHPAgent.from_pretrained(folder)
    assert plain.cfg.exploration_fn is None
    restored = ETDHPAgent.from_pretrained(folder, exploration_fn=excitation)
    assert restored.cfg.exploration_fn is excitation


@pytest.mark.parametrize("enabled", [False, True])
def test_etdhp_online_model_fit_uses_latest_transition_only_at_events(
    enabled, monkeypatch
):
    agent = make_etdhp(online_model_fit=enabled)
    fits = []
    original_fit = agent.fit_plant_model

    def record_fit(states, actions, next_states, **kwargs):
        fits.append((states.copy(), actions.copy(), next_states.copy()))
        return original_fit(states, actions, next_states, **kwargs)

    monkeypatch.setattr(agent, "fit_plant_model", record_fit)
    before = [p.detach().clone() for p in agent.plant_model.parameters()]
    agent.predict(np.array([0.2]), None, 0)
    assert agent.learn(np.array([0.25]), None, 0)["triggered"] == 1.0
    agent.predict(np.array([0.25]), None, 1)
    assert agent.learn(np.array([0.26]), None, 1)["triggered"] == 0.0
    assert len(fits) == int(enabled)
    held_action = agent.predict(np.array([0.26]), None, 2)
    assert agent.learn(np.array([0.5]), None, 2)["triggered"] == 1.0
    if enabled:
        assert len(fits) == 2
        np.testing.assert_array_equal(fits[-1][0], [[0.26]])
        np.testing.assert_array_equal(fits[-1][1], held_action.reshape(1, 1))
        np.testing.assert_array_equal(fits[-1][2], [[0.5]])
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, agent.plant_model.parameters())
        )
    else:
        assert not fits
        for a, b in zip(before, agent.plant_model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def make_ihdp(magnitude, rate):
    actor = dict(
        start_training=10,
        layers=(1,),
        activations=("tanh",),
        learning_rate=0.01,
        learning_rate_exponent_limit=3,
        type_PE=None,
        amplitude_3211=0,
        pulse_length_3211=1,
        maximum_input=1,
        maximum_q_rate=1,
        WB_limits=5,
        NN_initial=1,
        cascade_actor=False,
        learning_rate_cascaded=0.01,
    )
    critic = dict(
        Q_weights=[1.0],
        start_training=10,
        gamma=0.9,
        learning_rate=0.01,
        learning_rate_exponent_limit=3,
        layers=(1,),
        activations=("linear",),
        indices_tracking_states=[0],
        WB_limits=5,
        NN_initial=1,
    )
    incremental = dict(
        number_time_steps=5,
        dt=0.1,
        input_magnitude_limits=magnitude,
        input_rate_limits=rate,
    )
    return IHDPAgent(actor, critic, incremental, ["alpha"], ["alpha"], ["u"], 5, [0])


@pytest.mark.parametrize(
    "magnitude,rate,expected",
    [
        (0.3, 100.0, [0.3, -0.3]),
        (1.0, 1.0, [0.8, 0.7]),
        (0.3, 1.0, [0.3, 0.2]),
    ],
)
def test_ihdp_returns_the_command_used_by_its_identifier(magnitude, rate, expected):
    agent = make_ihdp(magnitude, rate)
    reference = np.zeros((1, 5))
    for step, desired in enumerate((0.8, -0.8)):
        with torch.no_grad():
            agent.actor.model[0].weight.zero_()
            agent.actor.model[0].bias.fill_(float(np.arctanh(desired)))
        command = agent.predict(np.zeros((1, 1)), reference, step)
        np.testing.assert_allclose(command, [[expected[step]]], atol=1e-7)
        np.testing.assert_array_equal(command, agent.incremental_model.ut)
        np.testing.assert_array_equal(command, agent.actor.ut)
        command[:] = 99.0
        np.testing.assert_allclose(
            agent.incremental_model.ut, [[expected[step]]], atol=1e-7
        )


@pytest.mark.parametrize(
    "agent_class,factory,state_file",
    [
        (IMGDHPAgent, make_imgdhp, "training_state.json"),
        (ETDHPAgent, make_etdhp, "control_state.json"),
    ],
)
def test_legacy_checkpoints_load_with_fresh_episode_state(
    tmp_path, agent_class, factory, state_file
):
    agent = factory()
    reference = np.zeros((1, 8))
    agent.predict(np.array([0.2]), reference, 0)
    agent.learn(np.array([0.25]), reference, 0)
    folder = Path(agent.save(tmp_path))
    (folder / state_file).unlink()
    restored = agent_class.from_pretrained(folder)
    assert_same_network(agent.actor, restored.actor)
    if isinstance(restored, ETDHPAgent):
        assert restored.event_trigger.num_triggers == 0
        assert restored._last_state is None
    else:
        assert restored._total_steps == 0
        assert restored._last_obs is None
    assert np.isfinite(restored.predict(np.array([0.4]), reference, 0)).all()


def test_etdhp_load_from_hub_forwards_exploration_callback(tmp_path, monkeypatch):
    import sys
    import types

    callback = lambda time_sec: np.array([time_sec * 0.01])
    folder = make_etdhp(exploration_fn=callback).save(tmp_path)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: folder
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    restored = ETDHPAgent.from_pretrained("test/agent", exploration_fn=callback)
    assert restored.cfg.exploration_fn is callback


def test_etdhp_reset_discards_pending_transition():
    agent = make_etdhp(online_model_fit=True)
    agent.predict(np.array([0.2]), None, 0)
    agent.reset()
    with pytest.raises(RuntimeError, match="predict"):
        agent.learn(np.array([0.3]), None, 0)
    assert agent.event_trigger.num_triggers == 0
    assert not agent.model_opt.state
