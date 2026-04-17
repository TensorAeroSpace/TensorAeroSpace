"""Coverage tests for the NARX-flavoured A2C helpers.

Exercises:
    * ``Mish`` / ``mish`` / ``t`` / ``clip_grad_norm_`` utilities
    * ``discounted_rewards`` and ``process_memory_narx``
    * ``A2CLearner.learn`` path with a real ``Runner``
    * ``A2CLearner.save`` / ``_load`` / ``from_pretrained`` round-trip
    * ``Runner`` reset + run loop with a tiny Gym env
"""

from __future__ import annotations

import json
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tensoraerospace.agent.a2c.narx import (
    A2CLearner,
    Actor,
    Critic,
    Mish,
    Runner,
    clip_grad_norm_,
    discounted_rewards,
    mish,
    process_memory_narx,
    t,
)


@pytest.fixture
def simple_env():
    """Pendulum-v1 is a tiny continuous-action env — ideal for unit-speed tests."""
    env = gym.make("Pendulum-v1")
    yield env
    env.close()


@pytest.fixture
def learner(simple_env):
    state_dim = simple_env.observation_space.shape[0]
    n_actions = simple_env.action_space.shape[0]
    actor = Actor(state_dim, n_actions, activation=Mish)
    critic = Critic(state_dim, activation=Mish)
    return A2CLearner(actor, critic, entropy_beta=0.01, device="cpu")


# ---- utility helpers ------------------------------------------------------
def test_mish_matches_manual():
    x = torch.linspace(-2.0, 2.0, 11)
    expected = x * torch.tanh(torch.nn.functional.softplus(x))
    torch.testing.assert_close(mish(x), expected)


def test_mish_module_forward_equals_function():
    x = torch.randn(5)
    torch.testing.assert_close(Mish()(x), mish(x))


def test_t_converts_list_and_array_to_float_tensor():
    tensor_from_list = t([1, 2, 3])
    assert tensor_from_list.dtype == torch.float32
    assert tensor_from_list.shape == (3,)
    tensor_from_arr = t(np.array([1.0, 2.0], dtype=np.float64))
    assert tensor_from_arr.dtype == torch.float32


def test_clip_grad_norm_clips_optimiser_params():
    layer = nn.Linear(3, 1)
    # Synthetic gradient with large norm.
    for p in layer.parameters():
        p.grad = torch.full_like(p, 10.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    clip_grad_norm_(opt, max_grad_norm=1.0)
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in layer.parameters()))
    assert float(total_norm) <= 1.0 + 1e-6


# ---- discounted-reward / memory processing -------------------------------
def test_discounted_rewards_terminal_resets_trail():
    dr = discounted_rewards([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], gamma=0.5)
    # Middle step is terminal → accumulator resets before it in the reverse pass.
    assert dr[2] == pytest.approx(1.0)
    assert dr[1] == pytest.approx(1.0)  # no future reward beyond terminal
    assert dr[0] == pytest.approx(1.0 + 0.5 * 1.0)


def test_process_memory_narx_concatenates_previous_state():
    mem = [
        (np.array([0.1]), 1.0, np.array([1.0, 2.0]), np.array([1.1, 2.1]), False),
        (np.array([0.2]), 0.5, np.array([1.1, 2.1]), np.array([1.2, 2.2]), True),
    ]
    actions, rewards, states, next_states, dones, critic_states = process_memory_narx(
        mem, gamma=0.9
    )
    assert actions.shape == (2, 1)
    # Critic input is state_t concatenated with state_{t-1} (zeros at t=0).
    assert critic_states.shape == (2, 4)
    # Reward is discounted by the helper.
    # r0 = 1.0 + gamma*0.5*(1-terminal at step 1); since step 1 is terminal, reset.
    expected_r0 = 1.0 + 0.9 * 0.5 * (1 - 0)
    assert rewards[0].item() == pytest.approx(expected_r0, rel=1e-5)
    assert dones[1].item() == 1.0


def test_process_memory_narx_no_discount_passes_raw_rewards():
    mem = [
        (np.array([0.0]), 2.0, np.array([1.0]), np.array([1.5]), False),
    ]
    _, rewards, *_ = process_memory_narx(mem, gamma=0.9, discount_rewards=False)
    assert rewards[0].item() == pytest.approx(2.0)


# ---- A2CLearner / Runner integration -------------------------------------
def test_runner_collects_transitions(simple_env, learner):
    runner = Runner(simple_env, learner.actor, writer=learner.writer)
    memory = runner.run(4)
    assert len(memory) == 4
    # Each transition is (action, reward, state, next_state, done).
    assert len(memory[0]) == 5
    assert runner.steps == 4


def test_runner_reset_clears_episode_state(simple_env, learner):
    runner = Runner(simple_env, learner.actor, writer=learner.writer)
    runner.run(2)
    runner.reset()
    assert runner.done is False
    assert runner.episode_reward == 0


def test_a2c_learner_learn_updates_nets(simple_env, learner):
    runner = Runner(simple_env, learner.actor, writer=learner.writer)
    memory = runner.run(8)
    params_before = [p.detach().clone() for p in learner.actor.parameters()]
    learner.learn(memory, steps=runner.steps, discount_rewards=True)
    params_after = [p.detach().clone() for p in learner.actor.parameters()]
    # At least one parameter must have moved.
    deltas = [
        float((a - b).abs().sum()) for a, b in zip(params_after, params_before)
    ]
    assert max(deltas) > 0.0


def test_a2c_learner_learn_without_discount_runs(simple_env, learner):
    runner = Runner(simple_env, learner.actor, writer=learner.writer)
    memory = runner.run(4)
    learner.learn(memory, steps=runner.steps, discount_rewards=False)


# ---- save / _load round-trip --------------------------------------------
def test_get_param_env_returns_reconstructible_config(learner):
    cfg = learner.get_param_env()
    assert "policy" in cfg
    assert "name" in cfg["policy"]
    assert "params" in cfg["policy"]
    assert cfg["policy"]["params"]["state_dim"] > 0
    assert cfg["policy"]["params"]["n_actions"] > 0


def test_save_and_load_round_trip(tmp_path, simple_env, learner):
    save_dir = learner.save(tmp_path, save_gradients=True)
    assert save_dir.exists()
    # The directory contains the expected artefacts.
    assert (save_dir / "config.json").exists()
    assert (save_dir / "actor.pth").exists()
    assert (save_dir / "critic.pth").exists()
    assert (save_dir / "actor_optim.pth").exists()
    assert (save_dir / "critic_optim.pth").exists()
    # Load back with gradients.
    restored = A2CLearner._load(save_dir, load_gradients=True)
    # Actor weights match bitwise after round-trip.
    for p_orig, p_new in zip(
        learner.actor.parameters(), restored.actor.parameters()
    ):
        torch.testing.assert_close(p_orig, p_new)


def test_save_with_default_path_creates_cwd_dir(tmp_path, monkeypatch, learner):
    monkeypatch.chdir(tmp_path)
    save_dir = learner.save(None)
    assert save_dir.parent == tmp_path


def test_from_pretrained_local_directory(tmp_path, learner):
    save_dir = learner.save(tmp_path)
    restored = A2CLearner.from_pretrained(str(save_dir))
    assert isinstance(restored, A2CLearner)


def test_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        A2CLearner.from_pretrained(str(tmp_path / "does-not-exist"))


def test_load_cuda_device_fallback_when_unavailable(tmp_path, learner, monkeypatch):
    # Save with device=cuda recorded, then ensure _load falls back to cpu.
    save_dir = learner.save(tmp_path)
    config_path = save_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["policy"]["params"]["device"] = "cuda"
    config_path.write_text(json.dumps(cfg))
    restored = A2CLearner._load(save_dir)
    assert str(next(restored.actor.parameters()).device) == "cpu"


def test_load_mps_device_fallback_when_unavailable(tmp_path, learner):
    save_dir = learner.save(tmp_path)
    config_path = save_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["policy"]["params"]["device"] = "mps"
    config_path.write_text(json.dumps(cfg))
    restored = A2CLearner._load(save_dir)
    assert str(next(restored.actor.parameters()).device) == "cpu"


def test_publish_to_hub_passes_arguments_through(tmp_path, learner, monkeypatch):
    # Stub ``huggingface_hub`` to avoid network calls while exercising the path.
    calls = {}

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    save_dir = learner.save(tmp_path)
    learner.publish_to_hub("me/my-model", folder_path=save_dir, access_token="tok")
    assert calls["repo_id"] == "me/my-model"
    assert calls["token"] == "tok"


def test_from_pretrained_via_snapshot_download(tmp_path, learner, monkeypatch):
    # Stub snapshot_download so the Hub code path is exercised without network.
    save_dir = learner.save(tmp_path)
    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.snapshot_download = lambda repo_id, token=None, revision=None: str(save_dir)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)
    # A repo id that does NOT look like a local path routes through snapshot_download.
    restored = A2CLearner.from_pretrained("some-user/some-repo")
    assert isinstance(restored, A2CLearner)
