"""Round-trip save/load/from_pretrained/publish_to_hub tests for IM-GDHP,
ET-DHP, and IHDP — the three agents that gained Hugging Face Hub support in
this PR.

Each agent follows the same pattern used by the rest of the tensoraerospace
agent family:

    * :meth:`save(path)` writes a directory with ``config.json`` + ``.pth``
      files and returns its absolute path.
    * :meth:`_load_from_dir(folder, ...)` reconstructs the agent from disk.
    * :meth:`from_pretrained(repo_or_local)` dispatches to
      ``_load_from_dir`` for local folders and to
      ``huggingface_hub.snapshot_download`` otherwise.
    * :meth:`publish_to_hub(repo, folder)` delegates to ``HfApi``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig


# ---------------------------------------------------------------------------
# IM-GDHP
# ---------------------------------------------------------------------------
def _mk_imgdhp(**overrides) -> IMGDHPAgent:
    cfg_kw = dict(
        gamma=0.9,
        actor_hidden=(8, 8),
        critic_hidden=(8, 8),
        actor_lr=1e-3,
        critic_lr=1e-3,
        track_Q=[1.0],
        u_max=1.0,
        warmup_steps=1,
        seed=0,
    )
    cfg_kw.update(overrides)
    cfg = IMGDHPConfig(**cfg_kw)
    return IMGDHPAgent(
        n_obs=2,
        n_action=1,
        reference_size=1,
        tracking_indices=[0],
        config=cfg,
    )


def test_imgdhp_save_creates_expected_files(tmp_path):
    agent = _mk_imgdhp()
    run_dir = Path(agent.save(path=tmp_path, save_gradients=True))
    assert run_dir.is_dir()
    for name in (
        "config.json",
        "actor.pth",
        "critic.pth",
        "target_critic.pth",
        "incremental_model.npz",
        "actor_optim.pth",
        "critic_optim.pth",
    ):
        assert (run_dir / name).exists(), f"missing {name}"


def test_imgdhp_load_round_trip_preserves_weights(tmp_path):
    agent = _mk_imgdhp()
    # Perturb weights away from init so the comparison is meaningful.
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    # Run a few steps so the RLS identifier has non-trivial theta / P.
    ref = np.zeros((1, 8))
    obs = np.array([0.1, 0.0])
    for t in range(4):
        action = agent.predict(obs, ref, t)
        obs = obs + 0.01 * np.array([float(action[0]), 0.0])
        agent.learn(obs, ref, t)

    run_dir = agent.save(path=tmp_path)
    restored = IMGDHPAgent.from_pretrained(run_dir)

    # Weights bitwise identical.
    for p_o, p_r in zip(agent.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(p_o, p_r)
    for p_o, p_r in zip(agent.critic.parameters(), restored.critic.parameters()):
        torch.testing.assert_close(p_o, p_r)

    # RLS state restored.
    np.testing.assert_allclose(
        agent.incremental_model.theta, restored.incremental_model.theta
    )
    np.testing.assert_allclose(agent.incremental_model.P, restored.incremental_model.P)
    assert agent.incremental_model.num_updates == restored.incremental_model.num_updates


def test_imgdhp_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        IMGDHPAgent.from_pretrained(str(tmp_path / "nope"))


def test_imgdhp_from_pretrained_snapshot_download(tmp_path, monkeypatch):
    # Save locally first, then stub snapshot_download to point at that folder.
    agent = _mk_imgdhp()
    run_dir = agent.save(path=tmp_path)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.snapshot_download = lambda repo_id, token=None, revision=None: run_dir
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    restored = IMGDHPAgent.from_pretrained("user/some-repo")
    assert isinstance(restored, IMGDHPAgent)


def test_imgdhp_publish_to_hub_uploads_folder(tmp_path, monkeypatch):
    agent = _mk_imgdhp()
    run_dir = agent.save(path=tmp_path)

    called = {}

    class _FakeApi:
        def upload_folder(self, **kwargs):
            called.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    agent.publish_to_hub("me/imgdhp", folder_path=run_dir, access_token="tok")
    assert called["repo_id"] == "me/imgdhp"
    assert called["repo_type"] == "model"
    assert called["token"] == "tok"
    assert Path(called["folder_path"]) == Path(run_dir)


def test_imgdhp_device_fallback_on_load(tmp_path, monkeypatch):
    # Forge a checkpoint that claims it was trained on cuda → loading on
    # a CPU host must silently downgrade to cpu.
    import json

    # Pretend no GPU is present so the cuda→cpu fallback is exercised even on
    # a CUDA-equipped host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    agent = _mk_imgdhp()
    run_dir = Path(agent.save(path=tmp_path))
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["policy"]["config"]["device"] = "cuda"
    cfg_path.write_text(json.dumps(cfg))

    restored = IMGDHPAgent.from_pretrained(run_dir)
    assert restored.device.type == "cpu"


# ---------------------------------------------------------------------------
# ET-DHP
# ---------------------------------------------------------------------------
def _mk_etdhp(state_transform=None, **overrides) -> ETDHPAgent:
    cfg_kw = dict(
        actor_hidden=(6, 6),
        critic_hidden=(6, 6),
        model_hidden=(6, 6),
        actor_lr=1e-3,
        critic_lr=1e-3,
        model_lr=1e-3,
        model_epochs=5,
        Q=[1.0, 0.1],
        R=[1.0],
        gamma=0.95,
        num_epochs_per_trigger=1,
        u_bound=1.0,
        rho=0.2,
        trigger_floor=0.05,
        weight_init_scale=0.2,
        seed=0,
    )
    cfg_kw.update(overrides)
    cfg = ETDHPConfig(**cfg_kw)
    return ETDHPAgent(
        n_state=2,
        n_control=1,
        state_transform=state_transform,
        config=cfg,
    )


def test_etdhp_save_creates_expected_files(tmp_path):
    agent = _mk_etdhp()
    run_dir = Path(agent.save(path=tmp_path, save_gradients=True))
    assert run_dir.is_dir()
    for name in (
        "config.json",
        "actor.pth",
        "critic.pth",
        "plant_model.pth",
        "event_trigger.json",
        "actor_optim.pth",
        "critic_optim.pth",
        "model_optim.pth",
    ):
        assert (run_dir / name).exists(), f"missing {name}"


def test_etdhp_load_round_trip_preserves_weights(tmp_path):
    agent = _mk_etdhp()
    # Perturb weights so there's something to compare.
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    run_dir = agent.save(path=tmp_path)
    restored = ETDHPAgent.from_pretrained(run_dir)

    for p_o, p_r in zip(agent.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(p_o, p_r)
    for p_o, p_r in zip(agent.critic.parameters(), restored.critic.parameters()):
        torch.testing.assert_close(p_o, p_r)
    for p_o, p_r in zip(
        agent.plant_model.parameters(), restored.plant_model.parameters()
    ):
        torch.testing.assert_close(p_o, p_r)


def test_etdhp_state_transform_reattached_on_load(tmp_path):
    calls = {"count": 0}

    def tf(obs, ref, ts):
        calls["count"] += 1
        return np.asarray(obs, dtype=np.float64).reshape(-1)

    agent = _mk_etdhp(state_transform=tf)
    run_dir = agent.save(path=tmp_path)
    # Without supplying the transform, the agent has the default (None).
    restored_plain = ETDHPAgent.from_pretrained(run_dir)
    assert restored_plain._state_transform is None
    # When supplied, it's re-attached correctly.
    restored_with_tf = ETDHPAgent.from_pretrained(run_dir, state_transform=tf)
    assert restored_with_tf._state_transform is tf


def test_etdhp_event_trigger_state_preserved(tmp_path):
    agent = _mk_etdhp()
    # Force an event-trigger state to be non-trivial.
    agent.event_trigger.state_at_trigger = np.array([1.0, 2.0])
    agent.event_trigger.step_at_trigger = 7
    agent.event_trigger.num_triggers = 3
    agent.event_trigger.norm_state_at_trigger = 2.236

    run_dir = agent.save(path=tmp_path)
    restored = ETDHPAgent.from_pretrained(run_dir)
    np.testing.assert_allclose(
        restored.event_trigger.state_at_trigger, np.array([1.0, 2.0])
    )
    assert restored.event_trigger.step_at_trigger == 7
    assert restored.event_trigger.num_triggers == 3


def test_etdhp_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        ETDHPAgent.from_pretrained(str(tmp_path / "missing"))


def test_etdhp_publish_to_hub_uploads_folder(tmp_path, monkeypatch):
    agent = _mk_etdhp()
    run_dir = agent.save(path=tmp_path)

    called = {}

    class _FakeApi:
        def upload_folder(self, **kwargs):
            called.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    agent.publish_to_hub("me/etdhp", folder_path=run_dir)
    assert called["repo_id"] == "me/etdhp"
    assert Path(called["folder_path"]) == Path(run_dir)


# ---------------------------------------------------------------------------
# IHDP
# ---------------------------------------------------------------------------
def _mk_ihdp() -> IHDPAgent:
    actor_settings = dict(
        start_training=3,
        layers=(4, 1),
        activations=("tanh", "tanh"),
        learning_rate=1.0,
        learning_rate_exponent_limit=3,
        type_PE="combined",
        amplitude_3211=1,
        pulse_length_3211=3,
        maximum_input=1,
        maximum_q_rate=1,
        WB_limits=2.0,
        NN_initial=1,
        cascade_actor=False,
        learning_rate_cascaded=0.1,
    )
    critic_settings = dict(
        Q_weights=[1.0],
        start_training=-1,
        gamma=0.9,
        learning_rate=0.1,
        learning_rate_exponent_limit=3,
        layers=(4, 1),
        activations=("tanh", "linear"),
        indices_tracking_states=[0],
        WB_limits=2.0,
        NN_initial=1,
    )
    incremental_settings = dict(
        number_time_steps=5,
        dt=0.1,
        input_magnitude_limits=1,
        input_rate_limits=10,
    )
    return IHDPAgent(
        actor_settings=actor_settings,
        critic_settings=critic_settings,
        incremental_settings=incremental_settings,
        tracking_states=["alpha"],
        selected_states=["alpha"],
        selected_input=["u"],
        number_time_steps=5,
        indices_tracking_states=[0],
    )


def test_ihdp_save_creates_expected_files(tmp_path):
    agent = _mk_ihdp()
    run_dir = Path(agent.save(path=tmp_path))
    assert run_dir.is_dir()
    for name in ("config.json", "actor.pth", "critic.pth"):
        assert (run_dir / name).exists(), f"missing {name}"


def test_ihdp_load_round_trip_preserves_weights(tmp_path):
    agent = _mk_ihdp()
    # Nudge the actor weights so the round-trip comparison isn't trivial.
    with torch.no_grad():
        for p in agent.actor.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    run_dir = agent.save(path=tmp_path)
    restored = IHDPAgent.from_pretrained(run_dir)

    for p_o, p_r in zip(
        agent.actor.model.parameters(), restored.actor.model.parameters()
    ):
        torch.testing.assert_close(p_o, p_r)
    for p_o, p_r in zip(
        agent.critic.model.parameters(), restored.critic.model.parameters()
    ):
        torch.testing.assert_close(p_o, p_r)


def test_ihdp_settings_persisted_in_config(tmp_path):
    agent = _mk_ihdp()
    run_dir = Path(agent.save(path=tmp_path))
    import json

    cfg = json.loads((run_dir / "config.json").read_text())
    params = cfg["policy"]["params"]
    assert params["tracking_states"] == ["alpha"]
    assert params["indices_tracking_states"] == [0]
    assert params["actor_settings"]["type_PE"] == "combined"
    assert params["critic_settings"]["gamma"] == 0.9


def test_ihdp_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        IHDPAgent.from_pretrained(str(tmp_path / "nope"))


def test_ihdp_publish_to_hub_uploads_folder(tmp_path, monkeypatch):
    agent = _mk_ihdp()
    run_dir = agent.save(path=tmp_path)

    called = {}

    class _FakeApi:
        def upload_folder(self, **kwargs):
            called.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    agent.publish_to_hub("me/ihdp", folder_path=run_dir, access_token="tok")
    assert called["repo_id"] == "me/ihdp"
    assert called["token"] == "tok"
