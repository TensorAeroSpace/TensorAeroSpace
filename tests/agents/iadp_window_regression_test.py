"""Data-window scheduling is distinct from lifetime RLS warmup."""

import json

import numpy as np
import pytest

from tensoraerospace.agent.iadp import IADPAgent, IADPConfig


def advance(agent, count):
    for _ in range(count):
        k = agent._step
        state = np.array([0.1 * np.sin(0.3 * k)])
        agent.predict(state, np.zeros(1), k)
        agent.learn(np.array([0.1 * np.sin(0.3 * (k + 1))]), np.zeros(1), k)


def make_agent():
    return IADPAgent(
        1,
        1,
        IADPConfig(
            policy_eval_window=10,
            policy_eval_min_samples=10,
            policy_eval_every=1,
            policy_eval_warmup_updates=0,
        ),
    )


def test_reset_waits_for_new_critic_data_despite_trained_identifier(monkeypatch):
    agent = make_agent()
    calls = []
    monkeypatch.setattr(
        agent, "_policy_evaluation", lambda: calls.append(len(agent._window))
    )
    advance(agent, 12)
    assert calls == [10, 10, 10]
    updates = agent.rls.num_updates
    agent.reset()
    assert agent.rls.num_updates == updates
    advance(agent, 9)
    assert calls == [10, 10, 10]
    advance(agent, 1)
    assert calls == [10, 10, 10, 10]


def test_sample_threshold_and_partial_window_survive_checkpoint(tmp_path, monkeypatch):
    agent = make_agent()
    advance(agent, 5)
    folder = agent.save(tmp_path)
    restored = IADPAgent.from_pretrained(folder)
    assert restored.cfg.policy_eval_min_samples == 10
    calls = []
    monkeypatch.setattr(
        restored, "_policy_evaluation", lambda: calls.append(len(restored._window))
    )
    advance(restored, 4)
    assert calls == []
    advance(restored, 1)
    assert calls == [10]


def test_legacy_checkpoint_retains_legacy_sample_threshold(tmp_path, monkeypatch):
    agent = make_agent()
    folder = agent.save(tmp_path)
    from pathlib import Path

    path = Path(folder) / "config.json"
    config = json.loads(path.read_text())
    config["policy"]["config"].pop("policy_eval_min_samples")
    path.write_text(json.dumps(config))
    restored = IADPAgent.from_pretrained(folder)
    assert restored.cfg.policy_eval_min_samples is None
    calls = []
    monkeypatch.setattr(
        restored, "_policy_evaluation", lambda: calls.append(len(restored._window))
    )
    advance(restored, 4)
    assert calls == [4]


@pytest.mark.parametrize("count", [0, -1, 3, 4.5, 11, np.nan, np.inf])
def test_unreachable_or_invalid_sample_threshold_rejected(count):
    with pytest.raises(ValueError, match="policy_eval_min_samples"):
        IADPAgent(
            1, 1, IADPConfig(policy_eval_window=10, policy_eval_min_samples=count)
        )
