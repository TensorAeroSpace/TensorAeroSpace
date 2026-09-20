"""Check the experiment's actual env/agent boundary, including noisy samples."""

import importlib.util
import json
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import pytest

from tensoraerospace.agent.iadp import IADPAgent


@pytest.mark.parametrize("kind, agent_type", [("iadp", IADPAgent)])
@pytest.mark.parametrize("plant", ["lapan", "b747"])
def test_validation_reuses_same_timestamp_observation(
    monkeypatch, tmp_path, kind, agent_type, plant
):
    path = Path(__file__).with_name("_adaptive_validation.py")
    spec = importlib.util.spec_from_file_location("adaptive_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict, learn = agent_type.predict, agent_type.learn
    seen = {}
    comparisons = []
    episode_costs = []

    def checked_predict(self, observation, reference, time_step=0, **kwargs):
        if self._step == 0:
            episode_costs.append([])
        if self._step:
            np.testing.assert_array_equal(observation, seen[id(self)])
            comparisons.append(True)
        return predict(self, observation, reference, time_step, **kwargs)

    @wraps(learn)
    def checked_learn(self, observation, reference, time_step=0, **kwargs):
        seen[id(self)] = np.asarray(observation).copy()
        metrics = learn(self, observation, reference, time_step, **kwargs)
        if kind == "iadp":
            episode_costs[-1].append(metrics["cost"])
        return metrics

    monkeypatch.setattr(agent_type, "predict", checked_predict)
    monkeypatch.setattr(agent_type, "learn", checked_learn)
    monkeypatch.setattr(agent_type, "save", lambda self, path: str(path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--repo",
            str(path.parents[2]),
            "--output",
            str(tmp_path / "result.json"),
            "--agent",
            kind,
            "--plant",
            plant,
            "--seed",
            "11",
            "--train-duration",
            ".12",
            "--eval-duration",
            ".12",
        ],
    )
    module.main()
    assert len(comparisons) == 45  # 9 episodes, 6 samples, 5 timestamp boundaries.
    if kind == "iadp":
        record = json.loads((tmp_path / "result.json").read_text())["training"]
        # Training has no sensor noise: physical and agent cost must agree.
        assert record["mean_physical_cost"] == pytest.approx(
            np.mean(episode_costs[0]), rel=1e-6
        )
        assert record["discounted_physical_cost"] == pytest.approx(
            np.dot(np.power(0.99, np.arange(6)), episode_costs[0]), rel=1e-6
        )


def test_public_b747_observation_preserves_velocity_units():
    path = Path(__file__).with_name("_adaptive_validation.py")
    spec = importlib.util.spec_from_file_location("adaptive_validator_units", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    public = np.array([12.0, -3.0, 180.0, -90.0], dtype=np.float32)
    converted = module.observation_in_si(public, "b747", True)
    np.testing.assert_allclose(converted, [12.0, -3.0, np.pi, -np.pi / 2])
    np.testing.assert_array_equal(public, [12.0, -3.0, 180.0, -90.0])


@pytest.mark.parametrize("dt", [0.005, 0.02, 0.1])
@pytest.mark.parametrize("phase", [0.2, 0.7, 1.3])
def test_oscillator_reference_obeys_autonomous_transition(dt, phase):
    path = Path(__file__).with_name("_adaptive_validation.py")
    spec = importlib.util.spec_from_file_location("adaptive_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    time = np.arange(100) * dt
    reference = module.tracking_reference(time, phase, True, "oscillator")
    expected_q = np.deg2rad(
        0.5 * np.sin(2 * np.pi * 0.12 * time)
        + 0.15 * np.sin(2 * np.pi * 0.31 * time + phase)
    )
    np.testing.assert_array_equal(reference[2], expected_q)
    np.testing.assert_allclose(
        module.reference_transition(dt) @ reference[:, :-1],
        reference[:, 1:],
        atol=2e-16,
        rtol=1e-12,
    )
