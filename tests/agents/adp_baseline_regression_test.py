"""Regression tests for ADP baseline composition and supervised warm start."""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.adp.adp import ADP


class LinearTrackingEnv(gym.Env):
    """Deterministic plant with known Jacobians and normalized elevator input."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    dt = 0.1
    max_stabilizer_angle_deg = 10.0
    max_pitch_rad = 0.4
    max_pitch_rate_rad_s = 0.2
    ref_theta_dot_clip_rad_s = 0.2

    def __init__(self, *, truncated=False):
        self.initial_state = np.array([0.01, -0.02, 0.01, 0.02], dtype=np.float32)
        self.reference_signal = np.full((1, 20), 0.06, dtype=np.float32)
        self.model = SimpleNamespace(
            filt_A=np.eye(4, dtype=np.float32) * 0.95,
            filt_B=np.array([[0.01], [0.02], [0.1], [0.05]], dtype=np.float32),
        )
        self.truncated = truncated
        self.records = []
        self.reset()

    def observation(self):
        return np.concatenate([self.state, [self.reference_signal[0, 0], 0.0]]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.initial_state.copy()
        return self.observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.records.append((self.observation(), action.copy()))
        physical_action = np.deg2rad(self.max_stabilizer_angle_deg) * action
        self.state = (
            self.model.filt_A @ self.state + self.model.filt_B @ physical_action
        )
        self.current_step += 1
        error = float(self.state[3] - self.reference_signal[0, 0])
        cost = error**2 + 0.01 * float(action[0]) ** 2
        done = self.current_step >= 3
        return (
            self.observation(),
            -cost,
            done and not self.truncated,
            done and self.truncated,
            {"cost_total": cost},
        )


BASELINES = [
    pytest.param("pd", "norm", True, id="pd"),
    pytest.param("pid", "norm", True, id="pid-normalized"),
    pytest.param("pid", "norm", False, id="pid-radians"),
    pytest.param("pid", "deg", False, id="pid-degrees"),
]


@pytest.fixture
def agent_factory(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    agents = []

    def make(**overrides):
        settings = dict(
            env=LinearTrackingEnv(),
            design="dhp",
            device="cpu",
            hidden_size=8,
            seed=23,
            exploration_std=0.0,
            log_every_updates=1,
            dhp_use_baseline=True,
            dhp_baseline_kp=0.6,
            dhp_baseline_ki=0.0,
            dhp_baseline_kd=0.0,
        )
        settings.update(overrides)
        agent = ADP(**settings)
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.writer.close()
        agent.env.close()


@pytest.mark.parametrize("baseline,mode,normalized", BASELINES)
def test_baseline_units_and_residual_composition(
    agent_factory, baseline, mode, normalized
):
    agent = agent_factory(
        dhp_baseline_type=baseline,
        dhp_pid_mode=mode,
        dhp_pid_use_normalized_theta=normalized,
        dhp_residual_scale=0.25,
    )
    obs, _ = agent.env.reset()
    error = float(obs[4] - obs[3])
    if baseline == "pd" or (mode == "norm" and normalized):
        expected_base = 0.6 * error / agent.env.max_pitch_rad
    elif mode == "deg":
        expected_base = 0.6 * np.rad2deg(error) / agent.env.max_stabilizer_angle_deg
    else:
        expected_base = 0.6 * error

    actual_base = agent._dhp_baseline_u_norm(
        x=obs[:4], theta_ref=float(obs[4]), q_ref=0.0
    )
    assert actual_base == pytest.approx(expected_base, abs=1e-7)
    with torch.no_grad():
        residual = agent.actor(torch.as_tensor(obs).unsqueeze(0)).item()
    agent.reset()
    actual = agent.select_action(obs, evaluate=True)
    np.testing.assert_allclose(actual, [expected_base + 0.25 * residual], atol=1e-7)
    with pytest.raises(ValueError, match="expects R\\(t\\)"):
        agent.select_action(obs[:4])


@pytest.mark.parametrize("baseline,mode,normalized", BASELINES)
@pytest.mark.parametrize("design", ["dhp", "hdp", "gdhp", "addhp", "adgdhp"])
def test_training_with_baseline_updates_actor_and_critic(
    agent_factory, baseline, mode, normalized, design
):
    agent = agent_factory(
        design=design,
        dhp_baseline_type=baseline,
        dhp_pid_mode=mode,
        dhp_pid_use_normalized_theta=normalized,
        dhp_actor_delta_l2=0.1,
        dhp_residual_scale=0.25,
        exploration_std=0.02,
        dhp_use_env_cost=False,
    )
    before_actor = [p.detach().clone() for p in agent.actor.parameters()]
    before_critic = [p.detach().clone() for p in agent.critic.parameters()]
    agent.train(num_episodes=2, max_steps=3)

    assert len(agent.env.records) == 6
    assert agent._updates == 6
    for old, network in ((before_actor, agent.actor), (before_critic, agent.critic)):
        assert any(not torch.equal(a, b) for a, b in zip(old, network.parameters()))
        assert all(torch.isfinite(p).all() for p in network.parameters())
    for _, action in agent.env.records:
        assert np.isfinite(action).all()
        assert agent.env.action_space.contains(action)
    agent.writer.assert_contract_satisfied()


@pytest.mark.parametrize("baseline,mode,normalized", BASELINES)
@pytest.mark.parametrize("truncated", [False, True])
def test_warmstart_reduces_imitation_error_without_updating_critic(
    agent_factory, baseline, mode, normalized, truncated
):
    agent = agent_factory(
        env=LinearTrackingEnv(truncated=truncated),
        dhp_baseline_type=baseline,
        dhp_pid_mode=mode,
        dhp_pid_use_normalized_theta=normalized,
        dhp_baseline_ki=0.1,
        dhp_warmstart_actor_episodes=2,
        dhp_warmstart_actor_epochs=40,
        actor_lr=0.005,
    )
    before_actor = deepcopy(agent.actor)
    before_critic = [p.detach().clone() for p in agent.critic.parameters()]
    agent._warmstart_actor_from_baseline(episodes=2, max_steps=None)
    assert len(agent.env.records) == 6
    observations = torch.as_tensor(np.stack([o for o, _ in agent.env.records]))
    targets = torch.as_tensor(np.stack([a for _, a in agent.env.records]))
    with torch.no_grad():
        initial_error = torch.mean((before_actor(observations) - targets) ** 2).item()
        final_error = torch.mean((agent.actor(observations) - targets) ** 2).item()
    assert final_error < initial_error * 0.2
    for old, new in zip(before_critic, agent.critic.parameters()):
        torch.testing.assert_close(old, new)
    # Episode reset must also reset PID history: identical plant resets produce
    # identical demonstrations from a deterministic baseline controller.
    for first, second in zip(agent.env.records[:3], agent.env.records[3:]):
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("disable_baseline", [False, True])
def test_train_warmstart_switches_baseline_as_configured(
    agent_factory, disable_baseline
):
    agent = agent_factory(
        dhp_baseline_type="pid",
        dhp_baseline_ki=0.1,
        dhp_warmstart_actor_episodes=1,
        dhp_warmstart_actor_epochs=3,
        dhp_warmstart_actor_disable_baseline_after=disable_baseline,
    )
    agent.train(num_episodes=1, max_steps=2)
    assert len(agent.env.records) == 4  # two demonstration + two learning steps
    assert agent._updates == 2
    assert agent._dhp_use_baseline is (not disable_baseline)
    agent.reset()
    obs, _ = agent.env.reset()
    first_action = agent.select_action(obs, evaluate=True)
    agent.select_action(obs, evaluate=True)
    agent.reset()
    np.testing.assert_allclose(agent.select_action(obs, evaluate=True), first_action)
    np.testing.assert_array_equal(agent._prev_u_norm, [0.0])
    np.testing.assert_array_equal(agent._prev2_u_norm, [0.0])


@pytest.mark.parametrize("design", ["dhp", "hdp", "gdhp", "addhp", "adgdhp"])
def test_critic_cycle_keeps_actor_fixed(agent_factory, design):
    agent = agent_factory(
        design=design, dhp_critic_cycle_episodes=1, dhp_action_cycle_episodes=1
    )
    before_actor = [p.detach().clone() for p in agent.actor.parameters()]
    before_critic = [p.detach().clone() for p in agent.critic.parameters()]
    agent.train(num_episodes=1, max_steps=3)
    for old, new in zip(before_actor, agent.actor.parameters()):
        torch.testing.assert_close(old, new)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_critic, agent.critic.parameters())
    )
