"""Regression tests for PPO's optional immediate-reward prediction task."""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tensoraerospace.agent.metrics import schema
from tensoraerospace.agent.ppo.model import PPO, Actor


class _VectorRewardEnv:
    """Small batched environment with a reward associated with each state."""

    def __init__(self):
        self.num_envs = 2
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.unwrapped = self
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.full((2, 3), 0.1 * (self.steps % 4), dtype=np.float32)
        return obs, np.array([1.0, 2.0]), np.zeros(2, bool), np.zeros(2, bool), {}

    def close(self):
        pass


@pytest.fixture
def make_agent(tmp_path):
    agents = []

    def make(**kwargs):
        torch.manual_seed(123)
        env = kwargs.pop("env", None)
        if env is None:
            env = gym.make("Pendulum-v1").unwrapped
        params = dict(
            device="cpu",
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            normalize_obs=False,
            normalize_reward=False,
            auxiliary_coef=0.25,
            entropy_coef=0.0,
            actor_log_std_min=-3.0,
            actor_log_std_max=-1.0,
            max_episodes=1,
            rollout_len=16,
            num_epochs=2,
            batch_size=8,
            eval_freq=100,
            save_best_model=False,
            log_dir=tmp_path / f"agent-{len(agents)}",
        )
        params.update(kwargs)
        agent = PPO(env, **params)
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()
        agent.env.close()


def _batch(agent):
    """Use zero advantages to isolate the auxiliary actor gradient."""
    states = torch.tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [0.1, 0.3, 0.5]]
    )
    with torch.no_grad():
        actions, dist = agent.actor(states)
        old_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        values = agent.critic(states)
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    return states, actions, torch.zeros(4, 1), old_probs, values, rewards, values


@pytest.mark.parametrize("column", [False, True])
def test_reward_mse_pairs_each_state_with_its_target(make_agent, column):
    agent = make_agent()
    with torch.no_grad():
        agent.actor.r.weight.zero_()
        agent.actor.r.bias.fill_(1.0)
    states = torch.zeros(2, 3, dtype=torch.float64)
    rewards = torch.tensor([0.0, 4.0], requires_grad=True)
    targets = rewards[:, None] if column else rewards
    rng = torch.get_rng_state().clone()

    loss = agent.auxiliary_task(states, targets)

    assert loss.item() == pytest.approx(5.0)
    assert loss.dtype == torch.float32
    assert torch.equal(torch.get_rng_state(), rng)
    assert agent.auxillary_task(states, targets).item() == loss.item()
    loss.backward()
    assert rewards.grad is None
    assert agent.actor.r.bias.grad is not None


@pytest.mark.parametrize("coefficient", [0.1, 0.5])
def test_learn_applies_weighted_auxiliary_gradient_to_shared_actor(
    make_agent, coefficient
):
    agent = make_agent(auxiliary_coef=coefficient, max_grad_norm=1e6)
    agent.a_opt = torch.optim.SGD(agent.actor.parameters(), lr=0.05)
    batch = _batch(agent)
    states, _, _, _, _, rewards, _ = batch
    parameters = dict(agent.actor.named_parameters())
    before = {name: param.detach().clone() for name, param in parameters.items()}
    expected_loss = agent.auxiliary_task(states, rewards)
    gradients = torch.autograd.grad(
        expected_loss, tuple(parameters.values()), allow_unused=True
    )

    metrics = agent.learn(*batch)

    assert metrics["auxiliary_loss"] == pytest.approx(expected_loss.item())
    for (name, param), grad in zip(parameters.items(), gradients):
        expected = (
            before[name] if grad is None else before[name] - 0.05 * coefficient * grad
        )
        torch.testing.assert_close(param, expected)
    assert not torch.equal(agent.actor.r.weight, before["r.weight"])
    assert not torch.equal(agent.actor.d1.weight, before["d1.weight"])
    assert agent.auxiliary_task(states, rewards).item() < expected_loss.item()


def test_disabled_auxiliary_preserves_policy_update(make_agent):
    agent = make_agent(auxiliary_coef=0.0)
    batch = _batch(agent)
    before = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}

    metrics = agent.learn(*batch)

    assert agent.actor.r is None
    assert "auxiliary_loss" not in metrics
    for name, param in agent.actor.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="auxiliary_coef > 0"):
        agent.auxiliary_task(batch[0], batch[5])


@pytest.mark.parametrize(
    "coefficient", [-0.1, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_auxiliary_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match="finite and non-negative"):
        PPO(env=None, auxiliary_coef=coefficient)


@pytest.mark.parametrize("reward_shape", [(3,), (2, 2), (1, 2)])
def test_reward_batch_mismatch_cannot_broadcast(make_agent, reward_shape):
    agent = make_agent()
    with pytest.raises(ValueError, match="rewards must have shape"):
        agent.auxiliary_task(torch.zeros(2, 3), torch.zeros(reward_shape))


@pytest.mark.parametrize("state_shape", [(3,), (0, 3), (2, 1, 3)])
def test_auxiliary_requires_nonempty_flat_batch(make_agent, state_shape):
    agent = make_agent()
    with pytest.raises(ValueError, match="non-empty"):
        agent.auxiliary_task(torch.zeros(state_shape), torch.zeros(2))


def test_actor_keeps_two_result_policy_api():
    actor = Actor(3, 1, hidden_dim=16, reward_prediction=True)
    actions, distribution = actor(torch.ones(2, 3))
    assert actions.shape == (2, 1)
    assert distribution.mean.shape == (2, 1)
    assert actor.predict_reward(torch.ones(2, 3)).shape == (2, 1)
    with pytest.raises(RuntimeError, match="reward_prediction=True"):
        Actor(3, 1).predict_reward(torch.ones(2, 3))


@pytest.mark.parametrize("vector", [False, True])
def test_training_updates_reward_head_and_logs_auxiliary_loss(make_agent, vector):
    env = _VectorRewardEnv() if vector else None
    agent = make_agent(env=env, normalize_obs=not vector, normalize_reward=True)
    before = agent.actor.r.weight.detach().clone()

    agent.train()
    agent.writer.flush()

    assert not torch.equal(agent.actor.r.weight, before)
    events = EventAccumulator(str(agent.log_dir)).Reload()
    records = events.Scalars(schema.PPO.LOSS_AUXILIARY)
    assert records and all(np.isfinite(record.value) for record in records)
    assert records[-1].step == agent.rollout_len * (2 if vector else 1)


@pytest.mark.parametrize("checkpoint", ["save", "best_sync", "best_async"])
def test_auxiliary_checkpoint_restores_predictions_and_training(
    make_agent, tmp_path, checkpoint
):
    agent = make_agent()
    batch = _batch(agent)
    agent.learn(*batch)
    if checkpoint == "save":
        model_dir = agent.save(tmp_path / "checkpoint")
    else:
        model_dir = tmp_path / "best"
        agent.save_best_model = True
        agent.best_model_dir = model_dir
        agent.save_best_async = checkpoint == "best_async"
        agent._save_best_checkpoint(eval_reward=1.0, episode=1)
        if agent._best_saver is not None:
            agent._best_saver.flush()
    restored = PPO.from_pretrained(str(model_dir))
    try:
        assert restored.auxiliary_coef == agent.auxiliary_coef
        torch.testing.assert_close(
            restored.actor.predict_reward(batch[0].to(restored.device)).cpu(),
            agent.actor.predict_reward(batch[0]),
        )
        if checkpoint == "best_sync":
            # Existing synchronous best checkpoints only persist network weights.
            before = restored.actor.r.weight.detach().clone()
            restored.learn(*batch)
            assert not torch.equal(restored.actor.r.weight, before)
        else:
            assert restored.a_opt.state_dict()["state"]
            # Restored optimizer state must produce the same next update.
            agent.learn(*batch)
            restored.learn(*batch)
            for expected, actual in zip(
                agent.actor.parameters(), restored.actor.parameters()
            ):
                torch.testing.assert_close(actual.cpu(), expected.cpu())
    finally:
        restored.close()
        restored.env.close()


def test_checkpoint_from_before_auxiliary_feature_still_loads(make_agent, tmp_path):
    agent = make_agent(auxiliary_coef=0.0)
    batch = _batch(agent)
    agent.learn(*batch)
    model_dir = agent.save(tmp_path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    del config["policy"]["params"]["auxiliary_coef"]
    config_path.write_text(json.dumps(config))
    assert set(agent.actor.state_dict()) == {
        f"{layer}.{parameter}"
        for layer in ("d1", "d2", "mu", "delta")
        for parameter in ("weight", "bias")
    }

    restored = PPO.from_pretrained(str(model_dir))
    try:
        assert restored.auxiliary_coef == 0.0
        assert restored.actor.r is None
        assert restored.a_opt.state_dict()["state"]
        for name, expected in agent.actor.state_dict().items():
            torch.testing.assert_close(
                restored.actor.state_dict()[name].cpu(), expected
            )
        restored.learn(*batch)
    finally:
        restored.close()
        restored.env.close()
