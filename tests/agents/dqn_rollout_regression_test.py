"""DQN replay ownership, target initialization and continued learning contracts."""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.dqn import model as dqn


class SharedEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, final_key=None, terminated=False):
        self.obs = np.zeros(1, np.float32)
        self.final_key = final_key
        self.terminated = terminated

    def reset(self, **kwargs):
        self.obs[:] = 1
        self.steps = 0
        return self.obs, {}

    def step(self, action):
        self.steps += 1
        self.obs[:] = self.steps + 1
        done = self.steps == 2
        info = {}
        if done and self.final_key:
            info[self.final_key] = self.obs.copy()
            self.obs[:] = -10
        return (
            self.obs,
            float(action),
            done and self.terminated,
            done and not self.terminated,
            info,
        )


@pytest.fixture(params=[dqn.DQNAgent, dqn.PERNARXAgent])
def make_agent(request, monkeypatch, tmp_path):
    monkeypatch.setattr(dqn, "_DEVICE", torch.device("cpu"))
    agents = []

    def make(env=None, **kwargs):
        torch.manual_seed(29)
        np.random.seed(29)
        options = dict(
            buffer_size=4,
            batch_size=2,
            train_nums=2,
            epsilon=0,
            min_epsilon=0,
            target_update_iter=3,
            log_dir=str(tmp_path / str(len(agents))),
        )
        options.update(kwargs)
        agent = request.param(dqn.Model(2), dqn.Model(2), env or SharedEnv(), **options)
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()


def test_target_initially_matches_online_and_is_independent(make_agent):
    agent = make_agent()
    obs = np.array([[1.0], [-2.0]], np.float32)
    np.testing.assert_array_equal(
        agent.model.predict(obs), agent.target_model.predict(obs)
    )
    for left, right in zip(agent.model.parameters(), agent.target_model.parameters()):
        assert left.data_ptr() != right.data_ptr()


def test_replay_owns_copies_of_both_observations(make_agent):
    agent = make_agent()
    obs, next_obs = np.array([1.0]), np.array([2.0])
    agent.store_transition(1, obs, 0, 1.0, next_obs, False)
    obs[:] = next_obs[:] = -99
    row = agent.replay_buffer.transitions[0]
    np.testing.assert_array_equal(row[0], [1])
    np.testing.assert_array_equal(row[3], [2])


@pytest.mark.parametrize(
    "final_key", [None, "final_observation", "terminal_observation"]
)
@pytest.mark.parametrize("terminated", [False, True])
def test_collector_preserves_shared_and_final_observations(
    make_agent, final_key, terminated
):
    agent = make_agent(SharedEnv(final_key, terminated))
    agent.train(verbose=False)
    first, second = agent.replay_buffer.transitions[:2]
    np.testing.assert_array_equal(first[0], [1])
    np.testing.assert_array_equal(first[3], [2])
    np.testing.assert_array_equal(second[0], [2])
    np.testing.assert_array_equal(second[3], [3])
    assert bool(second[4]) is terminated


def test_chunked_training_matches_continuous_updates(make_agent):
    def train(parts):
        agent = make_agent(epsilon_dacay=1)
        agent.env.action_space.seed(29)
        for frames in parts:
            agent.train(max_steps=frames, verbose=False)
        return agent

    whole = train([12])
    chunked = train([6, 6])
    assert whole.global_env_step == chunked.global_env_step == 12
    assert whole.global_step == chunked.global_step == 8
    for name in ["model", "target_model"]:
        for key, value in getattr(whole, name).state_dict().items():
            torch.testing.assert_close(
                value, getattr(chunked, name).state_dict()[key], rtol=0, atol=0
            )


def test_explicit_exploration_decay_preserves_floor(make_agent):
    agent = make_agent(epsilon=0.8, epsilon_dacay=0.5, min_epsilon=0.2)
    agent.train(max_steps=8, verbose=False)
    assert agent.epsilon == pytest.approx(0.8)
    for _ in range(10):
        agent.e_decay()
    assert agent.epsilon == pytest.approx(0.2)


def test_single_slot_sum_tree_handles_root_and_updates():
    tree = dqn.SumTree(1)
    tree.add(2, "first")
    assert tree.get_leaf(1) == (0, 2, "first")
    tree.add(3, "second")
    assert tree.total_p == 3
    assert tree.get_leaf(0) == (0, 3, "second")


def test_sampling_left_boundary_skips_zero_priority_leaf():
    tree = dqn.SumTree(4)
    tree.add(0, "unavailable")
    tree.add(2, "available")
    assert tree.get_leaf(0)[2] == "available"


def test_checkpoint_preserves_lagged_target_and_counters(tmp_path, monkeypatch):
    monkeypatch.setattr(dqn, "_DEVICE", torch.device("cpu"))
    agent = dqn.DQNAgent(
        dqn.Model(2),
        dqn.Model(2),
        SharedEnv(),
        buffer_size=4,
        batch_size=2,
        epsilon=0,
        target_update_iter=20,
        log_dir=str(tmp_path / "train"),
    )
    try:
        agent.train(max_steps=8, verbose=False)
        path = tmp_path / "checkpoint"
        agent.save(path, save_gradients=True)
        loaded = dqn.DQNAgent.load(path, SharedEnv(), load_gradients=True)
        try:
            assert loaded.global_env_step == agent.global_env_step == 8
            assert loaded.global_step == agent.global_step == 4
            assert loaded.episode_idx == agent.episode_idx == 4
            for name in ["model", "target_model"]:
                for key, value in getattr(agent, name).state_dict().items():
                    torch.testing.assert_close(
                        value, getattr(loaded, name).state_dict()[key], rtol=0, atol=0
                    )
            config = json.loads((path / "config.json").read_text())
            config.pop("training")
            (path / "config.json").write_text(json.dumps(config))
            legacy = dqn.DQNAgent.load(path, SharedEnv())
            assert (
                legacy.global_env_step == legacy.global_step == legacy.episode_idx == 0
            )
            legacy.close()
        finally:
            loaded.close()
    finally:
        agent.close()
