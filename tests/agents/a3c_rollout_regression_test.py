"""A3C must pair each action with its own advantage and preserve rollout samples."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tensoraerospace.agent.a3c import pytorch as a3c
from tensoraerospace.agent.a3c.utils import push_and_pull


@pytest.mark.parametrize("actions", [1, 2])
def test_batch_loss_and_gradients_equal_the_mean_of_independent_samples(actions):
    torch.manual_seed(31)
    net = a3c.Net(2, actions)
    states = torch.tensor([[0.1, 0.5], [-0.4, 0.2], [1.0, -0.7]])
    controls = torch.tensor([[0.2], [-0.6], [1.1]]).expand(-1, actions)
    targets = torch.tensor([[2.0], [-1.5], [0.3]])
    batch = net.loss_func(states, controls, targets)
    individual = torch.stack(
        [
            net.loss_func(s[None], a[None], v[None])
            for s, a, v in zip(states, controls, targets)
        ]
    ).mean()
    torch.testing.assert_close(batch, individual)
    batch_grad = torch.autograd.grad(batch, tuple(net.parameters()))
    individual_grad = torch.autograd.grad(individual, tuple(net.parameters()))
    for actual, expected in zip(batch_grad, individual_grad):
        torch.testing.assert_close(actual, expected)


def test_scalar_action_and_target_vectors_keep_the_batch_axis():
    torch.manual_seed(7)
    net = a3c.Net(2, 1)
    states, actions, values = torch.randn(4, 2), torch.randn(4), torch.randn(4)
    actual = net.loss_func(states, actions, values)
    expected = torch.stack(
        [
            net.loss_func(s[None], a.reshape(1, 1), v.reshape(1, 1))
            for s, a, v in zip(states, actions, values)
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_push_gradient_and_logged_loss_match_the_per_transition_objective():
    torch.manual_seed(5)
    global_net, local_net = a3c.Net(2, 1), a3c.Net(2, 1)
    local_net.load_state_dict(global_net.state_dict())
    states = np.array([[0.1, 0.2], [-0.4, 0.7], [0.8, -0.2]], dtype=np.float32)
    actions = np.array([[0.2], [-0.6], [0.8]], dtype=np.float32)
    rewards = np.array([0.5, -0.3, 0.8], dtype=np.float32)
    # gamma=0 makes the return exactly the corresponding immediate reward.
    mu, sigma, values = local_net(torch.from_numpy(states))
    advantage = torch.from_numpy(rewards) - values[:, 0]
    logp = (
        torch.distributions.Normal(mu, sigma)
        .log_prob(torch.from_numpy(actions))
        .sum(-1)
    )
    entropy = torch.distributions.Normal(mu, sigma).entropy().sum(-1)
    actor = -(logp * advantage.detach() + 0.005 * entropy).mean()
    expected = actor + advantage.square().mean()
    gradients = torch.autograd.grad(expected, tuple(local_net.parameters()))
    metrics = push_and_pull(
        torch.optim.SGD(global_net.parameters(), lr=0),
        local_net,
        global_net,
        True,
        np.zeros(2),
        list(states),
        list(actions),
        list(rewards),
        0.0,
    )
    assert metrics["loss"] == pytest.approx(expected.item())
    assert metrics["policy_loss"] == pytest.approx(actor.item())
    for p, g in zip(global_net.parameters(), gradients):
        torch.testing.assert_close(p.grad, g)


class ReusedObservationEnv:
    action_space = type(
        "Space", (), {"low": np.array([-1.0]), "high": np.array([1.0])}
    )()

    def __init__(self, final_key=None, legacy=False):
        self.state = np.zeros(1, dtype=np.float32)
        self.final_key, self.legacy = final_key, legacy

    def reset(self):
        self.state[:] = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        self.state[:] = self.step_count
        done = self.step_count == 2
        info = {}
        if done and self.final_key:
            info[self.final_key] = self.state.copy()
            self.state[:] = -99  # autoreset observation
        if self.legacy:
            info["TimeLimit.truncated"] = done
            return self.state, 1.0, done, info
        return self.state, 1.0, False, done, info


def collect(monkeypatch, env, episodes=1, interval=10):
    batches = []

    def capture(opt, local, global_, terminal, next_, states, actions, rewards, gamma):
        batches.append((np.array(states), np.array(next_), terminal))
        return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)

    monkeypatch.setattr(a3c, "push_and_pull", capture)
    net = a3c.Net(1, 1)
    queue = mp.Queue()
    worker = a3c.Worker(
        env=env,
        gnet=net,
        opt=MagicMock(),
        global_ep=mp.Value("i", 0),
        global_ep_r=mp.Value("d", 0),
        res_queue=queue,
        name=0,
        num_actions=1,
        num_observations=1,
        MAX_EP=episodes,
        MAX_EP_STEP=3,
        GAMMA=0.99,
        update_global_iter=interval,
    )
    try:
        worker.run()
    finally:
        queue.close()
        queue.join_thread()
    return batches


def test_worker_snapshots_observations_before_the_environment_reuses_them(monkeypatch):
    batches = collect(monkeypatch, ReusedObservationEnv())
    np.testing.assert_array_equal(batches[0][0], [[0.0], [1.0]])


@pytest.mark.parametrize("key", ["final_observation", "terminal_observation"])
def test_worker_bootstraps_the_final_observation_in_autoreset_envs(monkeypatch, key):
    batches = collect(monkeypatch, ReusedObservationEnv(final_key=key))
    np.testing.assert_array_equal(batches[0][1], [2.0])
    assert batches[0][2] is False


def test_legacy_time_limit_keeps_the_bootstrap(monkeypatch):
    batches = collect(monkeypatch, ReusedObservationEnv(legacy=True))
    assert batches[0][2] is False


def test_update_interval_counts_episode_boundary_steps(monkeypatch):
    batches = collect(monkeypatch, ReusedObservationEnv(), episodes=2, interval=3)
    assert [len(batch[0]) for batch in batches] == [2, 1, 1]
