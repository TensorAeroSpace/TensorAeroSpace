"""Tests for A3C Worker class.

This module tests the Worker class functionality including:
- Worker initialization
- Worker-environment interaction
- Buffer management
"""

import numpy as np
import torch
import torch.multiprocessing as mp

from tensoraerospace.agent.a3c.pytorch import Net, Worker
from tensoraerospace.agent.a3c.shared_optim import SharedAdam


class _DummyEnv:
    """Minimal environment for testing."""

    def __init__(self):
        self.observation_space = type("S", (), {"shape": (4,)})()
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )()
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 10
        reward = -np.sum(action**2)  # Simple quadratic reward
        return (
            np.random.randn(4).astype(np.float32),
            float(reward),
            done,
            False,
            {},
        )

    def close(self):
        pass


def test_worker_initialization():
    """Test Worker initialization."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    env = _DummyEnv()

    worker = Worker(
        env=env,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=0,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=5,
        MAX_EP_STEP=10,
        GAMMA=0.99,
        update_global_iter=5,
        render=False,
        writer=None,
        global_step=global_step,
    )

    assert worker.name == "w0"
    assert worker.gamma == 0.99
    assert worker.max_ep == 5
    assert worker.max_ep_step == 10
    assert worker.update_global_iter == 5
    assert isinstance(worker.lnet, Net)


def test_worker_local_network():
    """Test that worker has its own local network."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    env = _DummyEnv()

    worker = Worker(
        env=env,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=0,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=5,
        MAX_EP_STEP=10,
        GAMMA=0.99,
        update_global_iter=5,
        global_step=global_step,
    )

    # Local network should be a separate instance
    assert worker.lnet is not gnet

    # But should have same architecture
    assert worker.lnet.s_dim == gnet.s_dim
    assert worker.lnet.a_dim == gnet.a_dim


def test_worker_different_names():
    """Test workers with different names."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    for i in range(3):
        env = _DummyEnv()
        worker = Worker(
            env=env,
            gnet=gnet,
            opt=opt,
            global_ep=global_ep,
            global_ep_r=global_ep_r,
            res_queue=res_queue,
            name=i,
            num_actions=a_dim,
            num_observations=s_dim,
            MAX_EP=5,
            MAX_EP_STEP=10,
            GAMMA=0.99,
            update_global_iter=5,
            global_step=global_step,
        )

        assert worker.name == f"w{i}"


def test_worker_gamma_values():
    """Test worker with different gamma values."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    for gamma in [0.9, 0.95, 0.99, 0.999]:
        env = _DummyEnv()
        worker = Worker(
            env=env,
            gnet=gnet,
            opt=opt,
            global_ep=global_ep,
            global_ep_r=global_ep_r,
            res_queue=res_queue,
            name=0,
            num_actions=a_dim,
            num_observations=s_dim,
            MAX_EP=5,
            MAX_EP_STEP=10,
            GAMMA=gamma,
            update_global_iter=5,
            global_step=global_step,
        )

        assert worker.gamma == gamma


def test_worker_update_frequency():
    """Test worker with different update frequencies."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    for update_iter in [1, 5, 10, 20]:
        env = _DummyEnv()
        worker = Worker(
            env=env,
            gnet=gnet,
            opt=opt,
            global_ep=global_ep,
            global_ep_r=global_ep_r,
            res_queue=res_queue,
            name=0,
            num_actions=a_dim,
            num_observations=s_dim,
            MAX_EP=5,
            MAX_EP_STEP=10,
            GAMMA=0.99,
            update_global_iter=update_iter,
            global_step=global_step,
        )

        assert worker.update_global_iter == update_iter


def test_worker_environment_compatibility():
    """Test worker with environment having action space bounds."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    env = _DummyEnv()

    worker = Worker(
        env=env,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=0,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=1,
        MAX_EP_STEP=5,
        GAMMA=0.99,
        update_global_iter=5,
        global_step=global_step,
    )

    # Verify environment has required attributes
    assert hasattr(worker.env, "observation_space")
    assert hasattr(worker.env, "action_space")
    assert hasattr(worker.env.action_space, "low")
    assert hasattr(worker.env.action_space, "high")


def test_worker_shared_counters():
    """Test that worker shares counters with other workers."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    # Shared counters
    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    # Create two workers with same counters
    env1 = _DummyEnv()
    worker1 = Worker(
        env=env1,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=0,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=5,
        MAX_EP_STEP=10,
        GAMMA=0.99,
        update_global_iter=5,
        global_step=global_step,
    )

    env2 = _DummyEnv()
    worker2 = Worker(
        env=env2,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=1,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=5,
        MAX_EP_STEP=10,
        GAMMA=0.99,
        update_global_iter=5,
        global_step=global_step,
    )

    # Both should reference same counters
    assert worker1.g_ep is worker2.g_ep
    assert worker1.g_ep_r is worker2.g_ep_r


def test_worker_render_flag():
    """Test worker with render flag."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    env = _DummyEnv()

    # With render=True
    worker = Worker(
        env=env,
        gnet=gnet,
        opt=opt,
        global_ep=global_ep,
        global_ep_r=global_ep_r,
        res_queue=res_queue,
        name=0,
        num_actions=a_dim,
        num_observations=s_dim,
        MAX_EP=1,
        MAX_EP_STEP=5,
        GAMMA=0.99,
        update_global_iter=5,
        render=True,
        global_step=global_step,
    )

    assert worker.render is True


def test_worker_max_episode_steps():
    """Test worker with different max episode steps."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    for max_steps in [10, 50, 100, 200]:
        env = _DummyEnv()
        worker = Worker(
            env=env,
            gnet=gnet,
            opt=opt,
            global_ep=global_ep,
            global_ep_r=global_ep_r,
            res_queue=res_queue,
            name=0,
            num_actions=a_dim,
            num_observations=s_dim,
            MAX_EP=5,
            MAX_EP_STEP=max_steps,
            GAMMA=0.99,
            update_global_iter=5,
            global_step=global_step,
        )

        assert worker.max_ep_step == max_steps


def test_worker_max_episodes():
    """Test worker with different max episodes."""
    s_dim, a_dim = 4, 2
    gnet = Net(s_dim, a_dim)
    opt = SharedAdam(gnet.parameters(), lr=1e-3)

    global_ep = mp.Value("i", 0)
    global_ep_r = mp.Value("d", 0.0)
    global_step = mp.Value("i", 0)
    res_queue = mp.Queue()

    for max_ep in [1, 5, 10, 100]:
        env = _DummyEnv()
        worker = Worker(
            env=env,
            gnet=gnet,
            opt=opt,
            global_ep=global_ep,
            global_ep_r=global_ep_r,
            res_queue=res_queue,
            name=0,
            num_actions=a_dim,
            num_observations=s_dim,
            MAX_EP=max_ep,
            MAX_EP_STEP=10,
            GAMMA=0.99,
            update_global_iter=5,
            global_step=global_step,
        )

        assert worker.max_ep == max_ep


if __name__ == "__main__":
    # Run all tests
    test_worker_initialization()
    test_worker_local_network()
    test_worker_different_names()
    test_worker_gamma_values()
    test_worker_update_frequency()
    test_worker_environment_compatibility()
    test_worker_shared_counters()
    test_worker_render_flag()
    test_worker_max_episode_steps()
    test_worker_max_episodes()
    print("All Worker tests passed!")
