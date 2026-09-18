"""A3C workers must bootstrap time limits while still finishing the episode."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch.multiprocessing as mp

from tensoraerospace.agent.a3c import pytorch as a3c
from tensoraerospace.agent.a3c.shared_optim import SharedAdam
from tensoraerospace.agent.metrics.writer import MetricWriter


@pytest.mark.parametrize(
    "terminated,truncated", [(True, False), (False, True), (False, False)]
)
def test_worker_bootstraps_environment_and_trainer_time_limits(
    monkeypatch, terminated, truncated
):
    class Env:
        action_space = type(
            "Space", (), {"low": np.array([-1.0]), "high": np.array([1.0])}
        )()

        def reset(self, **kwargs):
            return np.array([1.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([2.0], dtype=np.float32), 1.0, terminated, truncated, {}

        def close(self):
            pass

    network = a3c.Net(1, 1)
    captured = []
    original = a3c.push_and_pull

    def update(opt, lnet, gnet, terminal, next_state, states, actions, rewards, gamma):
        captured.append(terminal)
        return original(
            opt, lnet, gnet, terminal, next_state, states, actions, rewards, gamma
        )

    monkeypatch.setattr(a3c, "push_and_pull", update)
    writer = MagicMock(spec=MetricWriter)
    queue = mp.Queue()
    worker = a3c.Worker(
        env=Env(),
        gnet=network,
        opt=SharedAdam(network.parameters(), lr=1e-3),
        global_ep=mp.Value("i", 0),
        global_ep_r=mp.Value("d", 0.0),
        res_queue=queue,
        name=0,
        num_actions=1,
        num_observations=1,
        MAX_EP=1,
        MAX_EP_STEP=1,
        GAMMA=0.9,
        update_global_iter=10,
        writer=writer,
        global_env_step=mp.Value("i", 0),
    )
    try:
        worker.run()
        assert captured == [terminated]
    finally:
        queue.close()
        queue.join_thread()
