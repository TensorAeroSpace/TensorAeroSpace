import numpy as np
import torch

from tensoraerospace.agent.a2c.model import (
    A2C,
    Actor,
    Critic,
    Mish,
    discounted_rewards,
    mish,
    process_memory,
    t,
)


class _DummyEnv:
    def __init__(self):
        self.observation_space = type("S", (), {"shape": (3,)})
        self.action_space = type(
            "A",
            (),
            {"shape": (1,), "low": np.array([-1.0]), "high": np.array([1.0])},
        )

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def test_mish_and_module():
    x = torch.tensor([0.0, 1.0, -1.0])
    y = mish(x)
    mod = Mish()
    y2 = mod(x)
    assert torch.allclose(y, y2)


def test_t_and_discounted_rewards_and_process_memory():
    arr = [1, 2, 3]
    tt = t(arr)
    assert isinstance(tt, torch.Tensor)

    rewards = [1.0, 0.0, 1.0]
    dones = [0, 0, 1]
    disc = discounted_rewards(rewards, dones, gamma=0.9)
    assert len(disc) == 3

    memory = [
        (np.array([0.0, 0.0]), 1.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0),
        (np.array([0.0, 0.0]), 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1),
    ]
    actions, _rewards_t, _states, _next_states, _dones_t = process_memory(
        memory, gamma=0.9
    )
    assert actions.shape[0] >= 2


def test_a2c_learn_one_step():
    env = _DummyEnv()
    actor = Actor(state_dim=3, n_actions=1)
    critic = Critic(state_dim=3)
    agent = A2C(env=env, actor=actor, critic=critic)
    memory = agent.run_episode(max_steps=2)
    agent.learn(memory, steps=1, discount_rewards=True)
