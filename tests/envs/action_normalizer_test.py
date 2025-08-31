from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.envs.utils import ActionNormalizer


class _DummyEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-2.0, -4.0], dtype=np.float32),
            high=np.array([2.0, 4.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.asarray(action, dtype=np.float32), 0.0, False, False, {}

    def render(self):
        return None


def test_action_normalizer_forward_and_reverse_mapping():
    env = _DummyEnv()
    wrapped = ActionNormalizer(env)

    # Input in normalized space (-1, 1)
    norm_action = np.array([-1.0, 1.0], dtype=np.float32)
    mapped = wrapped.action(norm_action.copy())

    # Expected mapping to (low, high)
    # scale = (high - low) / 2 = [2, 4], reloc = high - scale = [0, 0]
    # action = action * scale + reloc
    assert np.allclose(mapped, np.array([-2.0, 4.0], dtype=np.float32))

    # Reverse should bring it back to (-1, 1)
    reversed_action = wrapped.reverse_action(mapped.copy())
    assert np.allclose(reversed_action, norm_action)


def test_action_normalizer_clip():
    env = _DummyEnv()
    wrapped = ActionNormalizer(env)

    # Values outside (-1, 1) should map and then clip to bounds
    norm_action = np.array([-5.0, 5.0], dtype=np.float32)
    mapped = wrapped.action(norm_action.copy())
    box = cast(spaces.Box, env.action_space)
    assert np.all(mapped >= box.low)
    assert np.all(mapped <= box.high)
