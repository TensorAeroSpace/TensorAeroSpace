import gymnasium as gym
import numpy as np

from tensoraerospace.envs.unity_env import unity_discrete_env


class _DummyUnityEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self._closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        # Echo back the action for verification
        return np.asarray(action, dtype=np.float32), 0.0, False, False, {}

    def render(self):
        return None

    def close(self):
        self._closed = True


def test_unity_discrete_env_step_mapping_all_minus_one():
    env = unity_discrete_env(_DummyUnityEnv())
    obs, _ = env.reset()
    assert obs.shape == (7,)

    # action = 0 -> all digits 0 -> mapped to -1.0
    next_obs, _, _, _, _ = env.step(0)
    assert np.allclose(next_obs, -1.0)


def test_unity_discrete_env_step_mapping_mixed_values():
    env = unity_discrete_env(_DummyUnityEnv())
    # Construct an action whose base-3 digits are [0,1,2,0,1,2,0]
    base3_digits = [0, 1, 2, 0, 1, 2, 0]
    action = sum(d * (3**i) for i, d in enumerate(base3_digits))
    # Expected mapping: digit -> {-1, 0, 1}
    expected = np.array([d - 1.0 for d in base3_digits], dtype=np.float32)

    next_obs, _, _, _, _ = env.step(action)
    assert np.allclose(next_obs, expected)


def test_unity_discrete_env_close_passthrough():
    base = _DummyUnityEnv()
    env = unity_discrete_env(base)
    env.close()
    assert base._closed is True
