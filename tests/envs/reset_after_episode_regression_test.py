"""A completed plant simulation must start a fresh episode after reset."""

import numpy as np
import pytest

from tensoraerospace.envs.f4c import LinearLongitudinalF4C
from tensoraerospace.envs.geosat import GeoSatEnv
from tensoraerospace.envs.uav import LinearLongitudinalUAV
from tensoraerospace.envs.x15 import LinearLongitudinalX15


@pytest.mark.parametrize(
    "env_cls,n_states",
    [
        (LinearLongitudinalF4C, 4),
        (GeoSatEnv, 3),
        (LinearLongitudinalUAV, 4),
        (LinearLongitudinalX15, 4),
    ],
)
def test_reset_after_completed_episode_replays_the_same_trajectory(env_cls, n_states):
    env = env_cls(np.zeros(n_states), np.zeros((1, 4)), number_time_steps=4)
    action = np.array([0.1], dtype=np.float32)
    try:
        initial, _ = env.reset(seed=17)
        first_episode = [env.step(action) for _ in range(3)]
        expected_boundary = (
            (False, True)
            if env_cls in (GeoSatEnv, LinearLongitudinalUAV, LinearLongitudinalF4C)
            else (True, False)
        )
        assert first_episode[-1][2:4] == expected_boundary
        assert env.done is True
        restored, _ = env.reset(seed=17)
        assert env.done is False
        assert env.current_step == env.model.time_step == 0
        np.testing.assert_array_equal(restored, initial)
        for expected in first_episode:
            actual = env.step(action)
            np.testing.assert_allclose(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]
    finally:
        env.close()
