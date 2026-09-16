"""Action-space and damaged-rotor limits must describe actual plant inputs."""

import gymnasium as gym
import numpy as np
import pytest

from tensoraerospace.aerospacemodel.quadrotor.damage import (
    DamageProfile,
    RotorDamageEvent,
    RotorLossEvent,
)
from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv
from tensoraerospace.envs.utils import ActionNormalizer


class AsymmetricEnv(gym.Env):
    action_space = gym.spaces.Box(
        np.array([-2.0, 1.0]), np.array([4.0, 3.0]), dtype=np.float64
    )
    observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float64)

    def step(self, action):
        return np.asarray(action).copy(), 0.0, False, False, {}


def test_normalizer_advertises_the_domain_it_accepts():
    original = AsymmetricEnv()
    wrapped = ActionNormalizer(original)
    np.testing.assert_array_equal(wrapped.action_space.low, [-1, -1])
    np.testing.assert_array_equal(wrapped.action_space.high, [1, 1])
    np.testing.assert_array_equal(original.action_space.low, [-2, 1])
    observation, *_ = wrapped.step(np.zeros(2))
    np.testing.assert_array_equal(observation, [1, 2])
    np.testing.assert_allclose(wrapped.reverse_action(observation), [0, 0])


@pytest.mark.parametrize("mode", ["virtual", "rotor"])
@pytest.mark.parametrize("failed", [False, True])
def test_rotor_damage_applies_after_command_saturation(mode, failed):
    event = (
        RotorLossEvent(trigger_time=0.0, rotor_id=0)
        if failed
        else RotorDamageEvent(trigger_time=0.0, rotor_id=0, mu=0.5)
    )
    env = NonlinearQuadrotorEnv(
        np.zeros(12),
        3,
        dt=0.001,
        action_space=mode,
        omega_min=100.0,
        omega_max=1000.0,
        damage_profile=DamageProfile(events=[event]),
    )
    try:
        env.reset()
        rotor_command = np.full(4, 2e6)
        action = rotor_command if mode == "rotor" else env.allocator.mix(rotor_command)
        _, _, _, _, info = env.step(action)
        expected = np.array([0.0 if failed else 0.5e6, 1e6, 1e6, 1e6])
        np.testing.assert_allclose(info["omega2_eff"], expected)
        np.testing.assert_allclose(
            env.model.u_history[-1].reshape(-1), env.allocator.mix(expected)
        )
    finally:
        env.close()
