"""Scalar and vector B747 must represent the same bounded physical actuator."""

import numpy as np
import pytest
import torch

from tensoraerospace.aerospacemodel.b747.linear.longitudinal import LongitudinalB747
from tensoraerospace.envs.b747_vec_torch import ImprovedB747VecEnvTorch


@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
def test_vector_and_scalar_b747_match_with_saturated_reversing_commands(dt):
    commands = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 0.0, -1.0, 1.0])
    scalar = LongitudinalB747(np.zeros(4), len(commands), dt=dt)
    vector = ImprovedB747VecEnvTorch(
        num_envs=2, dt=dt, tn=10.0, device="cpu", auto_reset=False
    )
    vector.reset()
    previous = 0.0
    for command in commands:
        expected = scalar.run_step(np.array([np.deg2rad(25.0) * command])).reshape(-1)
        vector.step(torch.full((2, 1), float(command)))
        np.testing.assert_allclose(
            vector.state[0].numpy(), expected, atol=2e-6, rtol=2e-5
        )
        applied = float(vector.prev_u_rad[0, 0])
        assert abs(applied - previous) <= np.deg2rad(60.0) * dt + 1e-7
        previous = applied


def test_scalar_first_command_and_restart_respect_rate_limit_from_neutral():
    model = LongitudinalB747(np.zeros(4), 3, dt=0.01)
    for _ in range(2):
        model.run_step(np.array([np.deg2rad(25.0)]))
        assert model.store_input[0, 0] == pytest.approx(np.deg2rad(0.6))
        model.restart()
