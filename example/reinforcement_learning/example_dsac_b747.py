"""Minimal DSAC example on the B747 longitudinal env.

This follows the DSAC design from HybridRL-FlightControl (IQN critics + CAPS),
but uses the TensorAeroSpace DSAC agent and the normalized ImprovedB747Env.

Run:
    python example_dsac_b747.py
"""

from __future__ import annotations

import numpy as np
import torch

from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env


def make_reference(step_count: int, step_deg: float = 5.0) -> np.ndarray:
    """Create a simple step reference in radians, shape (1, T)."""
    ref = np.zeros((1, step_count), dtype=np.float32)
    ref[:, step_count // 5 :] = np.deg2rad(step_deg)
    return ref


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment parameters
    num_steps = 800
    env = ImprovedB747Env(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        reference_signal=make_reference(num_steps, step_deg=5.0),
        number_time_steps=num_steps,
        dt=0.02,
        reward_mode="step_response",
    )

    agent = DSAC(
        env,
        batch_size=128,
        memory_capacity=200_000,
        learning_starts=1_000,
        updates_per_step=1,
        num_quantiles=32,
        embedding_dim=32,
        hidden_layers=[64, 64],
        huber_threshold=1.0,
        lr=3e-4,
        policy_lr=3e-4,
        device=device,
        log_every_updates=50,
        automatic_entropy_tuning=True,
    )

    # Train a few episodes (adjust as needed)
    agent.train(num_episodes=5, save_best=False)

    agent.close()


if __name__ == "__main__":
    main()

