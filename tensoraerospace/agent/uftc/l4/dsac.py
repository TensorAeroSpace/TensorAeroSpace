"""Distributional SAC outer-loop planner — placeholder skeleton."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSACConfig:
    n_state: int
    n_ref_dim: int
    n_action: int
    cvar_alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    learn_every: int = 1
    update_to_data_ratio: int = 1
    target_entropy: float | None = None
    glr_reset_threshold: float = 0.10
    eval_mode: bool = True
    n_quantiles: int = 32
    huber_kappa: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    seed: int = 0
