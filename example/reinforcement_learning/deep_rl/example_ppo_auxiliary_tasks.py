"""Train PPO with auxiliary reward prediction on the linear F-16 environment.

Run from the repository root:
    poetry run python example/reinforcement_learning/deep_rl/example_ppo_auxiliary_tasks.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tensoraerospace.agent.ppo.model import PPO
from tensoraerospace.envs.f16.linear_longitudinal import LinearLongitudinalF16


def run(episodes: int, steps: int, log_dir: Path) -> None:
    """Run a short reward-prediction demonstration, not a converged flight policy."""
    np.random.seed(42)
    torch.manual_seed(42)
    env = LinearLongitudinalF16(
        initial_state=np.zeros(2),
        reference_signal=np.full((1, steps + 2), np.deg2rad(1.0)),
        number_time_steps=steps + 2,
        tracking_states=["alpha"],
        state_space=["alpha", "q"],
        output_space=["alpha", "q"],
    )
    agent = PPO(
        env=env,
        auxiliary_coef=0.1,
        max_episodes=episodes,
        rollout_len=steps,
        num_epochs=2,
        batch_size=32,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_log_std_min=-3.0,
        actor_log_std_max=-1.0,
        device="cpu",
        log_dir=log_dir,
        save_best_model=False,
    )
    try:
        reward_head_before = agent.actor.r.weight.detach().clone()
        agent.train()
        delta = torch.linalg.vector_norm(
            agent.actor.r.weight.detach() - reward_head_before
        ).item()
        print(f"Reward head weight change: {delta:.6g}")
        print(f"Auxiliary MSE: TensorBoard tag loss/auxiliary in {log_dir}")
        saved = agent.save(log_dir / "checkpoints")
        print(f"Checkpoint (including auxiliary_coef and reward head): {saved}")
    finally:
        agent.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_auxiliary_f16"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 2:
        parser.error("--episodes must be >= 1 and --steps must be >= 2")
    run(args.episodes, args.steps, args.log_dir)
