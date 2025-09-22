import argparse
import pathlib
from typing import List

import gymnasium as gym
import numpy as np


def compute_torque_pd(
    observation: np.ndarray, kp: float, kd: float, max_torque: float
) -> float:
    """
    Simple PD controller for Pendulum-v1.

    observation = [cos(theta), sin(theta), theta_dot]
    torque = clamp(-kp * theta - kd * theta_dot, [-max_torque, max_torque])
    """
    cos_theta, sin_theta, theta_dot = (
        float(observation[0]),
        float(observation[1]),
        float(observation[2]),
    )
    theta = float(np.arctan2(sin_theta, cos_theta))
    torque_raw: float = -kp * theta - kd * theta_dot
    torque_clipped: float = float(np.clip(torque_raw, -max_torque, max_torque))
    return torque_clipped


def generate_expert_dataset(
    env_name: str,
    num_episodes: int,
    max_steps: int,
    kp: float,
    kd: float,
    action_noise_std: float,
    seed: int,
) -> np.ndarray:
    """
    Collects expert state-action pairs for GAIL training.

    Returns array of shape [N, obs_dim + act_dim]. For Pendulum-v1: [N, 4].
    """
    env = gym.make(env_name)

    rng = np.random.default_rng(seed)
    expert_rows: List[np.ndarray] = []

    # Get action limits (assume symmetric Box)
    if hasattr(env.action_space, "high"):
        max_torque = float(np.asarray(env.action_space.high, dtype=np.float32)[0])
    else:
        max_torque = 2.0

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        for _ in range(max_steps):
            torque = compute_torque_pd(obs, kp=kp, kd=kd, max_torque=max_torque)
            if action_noise_std > 0.0:
                torque = float(
                    np.clip(
                        torque + rng.normal(0.0, action_noise_std),
                        -max_torque,
                        max_torque,
                    )
                )

            action = np.array([torque], dtype=np.float32)
            expert_rows.append(np.hstack([obs.astype(np.float32), action]))

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()
    return np.asarray(expert_rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate expert state-action pairs for GAIL (Pendulum-v1)")
    )
    parser.add_argument(
        "--env", type=str, default="Pendulum-v1", help="Gymnasium env id"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of expert episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode"
    )
    parser.add_argument("--kp", type=float, default=6.0, help="PD proportional gain")
    parser.add_argument("--kd", type=float, default=1.0, help="PD derivative gain")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Action noise std for data diversity",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out", type=str, default="data_pendulum_expert.npy", help="Output .npy path"
    )

    args = parser.parse_args()

    data = generate_expert_dataset(
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        kp=args.kp,
        kd=args.kd,
        action_noise_std=args.noise_std,
        seed=args.seed,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data)
    print(f"Saved expert dataset: {out_path} with shape {data.shape}")


if __name__ == "__main__":
    main()
