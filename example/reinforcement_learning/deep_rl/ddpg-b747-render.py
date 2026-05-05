"""TensorAeroSpace usage example: evaluating a pretrained DDPG agent
for controlling longitudinal dynamics of Boeing 747.

- Build `ImprovedB747Env` environment with sinusoidal pitch reference.
- Load pretrained policy from Hugging Face or local folder via DDPG.from_pretrained.
- Run one episode in evaluation mode and (optionally) visualize.

Usage:
    python ddpg-b747-render.py --render --repo TensorAeroSpace/ddpg-b747 \
        --dt 0.1 --tn 200

Where:
    --render  enable step-by-step visualization (enabled by default)
    --repo    repository or local path to agent weights
    --dt      model discretization step (sec)
    --tn      number of episode timesteps
"""

import argparse

import numpy as np

from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import sinusoid_vertical_shift
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period


def build_env(dt: float, tn: int, render: bool = False) -> ImprovedB747Env:
    """Create B747 environment with sinusoidal pitch angle reference.

    Args:
        dt: discretization step in seconds
        tn: number of episode timesteps
        render: whether to enable human-mode rendering

    Returns:
        Initialized ImprovedB747Env environment
    """
    tp = generate_time_period(tn=tn, dt=dt)
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp)

    reference_signals = np.reshape(
        sinusoid_vertical_shift(
            tp=np.asarray(tps),
            frequency=0.05,
            amplitude=np.deg2rad(1.0),
            vertical_shift=0.0,
        ),
        [1, -1],
    )

    # Initial state: [u, w, q, theta]
    initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)

    env = ImprovedB747Env(
        initial_state=initial_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=True,
        dt=dt,
        render_mode="human" if render else None,
    )
    env.unwrapped.model.discretisation_time = dt
    return env


def evaluate_episode(agent: DDPG, env: ImprovedB747Env, render: bool = True) -> float:
    """Evaluate policy in a single episode.

    Args:
        agent: pretrained DDPG agent
        env: ImprovedB747Env environment
        render: whether to visualize steps

    Returns:
        Total reward for the episode
    """
    state, _info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        # DDPG policy is deterministic; no exploration during eval
        action = agent.policy_net.get_action(state)
        state, reward, terminated, truncated, _info = env.step(action)
        if render:
            env.render()
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1
    print(f"Episode finished: steps={steps}, return={total_reward:.3f}")
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPG B747 evaluation example")
    parser.add_argument(
        "--repo",
        type=str,
        default="TensorAeroSpace/ddpg-b747",
        help="Hugging Face repo id or local path to weights",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Discretization step, sec"
    )
    parser.add_argument(
        "--tn", type=int, default=200, help="Number of episode timesteps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = build_env(dt=args.dt, tn=args.tn, render=args.render)
    agent = DDPG.from_pretrained(args.repo)
    try:
        evaluate_episode(agent, env, render=args.render)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
