"""TensorAeroSpace usage example: evaluating a pretrained SAC agent
for controlling longitudinal dynamics of Boeing 747.

- Build `ImprovedB747Env` environment with sinusoidal pitch reference.
- Load pretrained policy from Hugging Face: "TensorAeroSpace/sac-b747".
- Run one episode in evaluation mode and (optionally) visualize.

Usage:
    python sac-b747-render.py --render --repo TensorAeroSpace/sac-b747 \
        --dt 0.1 --tn 200

Where:
    --render  enable step-by-step visualization (enabled by default)
    --repo    repository or local path to agent weights
    --dt      model discretization step (sec)
    --tn      number of episode timesteps
"""

import argparse
import os
from contextlib import suppress
from typing import List, Optional

import numpy as np
import torch

from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Auto-detect and return the best available device.

    Args:
        device_str: Optional device string ('cuda', 'mps', 'cpu').
            If None, auto-detects based on availability.

    Returns:
        torch.device: The selected device.
    """
    if device_str is not None:
        return torch.device(device_str)

    # Auto-detect: prefer CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_theta_value(state: np.ndarray) -> float:
    """Return theta component from environment state."""
    arr = np.asarray(state, dtype=float)
    if arr.ndim == 2 and arr.shape[0] >= 4:
        return float(arr[3, 0])
    flat = arr.reshape(-1)
    idx = min(3, flat.size - 1)
    return float(flat[idx])


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
    # Alternative step reference:
    # reference_signals = np.reshape(
    #     unit_step(
    #         degree=1, tp=np.asarray(tps), time_step=5, output_rad=True
    #     ),
    #     [1, -1]
    # )

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


def evaluate_episode(agent: SAC, env: ImprovedB747Env, render: bool = True) -> float:
    """Evaluate policy in a single episode.

    Args:
        agent: pretrained SAC agent
        env: ImprovedB747Env environment
        render: whether to visualize steps

    Returns:
        Total reward for the episode
    """
    state, _info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    tracked_theta: List[float] = []
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _info = env.step(action)
        if render:
            env.render()
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1
        tracked_theta.append(extract_theta_value(state))
    print(f"Episode finished: steps={steps}, return={total_reward:.3f}")
    if tracked_theta and hasattr(env.unwrapped, "reference_signal"):
        reference = np.asarray(env.unwrapped.reference_signal)
        if reference.ndim == 2:
            ref_track = reference[0]
            idx = min(len(ref_track) - 1, len(tracked_theta) - 1)
            static_error = float(ref_track[idx]) - tracked_theta[-1]
            print(
                "Static error (theta_ref - theta_actual): "
                f"{static_error:.6f} rad / "
                f"{np.rad2deg(static_error):.4f} deg"
            )
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC B747 evaluation example")
    parser.add_argument(
        "--repo",
        type=str,
        default="TensorAeroSpace/sac-b747",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to use ('cuda', 'mps', 'cpu'). " "Auto-detects if not specified."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Auto-detect device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check GPU availability if CUDA requested
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. " "Falling back to CPU.")
            device = torch.device("cpu")
        else:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"CUDA memory: {total_memory_gb:.2f} GB")

    # Build environment
    env = build_env(dt=args.dt, tn=args.tn, render=args.render)

    # Load pretrained agent
    print(f"Loading pretrained agent from: {args.repo}")
    agent = SAC.from_pretrained(args.repo)

    # Move agent to the selected device
    # Note: from_pretrained loads model with saved device,
    # so we need to move it
    if agent.device != device:
        print(f"Moving agent from {agent.device} to {device}")
        agent.device = device
        agent.critic = agent.critic.to(device)
        agent.critic_target = agent.critic_target.to(device)
        agent.policy = agent.policy.to(device)
        # Move log_alpha if it exists (for automatic entropy tuning)
        if hasattr(agent, "log_alpha") and agent.log_alpha is not None:
            agent.log_alpha = agent.log_alpha.to(device)

    print(f"Agent device: {agent.device}")
    print(f"Policy device: {next(agent.policy.parameters()).device}")
    print(f"Critic device: {next(agent.critic.parameters()).device}")

    # Set display for rendering (if needed)
    if args.render:
        # Check if DISPLAY is set (for X11)
        if "DISPLAY" not in os.environ:
            print(
                "Warning: DISPLAY not set. Rendering may not work on headless "
                "systems."
            )
            print("Consider using --no-render or setting up virtual display " "(xvfb).")

    try:
        evaluate_episode(agent, env, render=args.render)
    finally:
        with suppress(Exception):
            env.close()


if __name__ == "__main__":
    main()
