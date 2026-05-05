"""PPO training example on ImprovedB747Env.

This script demonstrates training a PPO agent from
tensoraerospace/agent/ppo/model.py on the ImprovedB747Env from
tensoraerospace/envs/b747.py.

Features:
- Continuous actions with Gaussian policy
- Clipped surrogate objective for stable updates
- TensorBoard logging (losses, policy stats, reward)
- Checkpointing with optional gradient saving
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tensoraerospace.agent.ppo.model import PPO
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standard import sinusoid_vertical_shift
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

# Device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("Using device:", DEVICE)

# Reproducibility
np.random.seed(1)
_ = torch.manual_seed(1)

# =========================================
# Environment Setup
# =========================================
print("\n" + "=" * 60)
print("ENVIRONMENT SETUP")
print("=" * 60)

# Time base
dt = 0.1
_tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(_tp, dt=dt)
number_time_steps = len(_tp)

# Reference theta (radians)
reference_signals = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps),
        frequency=0.05,
        amplitude=np.deg2rad(1.0),
        vertical_shift=0.0,
    ),
    [1, -1],
)

# Initial state [u, w, q, theta] (SI units; angles in rad)
init_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Create environment
env = ImprovedB747Env(
    initial_state=init_state,
    reference_signal=reference_signals,
    number_time_steps=number_time_steps,
    dt=dt,
    initial_elevator_deg=0.0,
)

obs, info = env.reset()
print(f"Observation shape: {np.array(obs).shape}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# =========================================
# PPO Agent Creation
# =========================================
print("\n" + "=" * 60)
print("PPO AGENT CREATION")
print("=" * 60)

# PPO hyperparameters tuned for ImprovedB747Env
# Tips for aircraft pitch control:
# - Higher gamma (0.99) helps with long-horizon tracking
# - Larger rollout_len (2048) collects more diverse trajectories
# - Moderate clip_param (0.2) balances exploration and stability
# - Lower entropy_coef reduces randomness once policy converges

agent = PPO(
    env=env,
    gamma=0.99,  # Discount factor
    max_episodes=50,  # Number of training episodes
    rollout_len=2048,  # Steps per episode before update
    clip_pram=0.2,  # PPO clip parameter
    num_epochs=10,  # Optimization epochs per rollout
    batch_size=64,  # Mini-batch size for SGD
    entropy_coef=0.01,  # Entropy bonus coefficient
    actor_lr=3e-4,  # Actor learning rate
    critic_lr=1e-3,  # Critic learning rate
    seed=336699,
    device=DEVICE,
)

print(f"Actor parameters: {sum(p.numel() for p in agent.actor.parameters())}")
print(f"Critic parameters: {sum(p.numel() for p in agent.critic.parameters())}")
print(f"Actor learning rate: {agent.actor_lr}")
print(f"Critic learning rate: {agent.critic_lr}")

# =========================================
# Quick Training Run
# =========================================
print("\n" + "=" * 60)
print("STARTING PPO TRAINING")
print("=" * 60)

# Optional: Resume from checkpoint
ckpt_path = os.path.join("runs", "ppo_b747_improved", "ppo_checkpoint.pt")
if os.path.isfile(ckpt_path):
    try:
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.a_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        agent.c_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print("✓ Checkpoint loaded successfully")
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"⚠ Failed to load checkpoint: {e}")

# Create log directory
log_root = os.path.join("runs", "ppo_b747_improved")
os.makedirs(log_root, exist_ok=True)

# Train agent
try:
    agent.train()
    print("\n✓ Training completed successfully!")
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
except Exception as e:
    print(f"\n✗ Training failed with error: {e}")
    raise
finally:
    # Save checkpoint
    ckpt_out = os.path.join(log_root, "ppo_checkpoint.pt")
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "actor_optimizer_state_dict": agent.a_opt.state_dict(),
            "critic_optimizer_state_dict": agent.c_opt.state_dict(),
            "gamma": agent.gamma,
            "clip_param": agent.clip_pram,
        },
        ckpt_out,
    )
    print(f"\n✓ Checkpoint saved to: {ckpt_out}")
    agent.writer.close()

# =========================================
# Evaluation and Visualization
# =========================================
print("\n" + "=" * 60)
print("EVALUATION & VISUALIZATION")
print("=" * 60)

# Run one final episode for visualization
obs, info = env.reset()
done = False
episode_reward = 0
step_count = 0

print("Running final evaluation episode...")
while not done and step_count < number_time_steps:
    action, mean_action, _ = agent.act(obs)
    step_return = env.step(mean_action[0])

    if len(step_return) == 5:
        obs, reward, terminated, truncated, info = step_return
        done = terminated or truncated
    else:
        obs, reward, terminated, info = step_return[:4]
        done = terminated

    episode_reward += float(reward)
    step_count += 1

print(f"Final episode reward: {episode_reward:.4f}")
print(f"Episode length: {step_count} steps")

# Plot results
print("\nGenerating plots...")

# Plot theta (pitch angle) tracking
plt.figure(figsize=(15, 4))
env.unwrapped.model.plot_transient_process(
    "theta", tps, reference_signals[0], to_deg=True, figsize=(15, 4)
)
theta_path = os.path.join(log_root, "theta_tracking.png")
plt.savefig(theta_path, dpi=150, bbox_inches="tight")
print("✓ Saved theta tracking plot")

# Plot pitch rate (q)
plt.figure(figsize=(15, 4))
env.unwrapped.model.plot_state(state_name="q", time=tps, to_deg=True, figsize=(15, 4))
pitch_path = os.path.join(log_root, "pitch_rate.png")
plt.savefig(pitch_path, dpi=150, bbox_inches="tight")
print("✓ Saved pitch rate plot")

# Plot elevator control
plt.figure(figsize=(15, 4))
env.unwrapped.model.plot_control(
    control_name="ele", time=tps, to_deg=True, figsize=(15, 4)
)
elevator_path = os.path.join(log_root, "elevator_control.png")
plt.savefig(elevator_path, dpi=150, bbox_inches="tight")
print("✓ Saved elevator control plot")

plt.close("all")

# =========================================
# TensorBoard Logging Info
# =========================================
print("\n" + "=" * 60)
print("TENSORBOARD")
print("=" * 60)
print("To view training logs, run:")
print(f"  tensorboard --logdir={log_root}")
print("=" * 60)

print("\n✓ Script completed successfully!")
