"""Example usage of ImprovedX15Env for reinforcement learning.

This example demonstrates how to use the ImprovedX15Env environment
with normalized action/observation spaces and enhanced reward function
for training RL agents.
"""

import numpy as np

from tensoraerospace.envs import ImprovedX15Env
from tensoraerospace.signals import unit_step

# Simulation parameters
dt = 0.01  # Time step in seconds
duration = 10.0  # Total simulation duration in seconds
num_steps = int(duration / dt)

# Initial state [u, w, q, theta] in SI units
# u: longitudinal velocity (m/s)
# w: normal velocity (m/s)
# q: pitch rate (rad/s)
# theta: pitch angle (rad)
initial_state = np.array([600.0, 0.0, 0.0, 0.0])

# Reference signal: step command for pitch angle
# Target: 10 degrees pitch
target_pitch_deg = 10.0
time_array = np.arange(0, duration, dt)
reference_signal = unit_step(
    tp=time_array,
    degree=target_pitch_deg,
    time_step=1.0,  # Step starts at 1 second
    dt=dt,
    output_rad=True,  # Output in radians
).reshape(1, -1)

# Create environment
env = ImprovedX15Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=num_steps,
    dt=dt,
    initial_elevator_deg=0.0,  # Initial elevator position
    use_initial_action_on_first_step=True,
)

# Print environment info
print("=" * 60)
print("ImprovedX15Env - North American X-15 Pitch Control")
print("=" * 60)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Max pitch angle: ±{np.rad2deg(env.max_pitch_rad):.1f} deg")
print(f"Max pitch rate: ±{np.rad2deg(env.max_pitch_rate_rad_s):.1f} deg/s")
print(f"Max elevator: ±{env.max_elevator_angle_deg:.1f} deg")
print("=" * 60)

# Reset environment
obs, info = env.reset()
print(f"\nInitial observation (normalized): {obs.flatten()}")
print("  [pitch_error, pitch_rate, pitch_angle, prev_action]")

# Simple PD controller for demonstration
# (In practice, you would replace this with an RL agent)
Kp = 0.5  # Proportional gain
Kd = 0.2  # Derivative gain

total_reward = 0
done = False
truncated = False
step_count = 0

print("\nRunning simulation with simple PD controller...")
print("(Close the render window to continue)")

while not done and not truncated:
    # Extract normalized pitch error and pitch rate from observation
    pitch_error_norm = obs[0, 0]
    pitch_rate_norm = obs[1, 0]

    # PD control law (normalized action)
    action = np.clip(Kp * pitch_error_norm - Kd * pitch_rate_norm, -1.0, 1.0).reshape(1)

    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    # Render every 10 steps to speed up visualization
    if step_count % 10 == 0:
        env.render()

    # Print progress every 100 steps
    if step_count % 100 == 0:
        theta_actual = obs[2, 0] * np.rad2deg(env.max_pitch_rad)
        print(
            f"Step {step_count:4d}: Pitch = {theta_actual:+6.2f} deg, "
            f"Reward = {reward:+.4f}"
        )

# Close environment
env.close()

print("\n" + "=" * 60)
print("Simulation completed!")
print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.2f}")
print(f"Average reward: {total_reward / step_count:.4f}")
print("=" * 60)

# Note: To train an RL agent with this environment, you can use
# libraries like Stable Baselines3, Ray RLlib, or custom implementations.
# Example with Stable Baselines3:
#
# from stable_baselines3 import PPO
#
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("ppo_x15")
