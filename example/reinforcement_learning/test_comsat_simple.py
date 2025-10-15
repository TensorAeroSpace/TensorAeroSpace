"""Test simplified ComSat environment for easier PPO training."""

import numpy as np
from tensoraerospace.envs import ImprovedComSatEnv
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

print("=" * 80)
print("🧪 TESTING SIMPLIFIED COMSAT ENVIRONMENT")
print("=" * 80)

# Simpler configuration
dt = 0.01
_tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(_tp, dt=dt)
number_time_steps = len(_tp)

# CONSTANT reference signal (easier to learn)
target_angular_velocity = 0.001  # rad/s
reference_signals = np.ones((1, number_time_steps)) * target_angular_velocity

# Initial state matches reference
init_state = np.array([6371.0, 0.0, target_angular_velocity], dtype=np.float32)

env = ImprovedComSatEnv(
    initial_state=init_state,
    reference_signal=reference_signals,
    number_time_steps=number_time_steps,
    dt=dt,
)

print(f"\n✅ Configuration:")
print(f"  Reference: constant {target_angular_velocity} rad/s")
print(f"  Initial θ̇: {init_state[2]} rad/s")
print(f"  Initial error: {init_state[2] - target_angular_velocity:.6f}")
print()

# Test with simple proportional controller
print("Testing proportional controller (k=50):")
env.reset()
rewards = []
theta_dots = []

for i in range(100):
    theta_dot = env.state[2]
    theta_dots.append(theta_dot)
    
    # Proportional control
    error = theta_dot - target_angular_velocity
    control = -50 * error  # P-controller
    control = np.clip(control / 25.0, -1, 1)  # Normalize
    
    obs, reward, terminated, truncated, _ = env.step(np.array([control]))
    rewards.append(reward)
    
    if i < 10 or i % 20 == 0:
        print(f"  Step {i+1:3d}: θ̇={theta_dot:+.6f}, error={error:+.6f}, u={control*25:+6.2f}, reward={reward:+7.3f}")
    
    if terminated:
        print(f"\n⚠️ Terminated at step {i+1}")
        break

if not terminated:
    print(f"\n✅ Completed 100 steps!")

print(f"\n📊 Performance:")
print(f"  Total reward: {sum(rewards):.2f}")
print(f"  Mean reward: {np.mean(rewards):.3f}")
print(f"  Final θ̇: {theta_dots[-1]:.6f}")
print(f"  Final error: {theta_dots[-1] - target_angular_velocity:.6f}")

# Test with random policy
print("\n" + "=" * 80)
print("Testing RANDOM policy:")
env.reset()
rewards_random = []

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    rewards_random.append(reward)
    
    if terminated:
        print(f"  Terminated at step {i+1}")
        break

print(f"  Steps: {i+1}")
print(f"  Mean reward: {np.mean(rewards_random):.3f}")

print("\n" + "=" * 80)
print(f"💡 P-controller vs Random:")
print(f"  P-controller: {np.mean(rewards):.3f}")
print(f"  Random:       {np.mean(rewards_random):.3f}")
print(f"  Improvement:  {np.mean(rewards) - np.mean(rewards_random):.3f}")
print("=" * 80)

if np.mean(rewards) > np.mean(rewards_random) + 5:
    print("\n✅ Task is LEARNABLE! P-controller significantly better than random.")
    print("   PPO should be able to learn this!")
else:
    print("\n⚠️ Task might be too easy or rewards not well-tuned.")

