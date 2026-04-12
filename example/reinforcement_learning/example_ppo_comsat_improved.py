"""Example: PPO training for improved ComSat environment.

This script demonstrates training a PPO agent to control a communication
satellite's angular velocity using tangential thrust.

The environment features:
- Normalized observation and action spaces
- LQR-style reward function
- Multiple tracking objectives:
  * Angular velocity tracking (primary)
  * Orbital radius stabilization
  * Energy efficiency
  * Control smoothness
"""

import numpy as np

from tensoraerospace.envs import ImprovedComSatEnv
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period


def create_comsat_env(dt=0.01, total_time=20.0):
    """Create improved ComSat environment with reference signal.

    Args:
        dt (float): Time step in seconds. Defaults to 0.01.
        total_time (float): Total simulation time in seconds. Defaults to 20.0.

    Returns:
        ImprovedComSatEnv: Configured environment instance.
    """
    # Generate time period
    tp = generate_time_period(tn=total_time, dt=dt)
    number_time_steps = len(tp)

    # Reference signal: step change in angular velocity at t=10s
    # Target: change from 0.001 rad/s to 0.002 rad/s
    reference_signals = unit_step(
        degree=0.002,  # Target angular velocity
        tp=tp,
        time_step=10.0,  # Step at 10 seconds
        output_rad=True,
    ).reshape(1, -1)

    # Add initial baseline
    reference_signals = reference_signals + 0.001  # Baseline 0.001 rad/s

    # Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
    # Start at Earth radius with small angular velocity
    initial_state = np.array([6371.0, 0.0, 0.001])

    # Create environment
    env = ImprovedComSatEnv(
        initial_state=initial_state,
        reference_signal=reference_signals,
        number_time_steps=number_time_steps,
        dt=dt,
        initial_thrust=0.0,
        nominal_rho=6371.0,  # Earth radius
    )

    return env


def test_random_policy(env, num_episodes=5):
    """Test environment with random policy.

    Args:
        env (ImprovedComSatEnv): Environment instance.
        num_episodes (int): Number of test episodes. Defaults to 5.
    """
    print("Testing ImprovedComSatEnv with random policy...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < 100:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        print(f"Episode {episode + 1}: Steps={step}, Total Reward={total_reward:.4f}")


def train_ppo_comsat():
    """Train PPO agent for ComSat control.

    This function demonstrates the basic setup for PPO training.
    Requires stable-baselines3 to be installed.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Skipping PPO training.")
        print("Install with: pip install stable-baselines3")
        return

    print("\n" + "=" * 60)
    print("Training PPO agent for ComSat control")
    print("=" * 60 + "\n")

    # Create environment
    def make_env():
        return create_comsat_env(dt=0.01, total_time=20.0)

    env = DummyVecEnv([make_env])

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print("\nTraining agent...")
    # Train for 100k timesteps
    model.learn(total_timesteps=100_000, progress_bar=True)

    print("\nTraining complete!")
    print("Model can be saved with: model.save('ppo_comsat')")
    print("Model can be loaded with: model = PPO.load('ppo_comsat')")

    # Test trained agent
    print("\nTesting trained agent...")
    env = create_comsat_env(dt=0.01, total_time=20.0)
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < 2000:
        # Use trained policy
        action_tensor = model.policy.predict(obs, deterministic=True)[0]
        # Convert from tensor to numpy if needed
        if hasattr(action_tensor, "cpu"):
            action = action_tensor.cpu().numpy()
        else:
            action = np.array(action_tensor)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    print(f"Test episode: Steps={step}, Total Reward={total_reward:.4f}")


if __name__ == "__main__":
    # Create and test environment
    env = create_comsat_env(dt=0.01, total_time=20.0)

    print("=" * 60)
    print("ImprovedComSatEnv Environment Test")
    print("=" * 60)
    print()
    print("Environment Configuration:")
    print(f"  - Max angular velocity: {env.max_angular_velocity} rad/s")
    print(f"  - Max radial velocity: {env.max_radial_velocity} m/s")
    print(f"  - Max radial deviation: {env.max_radial_position_deviation} km")
    print(f"  - Max thrust: {env.max_thrust}")
    print(f"  - Nominal rho: {env.nominal_rho} km")
    print()
    print("Reward weights:")
    print(f"  - Angular velocity tracking: {env.w_theta_dot}")
    print(f"  - Orbital radius: {env.w_rho}")
    print(f"  - Radial velocity: {env.w_rho_dot}")
    print(f"  - Action cost: {env.w_action}")
    print(f"  - Smoothness: {env.w_smooth}")
    print(f"  - Jerk: {env.w_jerk}")
    print()

    # Test with random policy
    test_random_policy(env, num_episodes=3)

    # Optionally train PPO
    user_input = input("\nDo you want to train a PPO agent? (y/n): ")
    if user_input.lower() == "y":
        train_ppo_comsat()
