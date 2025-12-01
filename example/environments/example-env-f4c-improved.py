"""Example usage of F4CPitchEnvNormalized for F-4C Phantom II pitch control.

This example demonstrates how to use the normalized RL-friendly environment
for F-4C aircraft with normalized action/observation spaces.
"""

import matplotlib.pyplot as plt
import numpy as np

from tensoraerospace.envs import F4CPitchEnvNormalized


def create_step_reference(num_steps: int, dt: float = 0.01) -> np.ndarray:
    """Create step reference signal for pitch angle.

    Args:
        num_steps: Number of time steps
        dt: Time step duration in seconds

    Returns:
        Reference signal in radians, shape (1, num_steps)
    """
    ref = np.zeros((1, num_steps))
    # Step changes every 5 seconds
    step_duration = int(5.0 / dt)

    # Multiple step changes: 0 -> 5deg -> -5deg -> 10deg -> 0
    ref[:, :step_duration] = np.deg2rad(0.0)
    ref[:, step_duration : 2 * step_duration] = np.deg2rad(5.0)
    ref[:, 2 * step_duration : 3 * step_duration] = np.deg2rad(-5.0)
    ref[:, 3 * step_duration : 4 * step_duration] = np.deg2rad(10.0)
    ref[:, 4 * step_duration :] = np.deg2rad(0.0)

    return ref


def main():
    """Demonstrate F4CPitchEnvNormalized with simple P controller."""
    # Simulation parameters
    dt = 0.01  # 10 ms time step
    duration = 30.0  # 30 seconds
    num_steps = int(duration / dt)

    # Initial state: [u, w, q, theta] in SI units (ft/s, ft/s, rad, rad)
    # u: forward velocity (ft/s), w: vertical velocity (ft/s)
    # q: pitch rate (rad/s), theta: pitch angle (rad)
    initial_state = np.array([1469.76, 0.0, 0.0, 0.0])

    # Create reference signal (step changes in pitch)
    reference_signal = create_step_reference(num_steps, dt)

    # Create environment
    env = F4CPitchEnvNormalized(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=num_steps,
        dt=dt,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=True,
    )

    print("=" * 70)
    print("F4CPitchEnvNormalized - F-4C Phantom II Pitch Control")
    print("=" * 70)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max pitch angle: ±{np.rad2deg(env.max_pitch_rad):.1f} deg")
    print(f"Max pitch rate: ±{np.rad2deg(env.max_pitch_rate_rad_s):.1f} deg/s")
    print(f"Max elevator: ±{env.max_elevator_angle_deg:.1f} deg")
    print("=" * 70)

    # Storage for results
    time_history = []
    theta_history = []
    theta_ref_history = []
    elevator_history = []
    reward_history = []

    # Simple P controller gains
    kp_pitch = 2.5  # Proportional gain for pitch error
    kp_rate = 0.3  # Proportional gain for pitch rate

    # Reset environment
    obs, info = env.reset()
    total_reward = 0.0

    print("\nRunning simulation with simple PD controller...")
    print("-" * 70)

    for step in range(num_steps):
        # Extract normalized observations
        # obs = [pitch_error_norm, q_norm, theta_norm, prev_action_norm]
        pitch_error_norm = obs[0]
        q_norm = obs[1]

        # Simple PD control law (in normalized space)
        action_norm = kp_pitch * pitch_error_norm - kp_rate * q_norm
        action_norm = np.clip(action_norm, -1.0, 1.0)

        # Execute step
        obs, reward, terminated, truncated, info = env.step([action_norm])
        total_reward += reward

        # Store data for plotting
        current_time = step * dt
        time_history.append(current_time)
        # Convert normalized observation back to degrees for plotting
        theta_deg = obs[2] * np.rad2deg(env.max_pitch_rad)
        theta_history.append(theta_deg)
        theta_ref_deg = np.rad2deg(reference_signal[0, step])
        theta_ref_history.append(theta_ref_deg)
        elevator_deg = action_norm * env.max_elevator_angle_deg
        elevator_history.append(elevator_deg)
        reward_history.append(reward)

        # Print progress every 5 seconds
        if step % int(5.0 / dt) == 0:
            print(
                f"t={current_time:5.1f}s | "
                f"θ={theta_deg:+6.2f}° | "
                f"θ_ref={theta_ref_deg:+6.2f}° | "
                f"ele={elevator_deg:+6.2f}° | "
                f"reward={reward:+7.4f}"
            )

        if terminated:
            print(f"\n⚠️  Episode terminated at step {step} " f"(pitch exceeded limits)")
            break

        if truncated:
            break

    print("-" * 70)
    print(f"Simulation complete!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward / len(reward_history):.4f}")
    print("=" * 70)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Pitch angle
    axes[0].plot(time_history, theta_ref_history, "r--", linewidth=2, label="Reference")
    axes[0].plot(time_history, theta_history, "b-", linewidth=1.5, label="Actual")
    axes[0].set_ylabel("Pitch Angle (deg)", fontsize=12)
    axes[0].set_title(
        "F-4C Phantom II Pitch Control (F4CPitchEnvNormalized)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Elevator deflection
    axes[1].plot(time_history, elevator_history, "g-", linewidth=1.5)
    axes[1].set_ylabel("Elevator (deg)", fontsize=12)
    axes[1].axhline(y=env.max_elevator_angle_deg, color="r", linestyle=":", alpha=0.5)
    axes[1].axhline(y=-env.max_elevator_angle_deg, color="r", linestyle=":", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Reward
    axes[2].plot(time_history, reward_history, "m-", linewidth=1.5)
    axes[2].set_ylabel("Reward", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("f4c_improved_env_results.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Results saved to: f4c_improved_env_results.png")
    plt.show()

    # Demonstrate rendering (optional)
    try:
        print("\nPress Ctrl+C to stop rendering...")
        env.reset()
        for step in range(min(500, num_steps)):
            obs_norm = obs[0]
            q_norm = obs[1]
            action_norm = kp_pitch * obs_norm - kp_rate * q_norm
            action_norm = np.clip(action_norm, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step([action_norm])
            env.render()
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("\nRendering stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
