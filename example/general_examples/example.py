import gymnasium as gym
import numpy as np

from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16  # noqa: F401


def unit_step(x):
    return np.deg2rad(5) * (x > 1000)


t0 = 0  # Initial time
tn = 20  # Simulation time
dt = 0.01  # Discretization
number_time_steps = int(((tn - t0) / dt) + 1)  # Number of simulation steps
time = list(np.arange(0, number_time_steps * dt, dt))  # Array with step dt
t = np.linspace(-0, len(time), len(time))

reference_signals = np.reshape(unit_step(t), [1, -1])


env = gym.make(
    "LinearLongitudinalF16-v0",
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference_signals,
    number_time_steps=number_time_steps,
)

obs, info = env.reset()
action = np.array([0.0], dtype=np.float32)  # matches 1D action space
obs, reward, terminated, truncated, info = env.step(action)
print(f"reward={reward}, obs={obs}")
