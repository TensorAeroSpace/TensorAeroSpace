from tensorairspace.envs.f16.linear_longitudial import LinearLongitudinalF16
from tensorairspace.aircraftmodel.model.f16.linear.longitudinal import set_initial_state, initial_state


import numpy as np

def unit_step(x):
    return np.deg2rad(5) * (x > 1000)


t0 = 0  # Начальное время
tn = 20  # Время моделирования
dt = 0.01  # Дисретизация
number_time_steps = int(((tn - t0) / dt) + 1)  # Количество шагов моделирования
time = list(np.arange(0, number_time_steps * dt, dt)) # Массив с шагов dt
t = np.linspace(-0, len(time), len(time))

reference_signals =  np.reshape(unit_step(t),  [1, -1])


import gym
env = gym.make('LinearLongitudinalF16-v0', initial_state=[[0],[0],[0],[0]], reference_signal = reference_signals)
env.reset()

new_actin = [[0]]
reward = env.step([1])
print(reward)