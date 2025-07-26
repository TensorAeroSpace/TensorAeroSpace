#!/usr/bin/env python3
"""
Тест для проверки валидации состояний в модели F16
"""

import gymnasium as gym
import numpy as np
from tensoraerospace.signals.standart import unit_step

# Параметры симуляции
number_time_steps = 2000
tp = np.arange(0, number_time_steps * 0.01, 0.01)
reference_signals = unit_step(tp, 1, 500, 0.01, output_rad=True)

print("Тест 1: Проверка с корректными состояниями")
try:
    # Корректные состояния из файла keySet_states.txt
    initial_state = [[0], [0], [0], [0]]
    env = gym.make('LinearLongitudinalF16-v0', 
                   number_time_steps=number_time_steps, 
                   initial_state=initial_state,
                   reference_signal=reference_signals, 
                   use_reward=False, 
                   state_space=["theta", "alpha", "q", "ele"], 
                   output_space=["theta", "alpha", "q", "nz"], 
                   control_space=["ele"], 
                   tracking_states=["alpha"])
    print("✓ Успешно создана среда с корректными состояниями")
    env.reset()
    print("✓ Среда успешно инициализирована")
except Exception as e:
    print(f"✗ Ошибка с корректными состояниями: {e}")

print("\nТест 2: Проверка с некорректными состояниями")
try:
    # Некорректные состояния (не существуют в файле)
    initial_state = [[0], [0], [0], [0]]
    env = gym.make('LinearLongitudinalF16-v0', 
                   number_time_steps=number_time_steps, 
                   initial_state=initial_state,
                   reference_signal=reference_signals, 
                   use_reward=False, 
                   state_space=["theta", "alpha", "q", "invalid_state"], 
                   output_space=["theta", "alpha", "q", "nz"], 
                   control_space=["ele"], 
                   tracking_states=["alpha"])
    print("✗ Среда создана с некорректными состояниями (не должно было произойти)")
except ValueError as e:
    print(f"✓ Правильно обнаружена ошибка с некорректными состояниями: {e}")
except Exception as e:
    print(f"✗ Неожиданная ошибка: {e}")

print("\nТест 3: Проверка с некорректными входами")
try:
    # Некорректные входы
    initial_state = [[0], [0], [0], [0]]
    env = gym.make('LinearLongitudinalF16-v0', 
                   number_time_steps=number_time_steps, 
                   initial_state=initial_state,
                   reference_signal=reference_signals, 
                   use_reward=False, 
                   state_space=["theta", "alpha", "q", "ele"], 
                   output_space=["theta", "alpha", "q", "nz"], 
                   control_space=["invalid_input"], 
                   tracking_states=["alpha"])
    print("✗ Среда создана с некорректными входами (не должно было произойти)")
except ValueError as e:
    print(f"✓ Правильно обнаружена ошибка с некорректными входами: {e}")
except Exception as e:
    print(f"✗ Неожиданная ошибка: {e}")

print("\nТест 4: Вывод доступных состояний, выходов и входов")
try:
    from tensoraerospace.aerospacemodel.f16.linear.longitudinal.model import LongitudinalF16
    
    # Создаем модель для получения доступных ключей
    model = LongitudinalF16(x0=[[0], [0], [0], [0]], number_time_steps=100)
    
    print(f"Доступные состояния: {model.available_states}")
    print(f"Доступные выходы: {model.available_outputs}")
    print(f"Доступные входы: {model.available_inputs}")
except Exception as e:
    print(f"✗ Ошибка при получении доступных ключей: {e}")

print("\nТесты завершены!")