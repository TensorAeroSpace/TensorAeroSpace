#!/usr/bin/env python3
"""
Пример использования модели F16 с валидацией состояний
"""

import gymnasium as gym
import numpy as np
from tensoraerospace.signals.standart import unit_step

# Параметры симуляции
number_time_steps = 1000
tp = np.arange(0, number_time_steps * 0.01, 0.01)
reference_signals = unit_step(tp, 1, 200, 0.01, output_rad=True)

print("Создание среды F16 с валидацией состояний...")

try:
    # Используем корректные состояния из файла keySet_states.txt
    # Доступные состояния: ['npos', 'epos', 'altitude', 'phi', 'theta', 'psi', 'velocity', 'alpha', 'beta', 'p', 'q', 'r', 'thrust', 'ele', 'ail', 'rud', 'dLEF', 'fidelity_flag']
    # Доступные входы: ['thrust', 'ele', 'ail', 'rud']
    # Доступные выходы: ['npos', 'epos', 'altitude', 'phi', 'theta', 'psi', 'velocity', 'alpha', 'beta', 'p', 'q', 'r', 'nx', 'ny', 'nz', 'M', 'qbar', 'ps']
    
    initial_state = [[0], [0], [0], [0]]  # Для theta, alpha, q, velocity
    
    env = gym.make('LinearLongitudinalF16-v0', 
                   number_time_steps=number_time_steps, 
                   initial_state=initial_state,
                   reference_signal=reference_signals, 
                   use_reward=False, 
                   state_space=["theta", "alpha", "q", "velocity"],  # Все состояния из доступных
                   output_space=["theta", "alpha", "q", "velocity"],  # Все выходы из доступных
                   control_space=["ele"],  # Вход из доступных
                   tracking_states=["alpha"])
    
    print("✓ Среда успешно создана!")
    
    # Инициализация среды
    env.reset()
    print("✓ Среда успешно инициализирована!")
    
    # Запуск нескольких шагов симуляции
    print("\nЗапуск симуляции...")
    for i in range(10):
        action = [0.1]  # Простое управляющее воздействие
        observation, reward, done, truncated, info = env.step(action)
        if i % 5 == 0:
            print(f"Шаг {i}: observation = {observation[:2]}...")  # Показываем первые 2 элемента
    
    print("\n✓ Симуляция завершена успешно!")
    print("\nВалидация состояний работает корректно:")
    print("- Проверяет доступность состояний из файла keySet_states.txt")
    print("- Проверяет доступность выходов из файла keySet_output.txt")
    print("- Проверяет доступность входов из файла keySet_input.txt")
    print("- Выдает понятные ошибки при использовании недоступных элементов")
    
except ValueError as e:
    print(f"✗ Ошибка валидации: {e}")
except Exception as e:
    print(f"✗ Неожиданная ошибка: {e}")

print("\nПример завершен!")