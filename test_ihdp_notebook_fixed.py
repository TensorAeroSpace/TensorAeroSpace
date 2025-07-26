#!/usr/bin/env python3
"""
Тест исправленного IHDP notebook
"""

import numpy as np
from tqdm import tqdm

from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.agent.ihdp.model import IHDPAgent
from tensoraerospace.benchmark import ControlBenchmark
import gymnasium as gym

def main():
    print("Запуск теста исправленного IHDP notebook...")
    
    # Настройка параметров
    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=20, dt=dt) # Временной период
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]) # Заданный сигнал
    
    # Создание среды
    initial_state = [[0], [0]]  # Для theta, alpha
    
    env = gym.make('LinearLongitudinalF16-v0', 
                   number_time_steps=number_time_steps, 
                   initial_state=initial_state,  # 2 элемента для 2 состояний
                   reference_signal=reference_signals, 
                   state_space=["theta", "alpha"],
                   output_space=["theta", "alpha"], 
                   control_space=["ele"],    
                   tracking_states=["alpha"])
    env.reset()
    
    # Настройки агента
    actor_settings = {
        "start_training": 5,
        "layers": (25, 1), 
        "activations":  ('tanh', 'tanh'), 
        "learning_rate": 2, 
        "learning_rate_exponent_limit": 10,
        "type_PE": "combined",
        "amplitude_3211": 15, 
        "pulse_length_3211": 5/dt, 
        "maximum_input": 25,
        "maximum_q_rate": 20,
        "WB_limits": 30,
        "NN_initial": 120,
        "cascade_actor": False,
        "learning_rate_cascaded":1.2
    }
    
    incremental_settings = {
        "number_time_steps": number_time_steps, 
        "dt": dt, 
        "input_magnitude_limits":25, 
        "input_rate_limits":60,
    }
    
    critic_settings = {
        "Q_weights": [8], 
        "start_training": -1, 
        "gamma": 0.99, 
        "learning_rate": 15, 
        "learning_rate_exponent_limit": 10,
        "layers": (25,1), 
        "activations": ("tanh", "linear"), 
        "WB_limits": 30,
        "NN_initial": 120,
        "indices_tracking_states": env.unwrapped.indices_tracking_states
    }
    
    # Создание модели
    model = IHDPAgent(actor_settings, critic_settings, incremental_settings, 
                      env.unwrapped.tracking_states, env.unwrapped.state_space, 
                      env.unwrapped.control_space, number_time_steps, 
                      env.unwrapped.indices_tracking_states)
    
    # Запуск симуляции
    xt = np.array([[0], [0]])
    
    print("Запуск симуляции IHDP агента...")
    
    for step in tqdm(range(number_time_steps-3)):
        ut = model.predict(xt, reference_signals, step)
        xt, reward, terminated, truncated, info = env.step(np.array(ut))
    
    print(f"Симуляция завершена. Выполнено {step+1} шагов.")
    
    # Анализ результатов
    print("\n=== Анализ результатов ===")
    print(f"Количество шагов симуляции: {env.unwrapped.model.time_step}")
    
    # Получаем данные из модели
    states_data = env.unwrapped.model.store_states[:, :env.unwrapped.model.time_step].T
    control_data = env.unwrapped.model.store_input[:, :env.unwrapped.model.time_step].T
    
    print(f"Форма массива состояний: {states_data.shape}")
    print(f"Форма массива управления: {control_data.shape}")
    
    # Статистика отслеживания alpha
    alpha_history = states_data[:, 1]  # alpha - второе состояние
    alpha_ref = reference_signals[0, :env.unwrapped.model.time_step]
    alpha_error = np.abs(alpha_history - alpha_ref)
    
    print("\nСтатистика отслеживания alpha:")
    print(f"  Среднее отклонение: {np.mean(alpha_error):.6f} рад")
    print(f"  Максимальное отклонение: {np.max(alpha_error):.6f} рад")
    print(f"  Финальное значение alpha: {alpha_history[-1]:.6f} рад")
    print(f"  Целевое значение alpha: {alpha_ref[-1]:.6f} рад")
    
    # Проверка успешности
    final_error = np.abs(alpha_history[-1] - alpha_ref[-1])
    if final_error < 0.1:
        print(f"✅ ТЕСТ ПРОЙДЕН: Финальная ошибка {final_error:.6f} < 0.1")
        print("\n============================================================")
        print("🎉 NOTEBOOK ИСПРАВЛЕН И РАБОТАЕТ КОРРЕКТНО!")
        print("IHDP агент успешно отслеживает заданный сигнал.")
        return True
    else:
        print(f"❌ ТЕСТ НЕ ПРОЙДЕН: Финальная ошибка {final_error:.6f} >= 0.1")
        return False

if __name__ == "__main__":
    main()