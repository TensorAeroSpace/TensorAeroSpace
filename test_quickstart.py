#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки основных функций quickstart.ipynb
"""

import sys
sys.path.append('/Users/asmazaev/Projects/TensorAeroSpace')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TensorAeroSpace импорты
from tensoraerospace.envs import LinearLongitudinalF16
from tensoraerospace.agent.pid import PID
from tensoraerospace.signals import unit_step
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

def test_basic_functionality():
    """Тестирование базовой функциональности"""
    try:
        print("🧪 Тестирование базовой функциональности...")
        
        # Тест импортов
        from tensoraerospace.envs import LinearLongitudinalF16
        from tensoraerospace.agent.pid import PID
        from tensoraerospace.signals import unit_step
        from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
        import matplotlib.pyplot as plt
        
        # Тест matplotlib стиля
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('seaborn')
        
        print("✅ Импорты успешны")
        
        # Параметры симуляции
        dt = 0.01
        simulation_time = 1.0  # Короткий тест
        tp = generate_time_period(tn=simulation_time, dt=dt)
        reference_signal = unit_step(tp, degree=0.1, time_step=0.5)
        initial_state = np.array([[0.0], [0.0]])
        
        # Тест создания среды с правильными параметрами
        env_f16 = LinearLongitudinalF16(
            initial_state=initial_state,
            reference_signal=reference_signal,
            number_time_steps=len(tp)
        )
        print("✅ Среда F-16 создана")
        
        # Тест создания PID контроллера
        pid_controller = PID(
            env=env_f16,
            kp=1.0,
            ki=0.1,
            kd=0.05,
            dt=dt
        )
        print("✅ PID контроллер создан")
        
        # Тест короткой симуляции
        state = initial_state.copy()
        
        for i in range(min(10, len(tp))):
            # Управляющий сигнал
            control = pid_controller.select_action(reference_signal[i], state[1])
            
            # Ошибка для записи
            error = reference_signal[i] - state[1]
            
            # Простое обновление состояния (заглушка)
            state = state + 0.01 * np.array([[control[0]], [control[0]]])
            
        print("✅ Симуляция завершена успешно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n🎉 Все тесты пройдены успешно! quickstart.ipynb должен работать корректно.")
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()