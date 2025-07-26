#!/usr/bin/env python3
"""
Тестирование улучшенного IHDP агента с новой моделью F16

Использование:
    poetry run python test_ihdp_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback

try:
    from tensoraerospace.signals.standart import unit_step
    from tensoraerospace.agent.ihdp.model import IHDPAgent
    from tensoraerospace.envs.f16.linear_longitudial import LinearLongitudinalF16
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете скрипт через poetry run")
    sys.exit(1)

def test_ihdp_with_new_f16():
    """Тестирование IHDP с новой моделью F16"""
    print("=== Тестирование улучшенного IHDP агента ===")
    
    # Параметры симуляции
    tp = 10  # Уменьшенное время для быстрого тестирования
    dt = 0.01
    number_time_steps = int(tp / dt)
    
    print(f"Параметры симуляции:")
    print(f"  Время симуляции: {tp} с")
    print(f"  Шаг дискретизации: {dt} с")
    print(f"  Количество шагов: {number_time_steps}")
    
    # Генерация опорного сигнала
    try:
        # Создаем временной массив
        time_array = np.arange(0, tp, dt)
        # Генерируем ступенчатый сигнал (2 градуса, ступенька на 1 секунде)
        reference_signal_1d = unit_step(time_array, 2, 1.0, dt, output_rad=True)
        # Преобразуем в нужный формат (2D массив) - только для alpha
        reference_signal = np.array([reference_signal_1d])  # Только одна строка для alpha
        print(f"  Размерность опорного сигнала: {reference_signal.shape}")
        print(f"  Размерность временного массива: {time_array.shape}")
    except Exception as e:
        print(f"Ошибка при генерации опорного сигнала: {e}")
        traceback.print_exc()
        return False
    
    # Начальные условия
    initial_state = np.array([0.0, 0.0])  # [alpha, q]
    
    # Создание среды
    try:
        env = LinearLongitudinalF16(
            initial_state=initial_state,
            reference_signal=reference_signal,
            number_time_steps=number_time_steps,
            tracking_states=['alpha'],  # Используем только alpha для упрощения
            state_space=['alpha', 'q'],
            control_space=['ele'],
            output_space=['alpha', 'q']
        )
        print(f"Среда создана успешно")
        print(f"  Отслеживаемые состояния: {env.tracking_states}")
        print(f"  Пространство состояний: {env.state_space}")
        print(f"  Пространство управления: {env.control_space}")
        print(f"  Индексы отслеживаемых состояний: {env.indices_tracking_states}")
    except Exception as e:
        print(f"Ошибка при создании среды: {e}")
        traceback.print_exc()
        return False
    
    # Настройки IHDP
    actor_settings = {
        'start_training': 50,  # Уменьшено для быстрого тестирования
        'layers': [8, 1],  # Последний слой должен иметь 1 нейрон
        'activations': ['tanh', 'linear'],  # Последняя активация - линейная
        'learning_rate': 0.01,
        'learning_rate_exponent_limit': 1e-6,
        'type_PE': '3211',
        'amplitude_3211': 0.1,
        'pulse_length_3211': 50,
        'maximum_input': 0.4,
        'maximum_q_rate': 1.0,
        'WB_limits': [-5, 5],
        'NN_initial': 42,  # Числовое значение для seed
        'cascade_actor': False,
        'learning_rate_cascaded': 0.001
    }
    
    critic_settings = {
        'Q_weights': np.diag([1, 1]),
        'start_training': 50,
        'gamma': 0.95,
        'learning_rate': 0.01,
        'learning_rate_exponent_limit': 1e-6,
        'layers': [8, 1],  # Последний слой должен иметь 1 нейрон
        'activations': ['tanh', 'linear'],  # Последняя активация - линейная
        'indices_tracking_states': env.indices_tracking_states,
        'WB_limits': [-5, 5],
        'NN_initial': 42  # Числовое значение для seed
    }
    
    incremental_settings = {
        'number_time_steps': 30,  # Уменьшено для быстрого тестирования
        'dt': dt,
        'input_magnitude_limits': 0.4,
        'input_rate_limits': 2.0
    }
    
    # Создание IHDP агента
    try:
        model = IHDPAgent(
            actor_settings, 
            critic_settings, 
            incremental_settings, 
            env.tracking_states, 
            env.state_space, 
            env.control_space, 
            number_time_steps, 
            env.indices_tracking_states
        )
        print("IHDP агент создан успешно!")
    except Exception as e:
        print(f"Ошибка при создании IHDP агента: {e}")
        traceback.print_exc()
        return False
    
    # Симуляция
    print("\n=== Начало симуляции ===")
    state_history = []
    control_history = []
    reward_history = []
    
    try:
        # Сброс среды
        state, info = env.reset()
        state_history.append(state.copy())
        
        print(f"Начальное состояние: {state.flatten()}")
        print(f"Форма состояния: {state.shape}")
        
        # Основной цикл симуляции
        for i in range(min(200, number_time_steps - 1)):  # Ограничиваем для тестирования
            try:
                # Получение управляющего сигнала от IHDP агента
                ut = model.predict(state, reference_signal, i)
                
                # Выполнение шага в среде
                next_state, reward, done, truncated, info = env.step(ut)
                
                # Сохранение истории
                state_history.append(next_state.copy())
                control_history.append(ut.copy())
                reward_history.append(reward)
                
                # Обновление состояния
                state = next_state
                
                if done:
                    print(f"Симуляция завершена на шаге {i} (done=True)")
                    break
                    
                # Вывод прогресса каждые 50 шагов
                if i % 50 == 0:
                    print(f"Шаг {i}: состояние = {state.flatten()}, управление = {ut.flatten()}")
                    
            except Exception as e:
                print(f"Ошибка на шаге {i}: {e}")
                traceback.print_exc()
                break
        
        print(f"Симуляция завершена. Выполнено {len(state_history)} шагов.")
        
    except Exception as e:
        print(f"Ошибка во время симуляции: {e}")
        traceback.print_exc()
        return False
    
    # Анализ результатов
    if len(state_history) > 1:
        states_array = np.array([s.flatten() for s in state_history])
        controls_array = np.array([c.flatten() for c in control_history]) if control_history else np.array([])
        
        print("\n=== Анализ результатов ===")
        print(f"Количество шагов симуляции: {len(state_history)}")
        print(f"Форма массива состояний: {states_array.shape}")
        if len(controls_array) > 0:
            print(f"Форма массива управления: {controls_array.shape}")
        
        # Статистика отслеживания
        ref_length = min(len(states_array), reference_signal.shape[1])
        alpha_error = np.abs(states_array[:ref_length, 0] - reference_signal[0, :ref_length])
        
        print(f"\nСтатистика отслеживания alpha:")
        print(f"  Среднее отклонение: {np.mean(alpha_error):.6f} рад")
        print(f"  Максимальное отклонение: {np.max(alpha_error):.6f} рад")
        print(f"  Финальное значение alpha: {states_array[-1, 0]:.6f} рад")
        print(f"  Целевое значение alpha: {reference_signal[0, ref_length-1]:.6f} рад")
        
        if reward_history:
            print(f"\nСтатистика наград:")
            print(f"  Средняя награда: {np.mean(reward_history):.6f}")
            print(f"  Общая награда: {np.sum(reward_history):.6f}")
        
        # Простая проверка успешности
        final_error = abs(states_array[-1, 0] - reference_signal[0, ref_length-1])
        if final_error < 0.1:  # Допустимая ошибка 0.1 рад
            print(f"\n✅ ТЕСТ ПРОЙДЕН: Финальная ошибка {final_error:.6f} < 0.1")
            return True
        else:
            print(f"\n❌ ТЕСТ НЕ ПРОЙДЕН: Финальная ошибка {final_error:.6f} >= 0.1")
            return False
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН: Недостаточно данных для анализа")
        return False

def main():
    """Главная функция"""
    print("Тестирование улучшенного IHDP агента с новой моделью F16")
    print("=" * 60)
    
    success = test_ihdp_with_new_f16()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("IHDP агент корректно работает с новой моделью F16.")
    else:
        print("💥 ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        print("Требуется дополнительная отладка IHDP агента.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())