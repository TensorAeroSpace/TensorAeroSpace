#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования улучшенного бенчмарка для анализа систем управления.

Этот файл демонстрирует, как использовать новые возможности класса ControlBenchmark
для создания красивых и информативных графиков анализа качества управления.
"""

import matplotlib.pyplot as plt
import numpy as np

from tensoraerospace.benchmark import ControlBenchmark


def generate_sample_system_response(
    time, overshoot=0.2, settling_time=3.0, noise_level=0.01
):
    """
    Генерирует примерный отклик системы управления.

    Args:
        time (np.ndarray): Временной массив
        overshoot (float): Перерегулирование (0.0 - 1.0)
        settling_time (float): Время установления
        noise_level (float): Уровень шума

    Returns:
        tuple: (control_signal, system_signal)
    """
    # Задающий сигнал (ступенчатый)
    control_signal = np.ones_like(time)
    control_signal[time < 1.0] = 0

    # Параметры системы второго порядка
    wn = 4.0 / settling_time  # Собственная частота
    zeta = -np.log(overshoot) / np.sqrt(
        np.pi**2 + np.log(overshoot) ** 2
    )  # Коэффициент затухания

    # Отклик системы второго порядка на ступенчатое воздействие
    system_signal = np.zeros_like(time)

    for i, t in enumerate(time):
        if t >= 1.0:
            tau = t - 1.0
            if zeta < 1.0:  # Недодемпфированная система
                wd = wn * np.sqrt(1 - zeta**2)
                response = 1 - np.exp(-zeta * wn * tau) * (
                    np.cos(wd * tau) + (zeta * wn / wd) * np.sin(wd * tau)
                )
            else:  # Передемпфированная система
                r1 = -wn * (zeta + np.sqrt(zeta**2 - 1))
                r2 = -wn * (zeta - np.sqrt(zeta**2 - 1))
                response = 1 + (r2 * np.exp(r1 * tau) - r1 * np.exp(r2 * tau)) / (
                    r2 - r1
                )

            system_signal[i] = response

    # Добавляем шум
    system_signal += np.random.normal(0, noise_level, len(system_signal))

    return control_signal, system_signal


def main():
    """
    Основная функция демонстрации возможностей бенчмарка.
    """
    print("🚀 Демонстрация улучшенного бенчмарка TensorAeroSpace")
    print("=" * 60)

    # Создаем временной массив
    dt = 0.01
    time = np.arange(0, 10, dt)

    # Создаем экземпляр бенчмарка
    benchmark = ControlBenchmark()

    print("\n1️⃣  Анализ одной системы управления")
    print("-" * 40)

    # Генерируем данные для одной системы
    control_signal, system_signal = generate_sample_system_response(
        time, overshoot=0.15, settling_time=2.5, noise_level=0.005
    )

    # Строим красивый график
    metrics = benchmark.plot(
        control_signal,
        system_signal,
        signal_val=0.5,
        dt=dt,
        tps=time,
        title="Анализ ПИД-регулятора",
    )

    # Генерируем отчет
    report = benchmark.generate_report(
        control_signal,
        system_signal,
        signal_val=0.5,
        dt=dt,
        system_name="ПИД-регулятор",
    )
    print(report)

    print("\n2️⃣  Сравнение нескольких систем управления")
    print("-" * 50)

    # Создаем данные для сравнения нескольких систем
    systems_data = {}

    # Система 1: Быстрая с перерегулированием
    control1, system1 = generate_sample_system_response(
        time, overshoot=0.25, settling_time=1.5, noise_level=0.003
    )
    systems_data["Быстрая система"] = {
        "control_signal": control1,
        "system_signal": system1,
        "time": time,
    }

    # Система 2: Медленная без перерегулирования
    control2, system2 = generate_sample_system_response(
        time, overshoot=0.05, settling_time=4.0, noise_level=0.002
    )
    systems_data["Медленная система"] = {
        "control_signal": control2,
        "system_signal": system2,
        "time": time,
    }

    # Система 3: Оптимальная
    control3, system3 = generate_sample_system_response(
        time, overshoot=0.10, settling_time=2.0, noise_level=0.004
    )
    systems_data["Оптимальная система"] = {
        "control_signal": control3,
        "system_signal": system3,
        "time": time,
    }

    # Сравниваем системы
    all_metrics = benchmark.compare_systems(systems_data, signal_val=0.5, dt=dt)

    # Выводим сравнительную таблицу в консоль
    print("\n📊 Сравнительная таблица метрик:")
    print("-" * 80)
    print(
        f"{'Система':<20} {'Перерег.%':<12} {'Время уст.':<12} {'Затухание':<12} {'Стат.ошибка':<12}"
    )
    print("-" * 80)

    for system_name, metrics in all_metrics.items():
        print(
            f"{system_name:<20} {metrics['overshoot']:<12.2f} "
            f"{metrics['settling_time']:<12.3f} {metrics['damping_degree']:<12.3f} "
            f"{metrics['static_error']:<12.4f}"
        )

    print("\n✅ Демонстрация завершена!")
    print("\n💡 Возможности улучшенного бенчмарка:")
    print("   • Красивые графики с современным дизайном")
    print("   • Автоматические аннотации ключевых точек")
    print("   • Таблицы с метриками качества")
    print("   • Сравнение нескольких систем")
    print("   • Графики ошибок регулирования")
    print("   • Текстовые отчеты с оценками")
    print("   • Настраиваемые цветовые схемы")


if __name__ == "__main__":
    main()
