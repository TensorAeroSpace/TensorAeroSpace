#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования улучшенного ControlBenchmark с расширенными метриками.

Этот файл демонстрирует:
- Анализ одиночной системы управления с новыми метриками
- Сравнение нескольких систем
- Генерацию детализированных отчетов
- Визуализацию с расширенными таблицами метрик
"""

import matplotlib.pyplot as plt
import numpy as np

from tensoraerospace.benchmark.bench import ControlBenchmark


def generate_sample_system_response(
    time_points, overshoot=10.0, settling_time=5.0, noise_level=0.01
):
    """
    Генерирует образец отклика системы управления.

    Args:
        time_points: Массив временных точек
        overshoot: Перерегулирование в процентах
        settling_time: Время установления
        noise_level: Уровень шума

    Returns:
        Массив значений отклика системы
    """
    # Параметры для генерации переходного процесса
    zeta = 0.7 - (overshoot / 100.0) * 0.5  # Коэффициент затухания
    wn = 4.0 / settling_time  # Собственная частота

    # Генерируем переходный процесс второго порядка
    if zeta < 1.0:
        wd = wn * np.sqrt(1 - zeta**2)
        response = 1 - np.exp(-zeta * wn * time_points) * (
            np.cos(wd * time_points) + (zeta * wn / wd) * np.sin(wd * time_points)
        )
    else:
        # Апериодический процесс
        r1 = -wn * (zeta + np.sqrt(zeta**2 - 1))
        r2 = -wn * (zeta - np.sqrt(zeta**2 - 1))
        response = 1 + (
            r2 * np.exp(r1 * time_points) - r1 * np.exp(r2 * time_points)
        ) / (r2 - r1)

    # Добавляем небольшой шум
    noise = np.random.normal(0, noise_level, len(time_points))
    response += noise

    return response


def main():
    """Демонстрация использования улучшенного ControlBenchmark с расширенными метриками."""
    print("🚀 Демонстрация улучшенного ControlBenchmark с расширенными метриками")
    print("=" * 70)

    # Создаем экземпляр бенчмарка
    benchmark = ControlBenchmark()

    # Генерируем тестовые данные
    time_points = np.linspace(0, 10, 1000)
    dt = time_points[1] - time_points[0]

    # Система 1: Хорошо настроенная система
    control_signal_1 = np.ones_like(time_points)
    system_signal_1 = generate_sample_system_response(
        time_points, overshoot=5.0, settling_time=3.0
    )

    print("\n📊 Анализ одиночной системы с расширенными метриками:")
    print("-" * 50)

    # Построение графика для одной системы
    benchmark.plot(
        control_signal_1,
        system_signal_1,
        signal_val=1.0,
        dt=dt,
        title="Хорошо настроенная система управления",
    )
    plt.show()

    # Генерация расширенного отчета
    report = benchmark.generate_report(
        control_signal_1,
        system_signal_1,
        signal_val=1.0,
        dt=dt,
        system_name="Оптимальная система",
    )
    print(report)

    print("\n📈 Сравнение нескольких систем с новыми метриками:")
    print("-" * 50)

    # Система 2: Переколебательная система
    system_signal_2 = generate_sample_system_response(
        time_points, overshoot=25.0, settling_time=5.0
    )

    # Система 3: Медленная система
    system_signal_3 = generate_sample_system_response(
        time_points, overshoot=2.0, settling_time=8.0
    )

    # Система 4: Быстрая но нестабильная
    system_signal_4 = generate_sample_system_response(
        time_points, overshoot=35.0, settling_time=2.5
    )

    # Сравнение систем
    systems_data = {
        "Оптимальная": (control_signal_1, system_signal_1),
        "Переколебательная": (control_signal_1, system_signal_2),
        "Медленная": (control_signal_1, system_signal_3),
        "Быстрая/Нестабильная": (control_signal_1, system_signal_4),
    }

    benchmark.compare_systems(
        systems_data,
        signal_val=1.0,
        dt=dt,
        title="Сравнение различных систем управления (расширенные метрики)",
    )
    plt.show()

    print("\n🔍 Демонстрация новых метрик:")
    print("-" * 30)

    # Вычисляем метрики для демонстрации
    metrics = benchmark.becnchmarking_one_step(
        control_signal_1, system_signal_1, signal_val=1.0, dt=dt
    )

    print(f"📏 Временные характеристики:")
    print(f"   • Время нарастания: {metrics['rise_time']:.3f} с")
    print(f"   • Время пика: {metrics['peak_time']:.3f} с")
    print(f"   • Время установления: {metrics['settling_time']:.3f} с")

    print(f"\n📊 Характеристики переходного процесса:")
    print(f"   • Максимальное отклонение: {metrics['maximum_deviation']:.3f}")
    print(f"   • Количество колебаний: {metrics['oscillation_count']}")
    print(f"   • Установившееся значение: {metrics['steady_state_value']:.3f}")

    print(f"\n🎯 Интегральные критерии качества:")
    print(f"   • IAE (Интегральная абсолютная ошибка): {metrics['iae']:.2f}")
    print(f"   • ISE (Интегральная квадратичная ошибка): {metrics['ise']:.2f}")
    print(f"   • ITAE (Взвешенная по времени абс. ошибка): {metrics['itae']:.2f}")
    print(f"   • Комплексный индекс качества: {metrics['performance_index']:.3f}")

    print("\n✅ Демонстрация завершена!")
    print("Новые возможности включают:")
    print("• 🕐 Временные характеристики (время нарастания, время пика)")
    print("• 📈 Характеристики переходного процесса (макс. отклонение, колебания)")
    print("• 🎯 Интегральные критерии качества (IAE, ISE, ITAE)")
    print("• 🏆 Комплексный индекс качества системы")
    print("• 📊 Расширенные таблицы сравнения")
    print("• 📋 Детализированные отчеты")


if __name__ == "__main__":
    main()
