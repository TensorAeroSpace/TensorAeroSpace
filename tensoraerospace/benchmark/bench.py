"""
Модуль для оценки качества систем управления.

Этот модуль предоставляет инструменты для анализа переходных процессов
в системах автоматического управления, включая расчет различных метрик
качества и визуализацию результатов с помощью интерактивных графиков.

Основные компоненты:
- ControlBenchmark: Класс для комплексной оценки систем управления
- Метрики качества: перерегулирование, время установления, статическая ошибка и др.
- Интерактивная визуализация с использованием Plotly
"""

from typing import Dict, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .function import (
    damping_degree,
    find_step_function,
    get_lower_upper_bound,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    maximum_deviation,
    oscillation_count,
    overshoot,
    peak_time,
    performance_index,
    rise_time,
    settling_time,
    static_error,
    steady_state_value,
)

# Цветовая палитра для графиков
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#7209B7"]


class ControlBenchmark:
    """Класс для проведения оценки системы управления и построения красивых графиков.

    Предоставляет инструменты для анализа качества переходных процессов
    в системах автоматического управления с визуализацией результатов.
    """

    def becnchmarking_one_step(
        self,
        control_signal: np.ndarray,
        system_signal: np.ndarray,
        signal_val: float,
        dt: float,
    ) -> dict:
        """
        Оценивает систему управления на одном шаге и возвращает расширенный набор результатов в виде словаря.

        Args:
            control_signal (numpy.ndarray): Сигнал управления системы.
            system_signal (numpy.ndarray): Сигнал системы, на которую воздействует управление.
            signal_val (float): Значение сигнала, с которого начинается функция перехода.
            dt (float): Шаг дискретизации.

        Returns:
            dict: Словарь с результатами оценки системы управления:
                  - "overshoot" (float): перерегулирование (%),
                  - "settling_time" (float): время установления (с),
                  - "damping_degree" (float): степень затухания,
                  - "static_error" (float): статическая ошибка,
                  - "rise_time" (float): время нарастания (с),
                  - "peak_time" (float): время достижения пика (с),
                  - "maximum_deviation" (float): максимальное отклонение,
                  - "iae" (float): интегральная абсолютная ошибка,
                  - "ise" (float): интегральная квадратичная ошибка,
                  - "itae" (float): интегральная абсолютная ошибка, взвешенная по времени,
                  - "oscillation_count" (int): количество колебаний,
                  - "steady_state_value" (float): установившееся значение,
                  - "performance_index" (float): комплексный индекс качества.
        """
        control_signal, system_signal = find_step_function(
            control_signal, system_signal, signal_val=signal_val
        )

        # Основные метрики
        overshooting = overshoot(control_signal, system_signal)
        cnt_time = settling_time(control_signal, system_signal)
        damp = damping_degree(system_signal)

        # Исправляем расчет статической ошибки - используем весь сигнал после переходного процесса
        if cnt_time is not None and cnt_time < len(control_signal):
            static_err = static_error(
                control_signal[cnt_time:], system_signal[cnt_time:]
            )
        else:
            # Если время установления не найдено или равно длине массива, используем последние 10% сигнала
            start_idx = max(0, int(0.9 * len(control_signal)))
            static_err = static_error(
                control_signal[start_idx:], system_signal[start_idx:]
            )

        # Дополнительные метрики
        rise_t = rise_time(control_signal, system_signal)
        peak_t = peak_time(system_signal)
        max_dev = maximum_deviation(control_signal, system_signal)
        iae = integral_absolute_error(control_signal, system_signal)
        ise = integral_squared_error(control_signal, system_signal)
        itae = integral_time_absolute_error(control_signal, system_signal, dt)
        osc_count = oscillation_count(system_signal)
        steady_val = steady_state_value(control_signal)
        perf_idx = performance_index(control_signal, system_signal, dt)

        return {
            "overshoot": overshooting,
            "settling_time": cnt_time * dt if cnt_time is not None else None,
            "damping_degree": damp,
            "static_error": static_err,
            "rise_time": rise_t * dt if rise_t is not None else None,
            "peak_time": peak_t * dt if peak_t is not None else None,
            "maximum_deviation": max_dev,
            "iae": iae,
            "ise": ise,
            "itae": itae,
            "oscillation_count": osc_count,
            "steady_state_value": steady_val,
            "performance_index": perf_idx,
        }

    def plot(
        self,
        control_signal: np.ndarray,
        system_signal: np.ndarray,
        signal_val: float,
        dt: float,
        tps: np.ndarray,
        figsize: tuple = (1600, 1000),
        title: str = "Анализ системы управления",
    ):
        """
        Строит интерактивный график анализа системы управления с использованием Plotly.

        Args:
            control_signal (numpy.ndarray): Сигнал управления системы.
            system_signal (numpy.ndarray): Сигнал системы, на которую воздействует управление.
            signal_val (float): Значение сигнала, с которого начинается функция перехода.
            dt (float): Шаг дискретизации.
            tps (numpy.ndarray): Массив временных меток.
            figsize (tuple): Размер графика в пикселях, по умолчанию (1600, 1000).
            title (str): Заголовок графика.
        """
        # Получаем метрики
        metrics = self.becnchmarking_one_step(
            control_signal, system_signal, signal_val, dt
        )
        control_signal_step, system_signal_step = find_step_function(
            control_signal, system_signal, signal_val=signal_val
        )
        lower, upper = get_lower_upper_bound(control_signal_step)
        ntime = tps[: len(control_signal_step)]

        # Создаем подграфики
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.7, 0.3],
            specs=[
                [{"type": "scatter"}, {"type": "table"}],
                [{"type": "scatter", "colspan": 2}, None],
            ],
            subplot_titles=(
                "Переходный процесс",
                "Метрики качества",
                "Ошибка регулирования",
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        # Основной график - переходный процесс
        # Задающий сигнал
        fig.add_trace(
            go.Scatter(
                x=ntime,
                y=control_signal_step,
                mode="lines",
                name="Задающий сигнал",
                line=dict(color=COLORS[0], width=3),
                hovertemplate="Время: %{x:.3f}с<br>Амплитуда: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Выходной сигнал
        fig.add_trace(
            go.Scatter(
                x=ntime,
                y=system_signal_step,
                mode="lines",
                name="Выходной сигнал",
                line=dict(color=COLORS[1], width=3),
                hovertemplate="Время: %{x:.3f}с<br>Амплитуда: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Зона установления (±5%)
        fig.add_trace(
            go.Scatter(
                x=ntime,
                y=upper,
                mode="lines",
                name="Верхняя граница (±5%)",
                line=dict(color=COLORS[2], width=1, dash="dash"),
                showlegend=False,
                hovertemplate="Время: %{x:.3f}с<br>Верхняя граница: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ntime,
                y=lower,
                mode="lines",
                name="Зона установления (±5%)",
                line=dict(color=COLORS[2], width=1, dash="dash"),
                fill="tonexty",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(COLORS[2])) + [0.2])}",
                hovertemplate="Время: %{x:.3f}с<br>Нижняя граница: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Время установления
        if (
            metrics["settling_time"] is not None
            and metrics["settling_time"] <= ntime[-1]
        ):
            fig.add_shape(
                type="line",
                x0=metrics["settling_time"],
                x1=metrics["settling_time"],
                y0=min(system_signal),
                y1=max(system_signal),
                line=dict(color=COLORS[3], width=2),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=metrics["settling_time"],
                y=max(system_signal),
                text=f"ts = {metrics['settling_time']:.3f}с",
                showarrow=False,
                yshift=10,
                row=1,
                col=1,
            )

        # Максимальное значение (перерегулирование)
        max_idx = np.argmax(system_signal_step)
        max_val = system_signal_step[max_idx]
        max_time = ntime[max_idx]

        fig.add_trace(
            go.Scatter(
                x=[max_time],
                y=[max_val],
                mode="markers",
                name=f"Максимум: {max_val:.3f}",
                marker=dict(color=COLORS[3], size=10, symbol="circle"),
                hovertemplate=f'Максимум<br>Время: {max_time:.3f}с<br>Значение: {max_val:.3f}<br>Перерегулирование: {metrics["overshoot"]:.1f}%<extra></extra>',
            ),
            row=1,
            col=1,
        )

        # Таблица с метриками
        table_data = [
            ["Перерегулирование", f'{metrics["overshoot"]:.2f}%'],
            [
                "Время установления",
                (
                    f'{metrics["settling_time"]:.3f}с'
                    if metrics["settling_time"]
                    else "N/A"
                ),
            ],
            [
                "Время нарастания",
                f'{metrics["rise_time"]:.3f}с' if metrics["rise_time"] else "N/A",
            ],
            [
                "Время пика",
                f'{metrics["peak_time"]:.3f}с' if metrics["peak_time"] else "N/A",
            ],
            ["Степень затухания", f'{metrics["damping_degree"]:.3f}'],
            ["Статическая ошибка", f'{metrics["static_error"]:.4f}'],
            ["Макс. отклонение", f'{metrics["maximum_deviation"]:.3f}'],
            ["Количество колебаний", f'{metrics["oscillation_count"]}'],
            ["IAE", f'{metrics["iae"]:.2f}'],
            ["ISE", f'{metrics["ise"]:.2f}'],
            ["ITAE", f'{metrics["itae"]:.2f}'],
            ["Индекс качества", f'{metrics["performance_index"]:.3f}'],
        ]

        # Создаем таблицу
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Метрика</b>", "<b>Значение</b>"],
                    fill_color=COLORS[0],
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=[
                        [row[0] for row in table_data],
                        [row[1] for row in table_data],
                    ],
                    fill_color=[["#E8F4FD", "#F8F9FA"] * len(table_data)],
                    font=dict(size=11),
                    align="center",
                    height=25,
                ),
            ),
            row=1,
            col=2,
        )

        # График ошибки во времени
        error_signal = control_signal_step - system_signal_step

        fig.add_trace(
            go.Scatter(
                x=ntime,
                y=error_signal,
                mode="lines",
                name="Ошибка регулирования",
                line=dict(color=COLORS[3], width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(COLORS[3])) + [0.3])}",
                hovertemplate="Время: %{x:.3f}с<br>Ошибка: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Нулевая линия для ошибки
        fig.add_shape(
            type="line",
            x0=ntime[0],
            x1=ntime[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="solid"),
            row=2,
            col=1,
        )

        # Настройка макета
        # Преобразуем figsize в пиксели (минимум 10 для каждого измерения)
        width_px = max(figsize[0] * 100, 10) if figsize[0] < 50 else figsize[0]
        height_px = max(figsize[1] * 100, 10) if figsize[1] < 50 else figsize[1]

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color="black")),
            width=width_px,
            height=height_px,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            template="plotly_white",
        )

        # Настройка осей
        fig.update_xaxes(
            title_text="Время, с",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Амплитуда",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_xaxes(
            title_text="Время, с",
            row=2,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Ошибка",
            row=2,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

        # Показываем график
        fig.show()

        return metrics

    def compare_systems(
        self,
        systems_data: Dict[str, Dict],
        signal_val: float,
        dt: float,
        figsize: tuple = (1800, 1200),
    ):
        """
        Сравнивает несколько систем управления на интерактивном графике.

        Args:
            systems_data (Dict[str, Dict]): Словарь с данными систем в формате:
                {
                    'Система 1': {
                        'control_signal': np.ndarray,
                        'system_signal': np.ndarray,
                        'time': np.ndarray
                    },
                    ...
                }
            signal_val (float): Значение сигнала, с которого начинается функция перехода.
            dt (float): Шаг дискретизации.
            figsize (tuple): Размер графика в пикселях, по умолчанию (1800, 1200).
        """
        # Создаем подграфики
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.7, 0.3],
            specs=[
                [{"type": "scatter"}, {"type": "table"}],
                [{"type": "scatter", "colspan": 2}, None],
            ],
            subplot_titles=(
                "Сравнение систем управления",
                "Сравнительная таблица",
                "Сравнение ошибок регулирования",
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        all_metrics = {}

        # Строим задающий сигнал (один для всех)
        first_system = list(systems_data.values())[0]
        control_step, _ = find_step_function(
            first_system["control_signal"],
            first_system["system_signal"],
            signal_val=signal_val,
        )
        ntime_ref = first_system["time"][: len(control_step)]

        fig.add_trace(
            go.Scatter(
                x=ntime_ref,
                y=control_step,
                mode="lines",
                name="Задающий сигнал",
                line=dict(color="black", width=3, dash="solid"),
                hovertemplate="Время: %{x:.3f}с<br>Амплитуда: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        for i, (system_name, data) in enumerate(systems_data.items()):
            color = COLORS[i % len(COLORS)]

            # Получаем метрики для каждой системы
            metrics = self.becnchmarking_one_step(
                data["control_signal"], data["system_signal"], signal_val, dt
            )
            all_metrics[system_name] = metrics

            # Находим переходную функцию
            control_step, system_step = find_step_function(
                data["control_signal"], data["system_signal"], signal_val=signal_val
            )

            # Обрезаем время под длину сигнала
            ntime = data["time"][: len(system_step)]

            # Строим сигналы систем
            fig.add_trace(
                go.Scatter(
                    x=ntime,
                    y=system_step,
                    mode="lines",
                    name=system_name,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"{system_name}<br>Время: %{{x:.3f}}с<br>Амплитуда: %{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Отмечаем время установления
            if (
                metrics["settling_time"] is not None
                and metrics["settling_time"] <= ntime[-1]
            ):
                fig.add_shape(
                    type="line",
                    x0=metrics["settling_time"],
                    x1=metrics["settling_time"],
                    y0=min(system_step),
                    y1=max(system_step),
                    line=dict(color=color, width=1.5, dash="dash"),
                    opacity=0.6,
                    row=1,
                    col=1,
                )

            # Добавляем ошибки на нижний график
            error_signal = control_step[: len(system_step)] - system_step
            fig.add_trace(
                go.Scatter(
                    x=ntime,
                    y=error_signal,
                    mode="lines",
                    name=f"{system_name} (ошибка)",
                    line=dict(color=color, width=2),
                    hovertemplate=f"{system_name}<br>Время: %{{x:.3f}}с<br>Ошибка: %{{y:.3f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Подготавливаем данные для таблицы сравнения
        table_data = []
        headers = [
            "Система",
            "Перерег.%",
            "Время уст.",
            "Время нар.",
            "Колеб.",
            "IAE",
            "Индекс кач.",
        ]

        for system_name, metrics in all_metrics.items():
            row = [
                system_name[:8] + "..." if len(system_name) > 8 else system_name,
                f'{metrics["overshoot"]:.1f}',
                (
                    f'{metrics["settling_time"]:.2f}'
                    if metrics["settling_time"]
                    else "N/A"
                ),
                f'{metrics["rise_time"]:.2f}' if metrics["rise_time"] else "N/A",
                f'{metrics["oscillation_count"]}',
                f'{metrics["iae"]:.1f}',
                f'{metrics["performance_index"]:.2f}',
            ]
            table_data.append(row)

        # Создаем таблицу сравнения
        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#2E86AB",
                    font=dict(color="white", size=10),
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[["#F8F9FA", "#E8F4FD"] * len(table_data)],
                    align="center",
                    font=dict(size=9),
                    height=25,
                ),
            ),
            row=1,
            col=2,
        )

        # Настройка осей и макета
        fig.update_xaxes(
            title_text="Время, с",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Амплитуда",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )

        fig.update_xaxes(
            title_text="Время, с",
            row=2,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Ошибка регулирования",
            row=2,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )

        # Добавляем нулевую линию на график ошибок
        fig.add_shape(
            type="line",
            x0=ntime[0],
            x1=ntime[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=1),
            opacity=0.5,
            row=2,
            col=1,
        )

        # Обновляем общий макет
        # Преобразуем figsize в пиксели (минимум 10 для каждого измерения)
        width_px = max(figsize[0] * 100, 10) if figsize[0] < 50 else figsize[0]
        height_px = max(figsize[1] * 100, 10) if figsize[1] < 50 else figsize[1]

        fig.update_layout(
            height=height_px,
            width=width_px,
            title=dict(
                text="Сравнительный анализ систем управления",
                x=0.5,
                font=dict(size=16, color="black"),
            ),
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            template="plotly_white",
        )

        fig.show()

        return all_metrics

    def generate_report(
        self,
        control_signal: np.ndarray,
        system_signal: np.ndarray,
        signal_val: float,
        dt: float,
        system_name: str = "Система управления",
    ) -> str:
        """
        Генерирует текстовый отчет о качестве системы управления.

        Args:
            control_signal (numpy.ndarray): Сигнал управления системы.
            system_signal (numpy.ndarray): Сигнал системы, на которую воздействует управление.
            signal_val (float): Значение сигнала, с которого начинается функция перехода.
            dt (float): Шаг дискретизации.
            system_name (str): Название системы для отчета.

        Returns:
            str: Форматированный текстовый отчет.
        """
        metrics = self.becnchmarking_one_step(
            control_signal, system_signal, signal_val, dt
        )

        # Форматируем значения для отчета
        settling_time_str = (
            f"{metrics['settling_time']:>8.3f} с"
            if metrics["settling_time"] is not None
            else "     N/A"
        )
        rise_time_str = (
            f"{metrics['rise_time']:>8.3f} с"
            if metrics["rise_time"] is not None
            else "     N/A"
        )
        peak_time_str = (
            f"{metrics['peak_time']:>8.3f} с"
            if metrics["peak_time"] is not None
            else "     N/A"
        )

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ОТЧЕТ О КАЧЕСТВЕ УПРАВЛЕНИЯ               ║
╠══════════════════════════════════════════════════════════════╣
║ Система: {system_name:<47} ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 📊 ОСНОВНЫЕ ПОКАЗАТЕЛИ КАЧЕСТВА:                            ║
║                                                              ║
║ • Перерегулирование:     {metrics['overshoot']:>8.2f}%                    ║
║ • Время установления:    {settling_time_str}                   ║
║ • Время нарастания:      {rise_time_str}                   ║
║ • Время пика:            {peak_time_str}                   ║
║ • Степень затухания:     {metrics['damping_degree']:>8.3f}                      ║
║ • Статическая ошибка:    {metrics['static_error']:>8.4f}                      ║
║ • Макс. отклонение:      {metrics['maximum_deviation']:>8.3f}                      ║
║ • Количество колебаний:  {metrics['oscillation_count']:>8}                        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ 🔢 ИНТЕГРАЛЬНЫЕ КРИТЕРИИ:                                   ║
║                                                              ║
║ • IAE (Интегр. абс. ошибка):  {metrics['iae']:>10.2f}                ║
║ • ISE (Интегр. кв. ошибка):   {metrics['ise']:>10.2f}                ║
║ • ITAE (Взвеш. по времени):   {metrics['itae']:>10.2f}                ║
║ • Индекс качества:            {metrics['performance_index']:>10.3f}                ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ 🎯 ОЦЕНКА КАЧЕСТВА:                                         ║
║                                                              ║"""

        # Добавляем оценки качества
        overshoot_rating = (
            "Отлично"
            if metrics["overshoot"] < 5
            else (
                "Хорошо"
                if metrics["overshoot"] < 15
                else ("Удовлетворительно" if metrics["overshoot"] < 25 else "Плохо")
            )
        )
        settling_rating = (
            "Быстро"
            if metrics["settling_time"] and metrics["settling_time"] < 2
            else (
                "Средне"
                if metrics["settling_time"] and metrics["settling_time"] < 5
                else "Медленно"
            )
        )
        error_rating = (
            "Минимальная"
            if abs(metrics["static_error"]) < 0.01
            else ("Малая" if abs(metrics["static_error"]) < 0.05 else "Значительная")
        )

        report += f"""
║ • Перерегулирование:     {overshoot_rating:<15}                    ║
║ • Быстродействие:        {settling_rating:<15}                    ║
║ • Точность:              {error_rating:<15}                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """

        return report
