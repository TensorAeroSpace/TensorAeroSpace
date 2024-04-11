import numpy as np
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
import torch 

class PIDController:
    """
    Класс PIDController реализует ПИД-регулятор для систем управления.

    Этот класс предназначен для создания и использования ПИД-регулятора в системах
    автоматического управления. ПИД-регулятор использует пропорциональный (P), интегральный (I)
    и дифференциальный (D) компоненты для вычисления управляющего сигнала.

    Атрибуты:
        kp (float): Коэффициент пропорциональной составляющей.
        ki (float): Коэффициент интегральной составляющей.
        kd (float): Коэффициент дифференциальной составляющей.
        dt (float): Шаг времени (разница времени между последовательными обновлениями).
        integral (float): Накопленное значение интегральной составляющей.
        prev_error (float): Предыдущее значение ошибки для вычисления дифференциальной составляющей.

    Методы:
        update(setpoint, measurement): Вычисляет и возвращает управляющий сигнал на основе заданного значения
                                       и текущего измерения.

    Args:
        kp (float): Коэффициент пропорциональной составляющей.
        ki (float): Коэффициент интегральной составляющей.
        kd (float): Коэффициент дифференциальной составляющей.
        dt (float): Шаг времени (разница времени между последовательными обновлениями).

    Пример:
        >>> pid = PIDController(0.1, 0.01, 0.05, 1)
        >>> control_signal = pid.update(10, 7)
    """

    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, setpoint, measurement):
        """
        Вычисляет и возвращает управляющий сигнал на основе заданного значения (setpoint) и текущего измерения.

        Этот метод использует текущее измерение и заданное значение для вычисления ошибки,
        затем применяет ПИД-алгоритм для вычисления управляющего сигнала.

        Args:
            setpoint (float): Заданное значение, к которому должна стремиться система.
            measurement (float): Текущее измеренное значение.

        Returns:
            float: Управляющий сигнал, вычисленный на основе ПИД-регулятора.

        Пример:
            >>> pid = PIDController(0.1, 0.01, 0.05, 1)
            >>> control_signal = pid.update(10, 7)
            >>> print(control_signal)
        """
        error = setpoint - measurement
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output
