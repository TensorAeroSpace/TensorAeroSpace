import numpy as np
from .function import find_step_function, overshoot, settling_time, damping_degree, static_error, get_lower_upper_bound
from matplotlib import pyplot as plt


class ControlBenchmark:
    """
    Класс для проведения оценки системы управления и построения графиков.
    """
    def becnchmarking_one_step(self, control_signal:np.ndarray, system_signal:np.ndarray, signal_val:float, dt)->dict:
        """
        Оценивает систему управления на одном шаге и возвращает результаты в виде словаря.
        
        Args:
            control_signal: numpy.ndarray
                Сигнал управления системы.
            system_signal: numpy.ndarray
                Сигнал системы, на которую воздействует управление.
            signal_val: float
                Значение сигнала, с которого начинается функция перехода.
            dt: float
                Шаг дискретизации.
        
        Returns:
            dict:
                Словарь с результатами оценки системы управления:
                {
                    "overshoot": перерегулирование,
                    "settling_time": время установления,
                    "damping_degree": степень затухания,
                    "static_error": статическая ошибка
                }
        """
        control_signal, system_signal = find_step_function(control_signal, system_signal, signal_val=signal_val)
        overshooting = overshoot(control_signal, system_signal)
        cnt_time = settling_time(control_signal, system_signal)
        damp = damping_degree(system_signal)
        static_err = static_error(control_signal[cnt_time:], system_signal[cnt_time:])
        return {
            "overshoot": overshooting, 
            "settling_time": cnt_time * dt, 
            "damping_degree": damp, 
            "static_error": static_err
        }

    def plot(self, control_signal:np.ndarray, system_signal:np.ndarray, signal_val:float, dt:float, tps:np.ndarray, figsize:tuple=(15,9)):
        """
        Строит график сигналов системы управления и системы, а также границ и времени установления.
        
        Args:
            control_signal: numpy.ndarray
                Сигнал управления системы.
            system_signal: numpy.ndarray
                Сигнал системы, на которую воздействует управление.
            signal_val: float
                Значение сигнала, с которого начинается функция перехода.
            dt: float
                Шаг дискретизации.
            tps: numpy.ndarray
                Массив временных меток.
            figsize: tuple, optional (default: (15, 9))
                Размер графика.
        """
        benhc = self.becnchmarking_one_step(control_signal, system_signal, signal_val, dt)
        control_signal, system_signal = find_step_function(control_signal, system_signal, signal_val=signal_val)
        lower, upper = get_lower_upper_bound(control_signal)
        ntime = tps[:len(control_signal)]
        plt.figure(figsize=figsize)
        plt.plot(ntime, control_signal)
        plt.plot(ntime, system_signal)
        plt.hlines(lower, xmin=ntime[0], xmax=ntime[-1],  color='r')
        plt.hlines(upper, xmin=ntime[0], xmax=ntime[-1],  color='r')
        plt.vlines(x=benhc['settling_time'], ymin=np.min(system_signal), ymax=np.max(system_signal), color='r')