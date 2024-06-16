import os

import matlab.engine
import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase


class LongitudinalF16(ModelBase):
    r"""
    Cамолет F-16 ✈ в изолированном продольном канале.

    Пространство действий:
        * stab_act: руль высоты [рад]


    Пространство состояний:
        * alpha:  угол атаки  [рад]
        * wz: угловая скорость тангажа [рад/с]
        * stab: полжение руля высоты [рад]
        * dstab: угловая скорость руля высоты [рад/с]


    Пример использования:

    >>> model = LongitudinalF16(initial_state)
    >>> x_t = model.run_step([ [0], ])

    Args:
        x0: Начальное состояние
        t0: (Optional) Начальное время
        x0: (Optional) Шаг дискетизации
    """

    def __init__(self, x0, selected_state_output=None, t0=0, dt: float = 0.01):
        super(LongitudinalF16, self).__init__(x0, selected_state_output, t0, dt)
        self.matlab_files_path = os.path.join(os.path.dirname(__file__), 'matlab_code')
        self.eng = matlab.engine.start_matlab()  # Запуск экземпляр Matlab
        self.eng.addpath(self.matlab_files_path)
        self.list_state = ['alpha', 'wz', 'stab', 'dstab']
        self.control_list = ['stab', ]
        self.action_space_length = len(self.control_list)
        self.param = self.eng.airplane_parameters()  # Получаем параметры объекта управления
        self.x_history = [x0]
        if self.selected_state_output:
            for val in self.selected_state_output:
                self.selected_state_index = self.list_state.index(val)

    def get_param(self):
        """
            Получить параметры объекта управления

        Returns:
            Параметры объекта управления
        """
        return self.param

    def set_param(self, new_param):
        """
            Установка новых параметров объекта управления

         Args:
            new_param: параметры объекта управления
        """
        self.param = new_param

    def run_step(self, u: matlab.double):
        """
        Расчет состояния объекта управления

        Управляющий сигнал имеет вид:

        >>> stab_act = 0
        >>> [
        >>>    [stab_act],
        >>> ]

        Args:
            u: управляющий сигнал

        Returns:
            Состояние объекта управления

        Пример использования:

        >>> from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import initial_state, LongitudinalF16
        >>> model = LongitudinalF16(initial_state)
        >>> x_t = model.run_step([ [0], ])
        """
        if not isinstance(u, matlab.double):
            u = matlab.double(u)
        if len(list(u)) != self.action_space_length:
            raise Exception(
                "Размерность управляющего вектора задана неверно." +
                f" Текущее значение {len(list(u))}, не соответсвует {self.action_space_length}")
        x_t = self.eng.step(self.x_history[-1], self.dt, u, self.t0, self.time_step, self.param)
        self.x_history.append(x_t)
        self.u_history.append(u)
        self.time_step += 1
        if self.selected_state_output:
            return np.array(x_t[self.selected_state_index])
        return np.array(x_t)
