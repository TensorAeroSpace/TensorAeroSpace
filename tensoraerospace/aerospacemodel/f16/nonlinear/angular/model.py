import os

import matlab.engine
import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase


class AngularF16(ModelBase):
    """

    Объект управления высокомоневренный самолет F-16 ✈ в угловых координатах.

    Пространство действий:
        * stab_act: руль высоты [рад]
        * ail_act: элероны [рад]
        * dir_act: руль направления [рад]

    Пространство состояний:
        * alpha: угол атаки [рад]
        * beta: угол скольжения [рад]
        * wx: угловая скорость крена [рад/с]
        * wy: угловая скорость рысканья [рад/с]
        * wz: угловая скорость тангажа [рад/с]
        * gamma: крен [рад]
        * psi: рысканье [рад]
        * theta: тангаж [рад]
        * stab: полжение руля высоты [рад]
        * ail: полжение элеронов [рад]
        * dir: положение руля направления [рад]
        * dstab: угловая скорость руля высоты [рад/с]
        * dail: угловая скорость элеронов [рад/с]
        * ddir: угловая скорость руля направления [рад/с]

    Пример использования:

    >>> from aerospacemodel.model.f16.nonlinear.angular import initial_state
    >>> model = AngularF16(initial_state)
    >>> x_t = model.run_step([ [0], [0], [0] ])

    Args:
        x0: Начальное состояние
        t0: (Optional) Начальное время
        x0: (Optional) Шаг дискетизации

    """

    def __init__(self, x0, selected_state_output=None, t0=0, dt: float = 0.01):
        super(AngularF16, self).__init__(x0, selected_state_output, t0, dt)
        self.matlab_files_path = os.path.join(os.path.dirname(__file__), 'matlab_code')
        self.eng = matlab.engine.start_matlab()  # Запуск экземпляр Matlab
        self.eng.addpath(self.matlab_files_path)
        self.list_state = ['alpha', 'beta', 'wx', 'wy', 'wz', 'gamma', 'psi', 'theta', 'stab', 'dstab', 'ail', 'dail',
                           'dir', 'ddir']
        self.control_list = ['stab', 'ail', 'dir']
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

        >>> stab_act, ail_act, dir_act = 0,0,0
        >>> [
        >>>    [stab_act],
        >>>    [ail_act],
        >>>    [dir_act]
        >>> ]

        Args:
            u: управляющий сигнал

        Returns:
            Состояние объекта управления

        Пример использования:
        >>> from aerospacemodel.model.f16.nonlinear.angular import initial_state
        >>> model = AngularF16(initial_state)
        >>> xt = model.run_step([ [0], [0], [0] ])
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
