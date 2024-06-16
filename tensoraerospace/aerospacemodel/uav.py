import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.f16.nonlinear.utils import output2dict
from tensoraerospace.aerospacemodel.utils.constant import (
    state_to_latex_eng,
    state_to_latex_rus,
)


class LongitudinalUAV(ModelBase):
    """
    UAV model в продольном канале управления
    
    
    Args:
        x0 (_type_): Начальное состояние объекта управления
        number_time_steps (_type_): Количество временных шагов
        selected_state_output (_type_, optional): Выбранные состояние объекта управления. Defaults to None.
        t0 (int, optional): Начальное время. Defaults to 0.
        dt (float, optional): Частота дискретизации. Defaults to 0.01.
            
    Пространство действий:
        - ele: руль высоты [град]


    Пространство состояний:
        - u - Продольная скорость ЛА [м/с]
        - w - Нормальная скорость ЛА [м/с]
        - q - Угловая скорость Тангажа [град/с]
        - theta - Тангаж [град]

    Пространство выхода:
        - u - Продольная скорость ЛА [м/с]
        - w - Нормальная скорость ЛА [м/с]
        - q - Угловая скорость Тангажа [град/с]
        - theta - Тангаж [град]
    """

    def __init__(self, x0, number_time_steps, selected_state_output=None, t0=0, dt: float = 0.01):
        super().__init__(x0, selected_state_output, t0, dt)

        self.discretisation_time = dt

        # Selected data for the system
        self.selected_states = ["u", "w", "q", "theta"]
        self.selected_output = ["u", "w", "q", "theta"]
        self.list_state = self.selected_states
        self.selected_input = ["ele", ]
        self.control_list = self.selected_input

        if self.selected_state_output:
            self.selected_state_index = [self.list_state.index(val) for val in self.selected_state_output]

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele
        # Limitations of the system
        self.input_magnitude_limits = [25, ]
        self.input_rate_limits = [60, ]

        # Store the number of inputs, states and outputs
        self.number_inputs = len(self.selected_input)
        self.number_outputs = len(self.selected_output)
        self.number_states = len(self.selected_states)
        self.output_history = []
        # Original matrices of the system
        self.A = None
        self.B = None
        self.C = None
        self.D = None

        # Processed matrices of the system
        self.filt_A = None
        self.filt_B = None
        self.filt_C = None
        self.filt_D = None

        self.initialise_system(x0, number_time_steps)

    def import_linear_system(self):
        """Сохраненные линеаризованные матрицы
        """
        self.A = np.array([
            [-0.1982, 0.593, 1.245, -9.779],
            [-0.7239, -3.9848, 18.7028, -0.6286],
            [0.3537, -5.5023, -5.4722, 0],
            [0, 0, 1, 0],
        ])

        self.B = np.array([
            [0.2281],
            [-4.6830],
            [-36.1341],
            [0]
        ])

        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.D = np.array([
            [0],
            [0],
            [0],
            [0],
        ])

    def initialise_system(self, x0, number_time_steps):
        """Инициализация системы

        Args:
            x0 (_type_): Начальное состояние объекта управления
            number_time_steps (_type_): количество временных шагов в итерации
        """

        # Import the stored system
        self.import_linear_system()

        # Store the number of time steps
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Discretise the system according to the discretisation time
        (self.filt_A, self.filt_B, self.filt_C, self.filt_D, _) = cont2discrete((self.A, self.B, self.C,
                                                                                 self.D),
                                                                                self.discretisation_time)

        self.store_states = np.zeros((self.number_states, self.number_time_steps + 1))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_outputs = np.zeros((self.number_outputs, self.number_time_steps))

        self.x0 = x0
        self.xt = x0
        self.store_states[:, self.time_step] = np.reshape(self.xt, [-1, ])

    def run_step(self, ut_0: np.ndarray) -> np.ndarray:
        """Выполняет один временной шаг итерации.

        Args:
            ut_0 (np.ndarray): Вектор управления

        Returns:
            xt1 (np.ndarray): Состояние объекта управления на шаге t+1
        """
        if self.time_step != 0:
            ut_1 = self.store_input[:, self.time_step - 1]
        else:
            ut_1 = ut_0
        ut = [0, ]
        for i in range(self.number_inputs):
            ut[i] = max(min(max(min(ut_0[i],
                                    np.reshape(
                                        np.array([ut_1[i] + self.input_rate_limits[i] * self.discretisation_time]),
                                        [-1, 1])),
                                np.reshape(np.array([ut_1[i] - self.input_rate_limits[i] * self.discretisation_time]),
                                           [-1, 1])),
                            np.array([[self.input_magnitude_limits[i]]])),
                        - np.array([[self.input_magnitude_limits[i]]]))
        ut = np.array(ut)
        self.xt1 = np.matmul(self.filt_A, np.reshape(self.xt, [-1, 1])) + np.matmul(self.filt_B,
                                                                                    np.reshape(ut, [-1, 1]))
        output = np.matmul(self.filt_C, np.reshape(self.xt, [-1, 1])) + np.matmul(self.filt_D, np.reshape(ut, [-1, 1]))

        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])
        self.store_outputs[:, self.time_step] = np.reshape(output, [output.shape[0]])
        self.store_states[:, self.time_step + 1] = np.reshape(self.xt1, [self.xt1.shape[0]])

        self.update_system_attributes()
        if self.selected_state_output:
            return np.array(self.xt1[self.selected_state_index])
        return np.array(self.xt1)

    def update_system_attributes(self):
        """Атрибуты, которые меняются с каждым временным шагом, обновляются
        """
        self.xt = self.xt1
        self.time_step += 1

    def get_state(self, state_name: str, to_deg: bool = False, to_rad: bool = False):
        """
        Получить массив состояния

        Args:
            state_name: Название состояния
            to_deg: Конвертировать в градусы
            to_rad: Конвертировать в радианы

        Returns:
            Массив истории выбранного состояния

        Пример:

        >>> state_hist = model.get_state('alpha', to_deg=True)

        """
        if state_name == 'wz':
            state_name = 'q'
        if state_name == 'wx':
            state_name = 'p'
        if state_name == 'wy':
            state_name = 'r'
        if state_name not in self.selected_states:
            raise Exception(f"{state_name} нет в списке состояний, доступные {self.selected_states}")
        index = self.selected_states.index(state_name)
        if to_deg:
            return np.rad2deg(self.store_states[index][:self.number_time_steps - 1])
        if to_rad:
            return np.deg2rad(self.store_states[index][:self.number_time_steps - 1])
        return self.store_states[index][:self.number_time_steps - 1]

    def get_control(self, control_name: str, to_deg: bool = False, to_rad: bool = False):
        """
        Получить массив сигнала управления

        Args:
            control_name: Название сигнала управления
            to_deg: Конвертировать в градусы

        Returns:
            Массив истории выбранного сигнала управления

        Пример:

        >>> state_hist = model.get_control('stab', to_deg=True)
        """
        if control_name in ['stab', 'ele']:
            control_name = 'ele'
        if control_name in ['rud', 'dir']:
            control_name = 'rud'
        if control_name not in self.selected_input or control_name not in ["ele", "ail", "rud"]:
            raise Exception(f"{control_name} нет в списке сигналов управления, доступные {self.selected_input}")
        index = self.selected_input.index(control_name)
        if to_deg:
            return np.rad2deg(self.store_input[index])[:self.number_time_steps - 1]
        if to_rad:
            return np.deg2rad(self.store_states[index][:self.number_time_steps - 1])
        return self.store_input[index][:self.number_time_steps - 1]

    def get_output(self, state_name: str, to_deg: bool = False, to_rad: bool = False):
        self.output_history = output2dict(self.store_outputs, self.selected_output)
        if to_deg:
            return np.rad2deg(self.state_history[state_name][:self.time_step - 1])
        if to_rad:
            return np.deg2rad(self.state_history[state_name][:self.time_step - 1])
        return self.output_history[state_name][:self.time_step - 1]

    def plot_output(self, output_name: str, time: np.ndarray, lang: str = 'rus', to_deg: bool = False,
                    to_rad: bool = False, figsize: tuple = (10, 10)):
        if to_rad and to_deg:
            raise Exception("Неверно указано форматирование, укажите один. to_rad или to_deg.")
        if output_name not in self.list_state:
            raise Exception(f"{output_name} нет в списке сигналов управления")
        if not self.control_history:
            self.control_history = output2dict(self.store_outputs, self.selected_output)
        state_hist = self.get_output(output_name, to_deg, to_rad)
        if output_name == 'u':
            state_hist *= 1.94384
        if lang == 'rus':
            label = state_to_latex_rus[output_name]
            label_time = 't, c'
        else:
            label = state_to_latex_eng[output_name]
            label_time = 't, sec.'
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.plot(time[:self.time_step - 1], state_hist, label=label)
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)
        return fig
