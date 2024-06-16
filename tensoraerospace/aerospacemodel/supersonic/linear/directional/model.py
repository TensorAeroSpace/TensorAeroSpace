import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.f16.nonlinear.utils import output2dict
from tensoraerospace.aerospacemodel.utils import state_to_latex_eng, state_to_latex_rus


class DirectionalSuperSonic(ModelBase):
    """
    Сверхзвуковой самолет в изолированном боковом канале.

    Пространство действий:
        * ele: руль высоты [град]


    Пространство состояний:
        * $v$ - Боковая скорость ЛА [м/с]
        * $p$ - Угловая скорость Крена [град/с]
        * $r$ - Угловая скорость Рысканью [град/с]
        * $\varphi$ - угол Крен [град]
        * $\varpsi$ - угол Рысканья [град]



    Пространство выхода:
        * $p$ - Угловая скорость Крена [град/с]
        * $r$ - Нормальная скорость Рысканью [м/с]
        * $\beta$ - Угол скольжения [град]
        * $\varphi$ - угол Крен [град]
        * $\varpsi$ - угол Рысканья [град]

    """

    def __init__(self, x0, number_time_steps, selected_state_output=None, t0=0, dt: float = 0.01):
        super().__init__(x0, selected_state_output, t0, dt)
        self.discretisation_time = dt

        # Selected data for the system
        self.selected_states = ["v", "p", "r", "phi", "psi"]
        self.selected_output = ["p", "r", "beta", "phi", "psi"]
        self.list_state = self.selected_output
        self.selected_input = ["ail", "rud"]
        self.control_list = self.selected_input

        if self.selected_state_output:
            for val in self.selected_state_output:
                self.selected_state_index = self.list_state.index(val)

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele
        # Limitations of the system
        self.input_magnitude_limits = [25, 21.5]
        self.input_rate_limits = [60, 90]

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
        """
        Retrieves the stored linearised matrices obtained from Matlab
        :return:
        """
        self.A = np.array([
            [-0.0773, 0.9723, 0.6484, 8.1749, 6.1553],
            [-0.0241, 0.9304, 0.7414, -0.4568, 1.9194],
            [-0.0001, -0.0604, -0.0815, -0.0022, 0.0094],
            [0, 1, 0.1899, 0, 0],
            [0, 0, 1.0179, 0, 0],
        ])

        self.B = np.array([
            [-1.3418, 1.1055],
            [-1.3649, 0.0310],
            [-0.1677, -0.1012],
            [0, 0],
            [0, 0]
        ])

        self.C = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0.0125, 0, 0, 0.2377, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],

        ])

        self.D = np.array([
            [0],
            [0],
            [0],
            [0],
            [0],
        ])

    def initialise_system(self, x0, number_time_steps):
        """
        Initialises the F-16 aircraft dynamics
        :param x0: the initial states
        :param number_time_steps: the number of time steps within an iteration
        :return:
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

    def run_step(self, ut_0: np.ndarray):
        """
        Runs one time step of the iteration.
        :param ut: input to the system
        :return: xt1 --> the next time step state
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
        output = np.matmul(self.filt_C, np.reshape(self.xt, [-1, 1]))

        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])
        self.store_outputs[:, self.time_step] = np.reshape(output, [output.shape[0]])
        self.store_states[:, self.time_step + 1] = np.reshape(self.xt1, [self.xt1.shape[0]])
        self.update_system_attributes()
        if self.selected_state_output:
            return np.array(self.xt1[self.selected_state_index])
        return np.array(self.xt1)

    def update_system_attributes(self):
        """
        The attributes that change with every time step are updated
        :return:
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
        if output_name == 'wz':
            output_name = 'q'
        if output_name == 'wx':
            output_name = 'p'
        if output_name == 'wy':
            output_name = 'r'
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
