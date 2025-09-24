import json
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase


class LongitudinalF16(ModelBase):
    """Линеаризованная продольная динамика F‑16 в пространстве состояний.

    Модель описывает продольный канал ЛА с входом по отклонению стабилизатора и
    выходами по углам/скоростям. Матрицы состояния загружаются из подготовленных
    файлов Matlab и редуцируются до выбранных переменных, после чего система
    дискретизируется с шагом ``dt``.

    Состояния (порядок internal-модели):
      - ``theta`` — тангаж [рад]
      - ``alpha`` — угол атаки [рад]
      - ``q`` — угловая скорость по тангажу [рад/с]
      - ``ele`` — положение руля высоты [рад]

    Управление:
      - ``ele`` — отклонение стабилизатора [рад]

    Args:
        x0 (np.ndarray | list[float]): Начальное состояние модели в порядке
            internal-модели (см. список выше).
        number_time_steps (int): Количество шагов моделирования.
        selected_state_output (list[str] | None): Имена состояний, которые
            возвращаются наружу (редуцированный вектор состояний). Если ``None``,
            возвращается полный вектор internal-модели.
        t0 (float): Начальное время, сек.
        dt (float): Шаг дискретизации, сек.

    Attributes:
        selected_states (list[str]): Список состояний internal-модели.
        selected_output (list[str]): Список выходов.
        selected_input (list[str]): Список управляющих воздействий.
        input_magnitude_limits (list[float]): Ограничения по величине управления.
        input_rate_limits (list[float]): Ограничения по скорости изменения управления.
        A, B, C, D (np.ndarray | None): Непрерывные матрицы исходной системы.
        filt_A, filt_B, filt_C, filt_D (np.ndarray | None): Отфильтрованные и
            дискретизованные матрицы редуцированной системы.
        store_states, store_input, store_outputs (np.ndarray): История состояний,
            входов и выходов за горизонт моделирования.

    Notes:
        - Матрицы загружаются из каталога ``../data`` относительно файла модели.
        - Дискретизация выполняется с помощью ``scipy.signal.cont2discrete``.
        - Единицы измерения: углы и угловые скорости внутри модели — в радианах.
    """

    def __init__(
        self, x0, number_time_steps, selected_state_output=None, t0=0, dt: float = 0.01
    ):
        super().__init__(x0, selected_state_output, t0, dt)
        self.discretisation_time = dt
        self.folder = os.path.join(os.path.dirname(__file__), "../data")

        # Selected data for the system
        self.selected_states = ["theta", "alpha", "q", "ele"]
        self.selected_output = ["theta", "alpha", "q", "nz"]
        self.list_state = self.selected_output
        self.selected_input = [
            "ele",
        ]
        self.control_list = self.selected_input

        if self.selected_state_output:
            self.selected_state_index = [
                self.list_state.index(val) for val in self.selected_state_output
            ]

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele
        # Limitations of the system
        self.input_magnitude_limits = [
            25,
        ]
        self.input_rate_limits = [
            60,
        ]

        # Store the number of inputs, states and outputs
        self.number_inputs = len(self.selected_input)
        self.number_outputs = len(self.selected_output)
        self.number_states = len(self.selected_states)

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

    def import_linear_system(self) -> None:
        """Загружает линеаризованные матрицы состояния из Matlab-файлов."""
        x = loadmat(self.folder + "/A.mat")
        self.A = x["A_lo"]

        x = loadmat(self.folder + "/B.mat")
        self.B = x["B_lo"]

        x = loadmat(self.folder + "/C.mat")
        self.C = x["C_lo"]

        x = loadmat(self.folder + "/D.mat")
        self.D = x["D_lo"]

    def simplify_system(self) -> None:
        """
        Редуцирует систему к выбранным состояниям/выходам и
        формирует фильтрованные матрицы.
        """

        # Create dictionaries with the information from the system
        states_rows = self.create_dictionary("states")
        selected_rows_states = np.array(
            [states_rows[state] for state in self.selected_states]
        )
        output_rows = self.create_dictionary("output")
        selected_rows_output = np.array(
            [output_rows[output] for output in self.selected_output]
        )
        input_rows = self.create_dictionary("input")
        selected_rows_input = np.array(
            [input_rows[input_var] for input_var in self.selected_input]
        )

        # Create the new system and initial condition
        self.filt_A = self.A[selected_rows_states[:, None], selected_rows_states]
        self.filt_B = (
            self.A[selected_rows_states[:, None], 12 + selected_rows_input]
            + self.B[selected_rows_states[:, None], selected_rows_input]
        )
        self.filt_C = self.C[selected_rows_output[:, None], selected_rows_states]
        self.filt_D = (
            self.C[selected_rows_output[:, None], 12 + selected_rows_input]
            + self.D[selected_rows_output[:, None], selected_rows_input]
        )

    def create_dictionary(self, file_name: str) -> dict[str, int]:
        """Создаёт словарь индексов по именам величин (state/input/output).

        Args:
            file_name (str): Имя файла-ключа (``states``, ``input``, ``output``).

        Returns:
            dict[str, int]: Отображение имени величины в индекс строки/столбца.
        """
        full_name = self.folder + "/keySet_" + file_name + ".txt"
        with open(full_name, "r", encoding="utf-8") as f:
            keySet = json.loads(f.read())
        rows = dict(zip(keySet, range(len(keySet))))
        return rows

    def initialise_system(self, x0: np.ndarray, number_time_steps: int) -> None:
        """Инициализирует систему, дискретизирует и выделяет буферы истории.

        Args:
            x0 (np.ndarray): Начальное состояние.
            number_time_steps (int): Горизонт моделирования.
        """
        # Import the stored system
        self.import_linear_system()

        # Simplify the system with the chosen states
        self.simplify_system()

        # Store the number of time steps
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Discretise the system according to the discretisation time
        (self.filt_A, self.filt_B, self.filt_C, self.filt_D, _) = cont2discrete(
            (self.filt_A, self.filt_B, self.filt_C, self.filt_D),
            self.discretisation_time,
        )

        self.store_states = np.zeros((self.number_states, self.number_time_steps + 1))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_outputs = np.zeros((self.number_outputs, self.number_time_steps))

        self.x0 = x0
        self.xt = x0
        self.store_states[:, self.time_step] = np.reshape(
            self.xt,
            [
                -1,
            ],
        )

    def run_step(self, ut_0: np.ndarray) -> np.ndarray:
        """Выполняет один шаг эволюции системы с ограничениями по управлению.

        Args:
            ut_0 (np.ndarray): Вектор управления на текущем шаге (рад).

        Returns:
            np.ndarray: Состояние на следующем шаге ``x[t+1]``.
        """
        if self.time_step != 0:
            ut_1 = self.store_input[:, self.time_step - 1]
        else:
            ut_1 = ut_0
        ut = [
            0,
        ]
        for i in range(self.number_inputs):
            ut[i] = max(
                min(
                    max(
                        min(
                            ut_0[i],
                            np.reshape(
                                np.array(
                                    [
                                        ut_1[i]
                                        + self.input_rate_limits[i]
                                        * self.discretisation_time
                                    ]
                                ),
                                [-1, 1],
                            ),
                        ),
                        np.reshape(
                            np.array(
                                [
                                    ut_1[i]
                                    - self.input_rate_limits[i]
                                    * self.discretisation_time
                                ]
                            ),
                            [-1, 1],
                        ),
                    ),
                    np.array([[self.input_magnitude_limits[i]]]),
                ),
                -np.array([[self.input_magnitude_limits[i]]]),
            )
        ut = np.array(ut)
        self.xt1 = np.matmul(self.filt_A, np.reshape(self.xt, [-1, 1])) + np.matmul(
            self.filt_B, np.reshape(ut, [-1, 1])
        )
        output = np.matmul(self.filt_C, np.reshape(self.xt, [-1, 1]))
        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])
        self.store_outputs[:, self.time_step] = np.reshape(output, [output.shape[0]])
        self.store_states[:, self.time_step + 1] = np.reshape(
            self.xt1, [self.xt1.shape[0]]
        )
        self.update_system_attributes()
        if self.selected_state_output:
            return np.array(self.xt1[self.selected_state_index])
        return np.array(self.xt1)

    def update_system_attributes(self) -> None:
        """Обновляет текущее состояние и внутренний таймер модели."""
        self.xt = self.xt1
        self.time_step += 1

    def get_state(
        self, state_name: str, to_deg: bool = False, to_rad: bool = False
    ) -> np.ndarray:
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
        if state_name == "wz":
            state_name = "q"
        if state_name == "wx":
            state_name = "p"
        if state_name == "wy":
            state_name = "r"
        if state_name not in self.selected_states:
            raise ValueError(
                f"{state_name} нет в списке состояний, доступные {self.selected_states}"
            )
        index = self.selected_states.index(state_name)
        if to_deg:
            return np.rad2deg(self.store_states[index][: self.number_time_steps - 1])
        if to_rad:
            values_deg = self.store_states[index][: self.number_time_steps - 1]
            return np.deg2rad(values_deg)
        return self.store_states[index][: self.number_time_steps - 1]

    def get_control(
        self, control_name: str, to_deg: bool = False, to_rad: bool = False
    ) -> np.ndarray:
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
        if control_name in ["stab", "ele"]:
            control_name = "ele"
        if control_name in ["rud", "dir"]:
            control_name = "rud"
        allowed_controls = [
            "ele",
            "ail",
            "rud",
        ]
        if (
            control_name not in self.selected_input
            or control_name not in allowed_controls
        ):
            message = (
                f"{control_name} нет в списке сигналов управления, доступные "
                f"{self.selected_input}"
            )
            raise ValueError(message)
        index = self.selected_input.index(control_name)
        if to_deg:
            rad_deg_input = np.rad2deg(self.store_input[index])
            return rad_deg_input[: self.number_time_steps - 1]
        if to_rad:
            values_rad = self.store_states[index][: self.number_time_steps - 1]
            return np.deg2rad(values_rad)
        return self.store_input[index][: self.number_time_steps - 1]
