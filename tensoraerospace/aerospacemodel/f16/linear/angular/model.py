import json
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase


class AngularF16(ModelBase):
    """High-maneuverability F-16 ✈ aircraft control object in angular coordinates.

    Action space:
        stab_act: elevator [deg]
        ail_act: ailerons [deg]
        dir_act: rudder [deg]

    State space:
        alpha: angle of attack [rad]
        beta: sideslip angle [rad]
        wx: roll angular velocity [rad/s]
        wy: yaw angular velocity [rad/s]
        wz: pitch angular velocity [rad/s]
        gamma: roll [rad]
        psi: yaw [rad]
        theta: pitch [rad]
        stab: elevator position [rad]
        ail: aileron position [rad]
        dir: rudder position [rad]
    """

    def __init__(
        self, x0, number_time_steps, selected_state_output=None, t0=0, dt: float = 0.01
    ):
        super().__init__(x0, selected_state_output, t0, dt)
        self.discretisation_time = dt
        self.folder = os.path.join(os.path.dirname(__file__), "../data")

        # Selected data for the system
        self.selected_states = [
            "phi",
            "theta",
            "psi",
            "alpha",
            "beta",
            "p",
            "q",
            "r",
            "ele",
            "ail",
            "rud",
        ]
        self.selected_output = [
            "phi",
            "theta",
            "psi",
            "alpha",
            "beta",
            "p",
            "q",
            "r",
            "nx",
            "ny",
            "nz",
        ]
        self.list_state = self.selected_output
        self.selected_input = ["ele", "ail", "rud"]
        self.control_list = self.selected_input

        self._initialize_selected_state_index(
            self.selected_state_output, self.list_state
        )

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele, ail, dir
        # Ограничения системы
        self.input_magnitude_limits = [25, 21.5, 30]
        self.input_rate_limits = [60, 80, 120]

        # Сохранение количества входов, состояний и выходов
        self.number_inputs = len(self.selected_input)
        self.number_outputs = len(self.selected_output)
        self.number_states = len(self.selected_states)

        # Оригинальные матрицы системы
        self.A = None
        self.B = None
        self.C = None
        self.D = None

        # Обработанные матрицы системы
        self.filt_A = None
        self.filt_B = None
        self.filt_C = None
        self.filt_D = None

        self.initialise_system(x0, number_time_steps)

    def import_linear_system(self):
        """Load stored linearized matrices from MATLAB files.

        Loads matrices A, B, C, D from .mat files in the data folder.
        """
        x = loadmat(self.folder + "/A.mat")
        self.A = x["A_lo"]

        x = loadmat(self.folder + "/B.mat")
        self.B = x["B_lo"]

        x = loadmat(self.folder + "/C.mat")
        self.C = x["C_lo"]

        x = loadmat(self.folder + "/D.mat")
        self.D = x["D_lo"]

    def simplify_system(self):
        """Simplify F-16 matrices by selecting only relevant states/inputs/outputs.

        Filtered matrices are stored as object attributes (filt_A, filt_B, filt_C, filt_D).
        """

        # Создавать словари с информацией из системы
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

        # Создайте новую систему и начальное состояние
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

    def create_dictionary(self, file_name):
        """Create dictionaries from available states, inputs, and outputs.

        Args:
            file_name (str): File name to read (e.g., "states", "input", "output").

        Returns:
            dict: Dictionary mapping state/input/output names to their row indices.
        """
        full_name = self.folder + "/keySet_" + file_name + ".txt"
        with open(full_name, "r") as f:
            keySet = json.loads(f.read())
        rows = dict(zip(keySet, range(len(keySet))))
        return rows

    def initialise_system(self, x0, number_time_steps):
        """Initialize F-16 aircraft dynamics.

        Args:
            x0: Initial states.
            number_time_steps: Number of time steps in the iteration.
        """
        # Импортировать сохраненную систему
        self.import_linear_system()

        # Упростите систему с выбранными состояниями
        self.simplify_system()

        # Сохраните количество временных шагов
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Дискретизировать систему в соответствии со временем дискретизации
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

    def run_step(self, ut_0: np.ndarray):
        """Execute one time step iteration.

        Args:
            ut_0 (np.ndarray): Control input vector.

        Returns:
            np.ndarray: State at the next time step.
        """
        if self.time_step != 0:
            ut_1 = self.store_input[:, self.time_step - 1]
        else:
            ut_1 = ut_0
        ut = [0, 0, 0]
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

    def update_system_attributes(self):
        """Update attributes that change with each time step."""
        self.xt = self.xt1
        self.time_step += 1

    def get_state(self, state_name: str, to_deg: bool = False, to_rad: bool = False):
        """Get state array history.

        Args:
            state_name (str): State name.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.

        Returns:
            np.ndarray: Array of selected state history.

        Example:
            >>> state_hist = model.get_state('alpha', to_deg=True)
        """
        if state_name == "wz":
            state_name = "q"
        if state_name == "wx":
            state_name = "p"
        if state_name == "wy":
            state_name = "r"
        if state_name not in self.selected_states:
            raise Exception(
                f"{state_name} нет в списке состояний, доступные {self.selected_states}"
            )
        index = self.selected_states.index(state_name)
        if to_deg:
            return np.rad2deg(self.store_states[index][: self.number_time_steps - 1])
        if to_rad:
            return np.deg2rad(self.store_states[index][: self.number_time_steps - 1])
        return self.store_states[index][: self.number_time_steps - 1]

    def get_control(
        self, control_name: str, to_deg: bool = False, to_rad: bool = False
    ):
        """Get control signal array history.

        Args:
            control_name (str): Control signal name.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.

        Returns:
            np.ndarray: Array of selected control signal history.

        Example:
            >>> control_hist = model.get_control('stab', to_deg=True)
        """
        if control_name in ["stab", "ele"]:
            control_name = "ele"
        if control_name in ["rud", "dir"]:
            control_name = "rud"
        if control_name not in self.selected_input or control_name not in [
            "ele",
            "ail",
            "rud",
        ]:
            raise Exception(
                f"{control_name} нет в списке сигналов управления, доступные {self.selected_input}"
            )
        index = self.selected_input.index(control_name)
        if to_deg:
            return np.rad2deg(self.store_input[index])[: self.number_time_steps - 1]
        if to_rad:
            return np.deg2rad(self.store_states[index][: self.number_time_steps - 1])
        return self.store_input[index][: self.number_time_steps - 1]
