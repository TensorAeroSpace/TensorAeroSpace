import json
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import *


class LongitudinalF16():
    def __init__(self, x0, selected_state_output=None, dt: float = 0.01):
        # Массивы с историей
        self.u_history = []
        self.x_history = []

        # Параметры для модли
        self.dt = dt
        self.x0 = x0

        self.selected_state_output = selected_state_output

        # Массивы с обаботанными данными
        self.state_history = []
        self.control_history = []
        self.store_outputs = []

        # Массивы с доступными
        # Пространстом состояний и пространством управления
        self.list_state = []
        self.control_list = []

        # Selected data for the system
        self.selected_states = ["theta", "alpha", "q", "ele"]  # сотояния системы
        self.selected_output = ["theta", "alpha", "q", "nz"]  # выход системы
        self.list_state = self.selected_output  # состояния
        self.selected_input = ["ele", ]  # выходные состояния
        self.control_list = self.selected_input  # чем управляем

        if self.selected_state_output:
            self.selected_state_index = [self.list_state.index(val) for val in self.selected_state_output]

        self.state_space = self.selected_states
        self.action_space = self.selected_input

        self.input_magnitude_limits = [25, ]
        self.input_rate_limits = [60, ]

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

        self.folder = os.path.join(os.path.dirname(__file__), './data')

    def restart(self):
        self.u_history = []
        self.x_history = [self.x0]
        self.state_history = []
        self.control_history = []
        self.list_state = []
        self.control_list = []

    def import_linear_system(self):
        """
        Retrieves the stored linearised matrices obtained from Matlab
        :return:
        """
        x = loadmat(self.folder + '/A.mat')
        self.A = x['A_lo']

        x = loadmat(self.folder + '/B.mat')
        self.B = x['B_lo']

        x = loadmat(self.folder + '/C.mat')
        self.C = x['C_lo']

        x = loadmat(self.folder + '/D.mat')
        self.D = x['D_lo']

    def simplify_system(self):
        """
        Function which simplifies the F-16 matrices. The filtered matrices are stored as part of the object
        :return:
        """

        # Create dictionaries with the information from the system
        states_rows = self.create_dictionary('states')
        selected_rows_states = np.array([states_rows[state] for state in self.selected_states])
        output_rows = self.create_dictionary('output')
        selected_rows_output = np.array([output_rows[output] for output in self.selected_output])
        input_rows = self.create_dictionary('input')
        selected_rows_input = np.array([input_rows[input_var] for input_var in self.selected_input])

        # Create the new system and initial condition
        self.filt_A = self.A[selected_rows_states[:, None], selected_rows_states]
        self.filt_B = self.A[selected_rows_states[:, None], 12 + selected_rows_input] + \
                      self.B[selected_rows_states[:, None], selected_rows_input]
        self.filt_C = self.C[selected_rows_output[:, None], selected_rows_states]
        self.filt_D = self.C[selected_rows_output[:, None], 12 + selected_rows_input] + \
                      self.D[selected_rows_output[:, None], selected_rows_input]

    def create_dictionary(self, file_name):
        """
        Creates dictionaries from the available states, inputs and outputs
        :param file_name: name of the file to be read
        :return: rows --> dictionary with the rows used of the input/state/output vectors
        """
        full_name = self.folder + '/keySet_' + file_name + '.txt'
        with open(full_name, 'r') as f:
            keySet = json.loads(f.read())
        rows = dict(zip(keySet, range(len(keySet))))
        return rows

    def initialise_system(self, x0):
        """
        Initialises the F-16 aircraft dynamics
        :param x0: the initial states
        :param number_time_steps: the number of time steps within an iteration
        :return:
        """
        # Import the stored system
        self.import_linear_system()

        # Simplify the system with the chosen states
        self.simplify_system()

        # Store the number of time steps

        # Discretise the system according to the discretisation time
        (self.filt_A, self.filt_B, self.filt_C, self.filt_D, _) = cont2discrete((self.filt_A, self.filt_B, self.filt_C,
                                                                                 self.filt_D),
                                                                                self.dt)

        self.store_states = [x0]
        self.store_input = []
        self.store_outputs = []

        self.x0 = x0
        self.xt = x0

    def run_step(self, ut_0: np.ndarray):
        """
        Runs one time step of the iteration.
        :param ut: input to the system
        :return: xt1 --> the next time step state
        """
        if len(self.store_input) != 0:
            print("self.store_input", self.store_input)
            print("self.store_input[-1]", self.store_input[-1])

            ut_1 = self.store_input[-1]
        else:
            ut_1 = ut_0
        print("ut_1", ut_1)
        ut = [0, ]
        for i in range(self.number_inputs):
            ut[i] = max(min(max(min(ut_0[i],
                                    np.reshape(
                                        np.array([ut_1[i] + self.input_rate_limits[i] * self.dt]),
                                        [-1, 1])),
                                np.reshape(np.array([ut_1[i] - self.input_rate_limits[i] * self.dt]),
                                           [-1, 1])),
                            np.array([[self.input_magnitude_limits[i]]])),
                        - np.array([[self.input_magnitude_limits[i]]]))
        ut = np.array(ut)
        self.xt1 = np.matmul(self.filt_A, np.reshape(self.xt, [-1, 1])) + np.matmul(self.filt_B,
                                                                                    np.reshape(ut, [-1, 1]))
        output = np.matmul(self.filt_C, np.reshape(self.xt, [-1, 1]))

        self.store_input.append([np.reshape(ut, [ut.shape[0]])])
        self.store_outputs.append(np.reshape(output, [output.shape[0]]))
        self.store_states.append(np.reshape(self.xt1, [self.xt1.shape[0]]))
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
