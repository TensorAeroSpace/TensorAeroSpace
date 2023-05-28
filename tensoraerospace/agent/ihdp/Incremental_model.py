import numpy as np


class IncrementalModel:
    """Предоставляет класс IncrementalModel для идентификации системы

    IncrementalModel вычисляет матрицы A и x, необходимые для идентификации системы,
    вычисляет матрицы F и G, необходимые для инкрементной модели, и оценивает идентифицированной модели, чтобы обеспечить оценки состояний на следующем временном шаге.
    
    Args:
        selected_states (_type_): Выбранные состояния
        selected_input (_type_): Выбранные управляющие сигналы
        number_time_steps (_type_): Количество временных шагов
        discretisation_time (float, optional): Время дискретизации. Defaults to 0.5.
        input_magnitude_limits (int, optional): Пределы входных управляющих сигнгалов. Defaults to 25.
        input_rate_limits (int, optional): Ограничения скорости управляющих сигнгалов. Defaults to 60.
    """

    def __init__(self, selected_states, selected_input, number_time_steps, discretisation_time=0.5,
                 input_magnitude_limits=25, input_rate_limits=60):
        # Define the inputs to the incremental model
        self.xt_1 = None
        self.xt = None
        self.ut_1 = None
        self.ut = None
        self.delta_xt = None
        self.delta_ut = None
        self.xt1_est = None

        # Define the data window size
        self.number_time_steps = number_time_steps
        self.number_states = len(selected_states)
        self.number_inputs = len(selected_input)
        self.L = 2 * (self.number_inputs + self.number_states)
        self.store_delta_xt = np.zeros((self.number_states, self.number_time_steps))
        self.store_delta_xt_0 = np.random.rand(self.number_states, self.L)
        self.store_delta_ut = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_delta_ut_0 = np.random.rand(self.number_inputs, self.L)
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))

        # Define the system identification matrices
        self.F = np.zeros((self.number_states, self.number_states))
        self.G = np.zeros((self.number_states, self.number_inputs))

        # Define the time variable
        self.time_step = 0
        self.discretisation_time = discretisation_time

        # Limitations of the system
        self.input_magnitude_limits = input_magnitude_limits
        self.input_rate_limits = input_rate_limits

    def save_matrix(self):
        """Сохранить матрицы
        """
        np.save('./incremental_model/g', self.G)
        np.save('./incremental_model/f', self.F)
        np.save('./incremental_model/delta_ut', self.delta_ut)
        np.save('./incremental_model/delta_xt', self.delta_xt)

    def load_matrix(self):
        """Загрузить матрицы
        """
        self.G = np.load('./incremental_model/g.npy', )
        self.F = np.load('./incremental_model/f.npy', )
        self.delta_ut = np.load('./incremental_model/delta_ut.npy', )
        self.delta_xt = np.load('./incremental_model/delta_xt.npy', )

    def build_A_LS_matrix(self):
        """Строит матрицу А, необходимую для онлайн-метода идентификации методом наименьших квадратов.

        Returns:
            A_LS_matrix: Матрица наименьших квадратов
        """
        
        if self.time_step >= self.L:
            x_component = np.flip(self.store_delta_xt[:, self.time_step - self.L:self.time_step], 1).T
            u_component = np.flip(self.store_delta_ut[:, self.time_step - self.L:self.time_step], 1).T
        else:
            x_component_1 = np.flip(self.store_delta_xt[:, :self.time_step], 1).T
            x_component_2 = self.store_delta_xt_0[:, :self.L - self.time_step].T
            x_component = np.vstack((x_component_1, x_component_2))

            u_component_1 = np.flip(self.store_delta_ut[:, :self.time_step], 1).T
            u_component_2 = self.store_delta_ut_0[:, :self.L - self.time_step].T
            u_component = np.vstack((u_component_1, u_component_2))
        A_LS_matrix = np.hstack((x_component, u_component))
        return A_LS_matrix

    def build_x_LS_vector(self):
        """Строит вектор x, требуемый в методе наименьших квадратов онлайн-идентификации системы.

        Returns:
            x_LS_vector: x вектор который в требуется в наименьших квадратах
        """
        if self.time_step == 0:
            self.xt_1 = self.xt

        # Computation and storage of the gradients
        self.delta_xt = self.xt - self.xt_1
        self.delta_ut = self.ut - self.ut_1
        self.store_delta_xt[:, self.time_step] = np.reshape(self.delta_xt, [self.delta_xt.shape[0]])
        self.store_delta_ut[:, self.time_step] = np.reshape(self.delta_ut, [self.delta_ut.shape[0]])
        if self.time_step >= self.L:
            x_LS_vector = np.flip(self.store_delta_xt[:, self.time_step - self.L + 1:self.time_step + 1], 1).T
        else:
            x_component_1 = np.flip(self.store_delta_xt[:, :self.time_step + 1], 1).T
            x_component_2 = self.store_delta_xt_0[:, :self.L - self.time_step - 1].T
            x_LS_vector = np.vstack((x_component_1, x_component_2))

        return x_LS_vector

    def identify_incremental_model_LS(self, xt, ut_0):
        """Вычисляет матрицы F и G идентификации системы

        Args:
            xt (_type_): текущее состояние на временном шаге t
            ut_0 (_type_): входной сигнал на текущем временном шаге

        Returns:
            G (_type_): матрица распределения входных данных
        """        
        # Verifying that the inputs meets the platforms constraints
        if self.time_step == 0:
            self.ut_1 = ut_0
        ut = max(min(max(min(ut_0,
                             np.reshape(np.array([self.ut_1 + self.input_rate_limits * self.discretisation_time]),
                                        [-1, 1])),
                         np.reshape(np.array([self.ut_1 - self.input_rate_limits * self.discretisation_time]),
                                    [-1, 1])),
                     np.array([[self.input_magnitude_limits]])),
                 - np.array([[self.input_magnitude_limits]]))

        # Store the input variables
        self.xt = xt
        self.ut = ut
        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])

        # Obtain the A matrix and the x vector
        A_LS_matrix = self.build_A_LS_matrix()
        x_LS_vector = self.build_x_LS_vector()
        identified_matrices = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A_LS_matrix.T, A_LS_matrix)), A_LS_matrix.T),
                                        x_LS_vector).T
        self.F = identified_matrices[:, :self.number_states]
        self.G = identified_matrices[:, self.number_states:]

        return self.G

    def evaluate_incremental_model(self, *args):
        """Оценивает состояния следующего временного шага

        Returns:
            xt1_est (_type_): оценка состояния следующего временного шага
        """
        
        if len(args) == 0:
            # Estimate the next time step states
            self.xt1_est = self.xt + np.matmul(self.F, self.delta_xt) + np.matmul(self.G, self.delta_ut)
            return self.xt1_est
        elif len(args) == 1:
            # Estimate the next time step states
            ut_0 = args[0]
            ut = max(min(max(min(ut_0,
                                 np.reshape(np.array([self.ut_1 + self.input_rate_limits * self.discretisation_time]),
                                            [-1, 1])),
                             np.reshape(np.array([self.ut_1 - self.input_rate_limits * self.discretisation_time]),
                                        [-1, 1])),
                         np.array([[self.input_magnitude_limits]])),
                     - np.array([[self.input_magnitude_limits]]))

            delta_ut = ut - self.ut_1
            xt1_est = self.xt + np.matmul(self.F, self.delta_xt) + np.matmul(self.G, delta_ut)
            return xt1_est
        elif len(args) == 2:
            self.xt = args[0]
            ut_0 = args[1]
            if self.time_step == 0:
                self.ut_1 = ut_0
                self.xt_1 = self.xt
            # Estimate the next time step states

            ut = max(min(max(min(ut_0,
                                 np.reshape(np.array([self.ut_1 + self.input_rate_limits * self.discretisation_time]),
                                            [-1, 1])),
                             np.reshape(np.array([self.ut_1 - self.input_rate_limits * self.discretisation_time]),
                                        [-1, 1])),
                         np.array([[self.input_magnitude_limits]])),
                     - np.array([[self.input_magnitude_limits]]))

            self.delta_ut = ut - self.ut_1
            self.delta_xt = self.xt - self.xt_1
            xt1_est = self.xt + np.matmul(self.F, self.delta_xt) + np.matmul(self.G, self.delta_ut)
            return xt1_est

    def update_incremental_model_attributes(self):
        """Атрибуты, которые меняются с каждым временным шагом, обновляются
        """

        # Update the object state and input variables
        self.xt_1 = self.xt
        self.ut_1 = self.ut
        self.time_step += 1

    def restart_time_step(self):
        """Обнуление временного шага
        """
        self.time_step = 0

    def restart_incremental_model(self):
        """Перезапускает инкрементную модель.
        """
        self.time_step = 0
        self.store_delta_xt = np.zeros((self.number_states, self.number_time_steps))
        self.store_delta_ut = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))
