import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.signal import cont2discrete

from .base import ModelBase
from .f16.nonlinear.utils import output2dict
from .utils.constant import state_to_latex_eng, state_to_latex_rus


class ELVRocket(ModelBase):
    """ELV rocket in longitudinal control channel.

    Args:
        x0: Initial state of the control object.
        number_time_steps: Number of time steps.
        selected_state_output (optional): Selected states of the control
            object. Defaults to None.
        t0 (int, optional): Initial time. Defaults to 0.
        dt (float, optional): Discretization frequency. Defaults to 0.01.

    Action space:
        ele: elevator [rad]

    State space (order):
        w: Longitudinal aircraft velocity [m/s]
        q: Pitch angular velocity [rad/s]
        theta: Pitch [rad]


    Output space (order):
        w: Longitudinal aircraft velocity [m/s]
        q: Pitch angular velocity [rad/s]
        theta: Pitch [rad]
    """

    def __init__(
        self,
        x0: np.ndarray | list[float],
        number_time_steps: int,
        selected_state_output: list[int] | None = None,
        t0: float = 0,
        dt: float = 0.01,
    ) -> None:
        super().__init__(x0, selected_state_output, t0, dt)

        self.discretisation_time = dt

        # Selected data for the system
        self.selected_states = ["w", "q", "theta"]
        self.selected_output = ["w", "q", "theta"]
        self.list_state = self.selected_states
        self.selected_input = [
            "ele",
        ]
        self.control_list = self.selected_input

        self._initialize_selected_state_index(self.selected_states, self.list_state)

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele (radians)
        # Limitations of the system
        self.input_magnitude_limits = [
            float(np.deg2rad(25.0)),
        ]
        self.input_rate_limits = [
            float(np.deg2rad(60.0)),
        ]

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

        self.initialise_system(x0, number_time_steps)

    def import_linear_system(self) -> None:
        """Load (set) stored linearized system matrices.

        The original matrices are defined for the legacy state order
        ``[alpha, q, theta]`` and are converted to the current order
        ``[w, q, theta]`` using a permutation matrix.
        """
        # Old-order matrices: x_old = [alpha, q, theta]
        A_old = np.array(
            [
                [-100.858, 1, -0.1256],
                [14.7805, 0, 0.01958],
                [0.0, 1.0, 0],
            ]
        )

        B_old = np.array(
            [
                [20.42],
                [3.4558],
                [0],
            ]
        )

        # Permutation from new -> old basis
        # new = [w, q, theta] corresponds to old = [alpha, q, theta]
        # so indices: old[0]=alpha->new[0]=w, old[1]=q->new[1]=q,
        # old[2]=theta->new[2]=theta. P[i,j] = 1 if old[i] = new[j]
        P = np.array(
            [
                [1, 0, 0],  # alpha_old = w_new
                [0, 1, 0],  # q_old = q_new
                [0, 0, 1],  # theta_old = theta_new
            ]
        )

        # Transform to new basis: A_new = P^T A_old P, B_new = P^T B_old
        self.A = P.T @ A_old @ P
        self.B = P.T @ B_old

        # Identity output in new basis
        self.C = np.eye(3)
        self.D = np.zeros((3, 1))

    def initialise_system(
        self, x0: np.ndarray | list[float], number_time_steps: int
    ) -> None:
        """Initialize the system and allocate history buffers.

        Args:
            x0: Initial state.
            number_time_steps: Number of simulation steps.
        """

        # Import the stored system
        self.import_linear_system()

        # Store the number of time steps
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Discretise the system according to the discretisation time
        (self.filt_A, self.filt_B, self.filt_C, self.filt_D, _) = cont2discrete(
            (self.A, self.B, self.C, self.D), self.discretisation_time
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
        """Run one discrete-time simulation step.

        Args:
            ut_0 (np.ndarray): Control vector.

        Returns:
            np.ndarray: Next state at time t+1.
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
        output = np.matmul(self.filt_C, np.reshape(self.xt, [-1, 1])) + np.matmul(
            self.filt_D, np.reshape(ut, [-1, 1])
        )

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
        """Update time-dependent attributes after each simulation step."""
        self.xt = self.xt1
        self.time_step += 1

    def get_state(
        self, state_name: str, to_deg: bool = False, to_rad: bool = False
    ) -> np.ndarray:
        """Return the time history of a state.

        Args:
            state_name: State name.
            to_deg: Convert radians to degrees.
            to_rad: Convert degrees to radians.

        Returns:
            np.ndarray: State history array.
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
    ) -> np.ndarray:
        """Return the time history of a control input.

        Args:
            control_name: Control signal name.
            to_deg: Convert radians to degrees.
            to_rad: Convert degrees to radians.

        Returns:
            np.ndarray: Control history array.
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

    def get_output(
        self, state_name: str, to_deg: bool = False, to_rad: bool = False
    ) -> np.ndarray:
        """Return the time history of an output signal.

        Args:
            state_name (str): Output name.
            to_deg (bool): Convert radians to degrees.
            to_rad (bool): Convert degrees to radians.

        Returns:
            np.ndarray: Output history array.
        """
        self.output_history = output2dict(self.store_outputs, self.selected_output)
        if to_deg:
            return np.rad2deg(self.state_history[state_name][: self.time_step - 1])
        if to_rad:
            return np.deg2rad(self.state_history[state_name][: self.time_step - 1])
        return self.output_history[state_name][: self.time_step - 1]

    def plot_output(
        self,
        output_name: str,
        time: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
    ) -> Figure:
        """Plot an output signal over time.

        Args:
            output_name (str): Output name.
            time (np.ndarray): Time vector.
            lang (str): Axis label language ('rus' or 'eng'). Defaults to 'rus'.
            to_deg (bool): Convert radians to degrees.
            to_rad (bool): Convert degrees to radians.
            figsize (tuple): Figure size.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        if to_rad and to_deg:
            raise Exception(
                "Неверно указано форматирование, укажите один. to_rad или to_deg."
            )
        if output_name not in self.list_state:
            raise Exception(f"{output_name} нет в списке сигналов управления")
        if not self.control_history:
            self.control_history = output2dict(self.store_outputs, self.selected_output)
        state_hist = self.get_output(output_name, to_deg, to_rad)
        if output_name == "u":
            state_hist *= 1.94384
        if lang == "rus":
            label = state_to_latex_rus[output_name]
            label_time = "t, c"
        else:
            label = state_to_latex_eng[output_name]
            label_time = "t, sec."
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.plot(time[: self.time_step - 1], state_hist, label=label)
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)
        return fig
