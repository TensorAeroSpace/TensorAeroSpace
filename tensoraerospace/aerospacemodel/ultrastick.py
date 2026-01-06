import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.f16.nonlinear.utils import output2dict
from tensoraerospace.aerospacemodel.utils.constant import (
    state_to_latex_eng,
    state_to_latex_rus,
)


class Ultrastick(ModelBase):
    """UAV Ultrastick-25e in longitudinal control channel.

    Args:
        x0: Initial state of the control object.
        number_time_steps: Number of time steps.
        selected_state_output (optional): Selected states of the control object. Defaults to None.
        t0 (int, optional): Initial time. Defaults to 0.
        dt (float, optional): Discretization frequency. Defaults to 0.01.

    Action space:
        ele: Elevator [rad]
        delta_t: Dimensionless value, 0 — off, 1 — max thrust

    State space:
        u: Longitudinal aircraft velocity [m/s]
        w: Normal aircraft velocity [m/s]
        q: Pitch angular velocity [rad/s]
        theta: Pitch [rad]
        h: Altitude [m]

    Output space:
        u: Longitudinal aircraft velocity [m/s]
        w: Normal aircraft velocity [m/s]
        q: Pitch angular velocity [rad/s]
        theta: Pitch [rad]
        h: Altitude [m]
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
        self.selected_states = ["u", "w", "q", "theta", "h"]
        self.selected_output = ["u", "w", "q", "theta", "h"]
        self.list_state = self.selected_states
        self.selected_input = ["ele", "delta_t"]
        self.control_list = self.selected_input

        self._initialize_selected_state_index(self.selected_states, self.list_state)

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele
        # Limitations of the system (SI units)
        # ele (radians), delta_t (dimensionless)
        self.input_magnitude_limits = [np.deg2rad(30), 1]
        self.input_rate_limits = [np.deg2rad(300), 10000]

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
        """Load (set) stored linearized system matrices."""
        self.A = np.array(
            [
                [-0.5944, -0.8008, 9.791, -0.8747, 5.077e-5],
                [-0.744, -7.56, 0.5294, -1.572, 0.000939],
                [0, 0, 0, 1, 0],
                [1.041, -7.406, 0, 0, 0],
                [-15.81, -7.284e-3, 0.05399, -0.9985, 0],
            ]
        )

        self.B = np.array([[0.4669, 0], [2.703, 0], [0, 0], [133.7, 0], [0, 1]])

        self.C = np.array(
            [
                [0.9985, 0.05399, 0, 0, 0],  # Va
                [0.003176, 0.05874, 0, 0, 0],  # alpha
                [0, 0, 1, 0, 0],  # theta
                [0, 0, 0, 1, 0],  # pitch rate (q)
                [0, 0, 0, 0, 1],  # altitude (h)
            ]
        )

        self.D = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )

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
            np.ndarray: System output at the current step (via C/D).
        """
        # Ensure 1D float control vector
        ut_0 = np.asarray(ut_0, dtype=float).reshape(-1)
        if self.time_step != 0:
            ut_1 = np.asarray(
                self.store_input[:, self.time_step - 1], dtype=float
            ).reshape(-1)
        else:
            ut_1 = ut_0.copy()

        # Rate and magnitude limiting (scalar clipping)
        ut = ut_0.copy()
        for i in range(self.number_inputs):
            rate = float(self.input_rate_limits[i]) * float(self.discretisation_time)
            low_rate = float(ut_1[i] - rate)
            high_rate = float(ut_1[i] + rate)
            ut[i] = float(np.clip(float(ut[i]), low_rate, high_rate))
            amp = float(self.input_magnitude_limits[i])
            ut[i] = float(np.clip(float(ut[i]), -amp, amp))
        ut = np.asarray(ut, dtype=float).reshape(-1)
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
        output_flat = np.reshape(output, [output.shape[0]])
        if self.selected_state_output:
            return np.array(output_flat[self.selected_state_index])
        return output_flat

    def update_system_attributes(self):
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
            return np.deg2rad(self.store_input[index][: self.number_time_steps - 1])
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
            return np.rad2deg(self.output_history[state_name][: self.time_step - 1])
        if to_rad:
            return np.deg2rad(self.output_history[state_name][: self.time_step - 1])
        return self.output_history[state_name][: self.time_step - 1]

    def plot_output(
        self,
        output_name: str,
        time: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
    ) -> plt.Figure:
        """Plot an output signal over time.

        Args:
            output_name (str): Output name.
            time (np.ndarray): Time vector.
            lang (str): Axis label language ('rus' or 'eng').
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
        if not self.output_history:
            self.output_history = output2dict(self.store_outputs, self.selected_output)
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
