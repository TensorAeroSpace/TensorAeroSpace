import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.base import ModelBase
from tensoraerospace.aerospacemodel.f16.nonlinear.utils import output2dict
from tensoraerospace.aerospacemodel.utils.constant import (
    state_to_latex_eng,
    state_to_latex_rus,
)


class GeoSat(ModelBase):
    """Normalized linear deviations from a circular orbit (Hla et al., 2012).

    Time is tau=t/sqrt(R/g); input is F2/(M*g). Legacy state names
    rho, theta, omega denote radial displacement, radial velocity and
    angular-rate deviations. In particular, theta is NOT angular position.

    Args:
        x0: Initial state of the control object.
        number_time_steps: Number of time steps.
        selected_state_output (optional): Selected states of the control object. Defaults to None.
        t0 (int, optional): Initial time. Defaults to 0.
        dt (float, optional): Step in normalized time tau. Defaults to 0.01.

    Action space:
        thrust: normalized tangential thrust F2/(M*g)

    State space:
        rho: normalized radial displacement delta(r/R)
        theta: legacy name for radial-velocity deviation d(delta_rho)/d(tau)
        omega: angular-rate deviation d(delta_theta)/d(tau)

    Output space:
        rho: normalized radial displacement delta(r/R)
        theta: legacy name for radial-velocity deviation d(delta_rho)/d(tau)
        omega: angular-rate deviation d(delta_theta)/d(tau)
    """

    def __init__(
        self,
        x0: np.ndarray | list[float],
        number_time_steps: int,
        selected_state_output: list[str] | None = None,
        t0: float = 0,
        dt: float = 0.01,
        initial_control: float = 0.0,
    ) -> None:
        """Initialize GeoSat instance.

        Args:
            x0: Initial state of the control object.
            number_time_steps: Number of time steps.
            selected_state_output: Selected states of the control object. Defaults to None.
            t0: Initial time. Defaults to 0.
            dt: Step in normalized time tau. Defaults to 0.01.
        """
        super().__init__(x0, selected_state_output, t0, dt)

        self.discretisation_time = dt
        if not np.isfinite(initial_control):
            raise ValueError("initial_control must be finite")
        self.initial_control = float(
            np.clip(initial_control, -np.deg2rad(25), np.deg2rad(25))
        )

        # Selected data for the system
        self.selected_states = ["rho", "theta", "omega"]
        self.selected_output = ["rho", "theta", "omega"]
        self.list_state = self.selected_states
        self.selected_input = [
            "ele",
        ]
        self.control_list = self.selected_input

        self._initialize_selected_state_index(
            self.selected_state_output, self.list_state
        )

        self.state_space = self.selected_states
        self.action_space = self.selected_input
        # ele
        # Legacy numerical bounds; these are not calibrated satellite thrust limits.
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
        self.output_history: dict[str, np.ndarray] = {}
        # Original matrices of the system
        self.A: np.ndarray = np.empty((0, 0))
        self.B: np.ndarray = np.empty((0, 0))
        self.C: np.ndarray = np.empty((0, 0))
        self.D: np.ndarray = np.empty((0, 0))

        self.initialise_system(x0, number_time_steps)

    def import_linear_system(self):
        """Reduced normalized model of Hla et al. (2012), equations (61), (66).

        The three-state system uses -0.01774; -0.1774 in the old MATLAB
        fixture disagreed with the reduced equations and orbital Jacobian.
        Remaining coefficients retain the publication's rounding.
        """
        self.A = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.01036, 0, 0.7753],
                [0, -0.01774, 0],
            ]
        )

        self.B = np.array(
            [
                [0.0],
                [0.0],
                [0.1512],
            ]
        )

        self.C = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        self.D = np.array(
            [
                [0],
                [0],
                [0],
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

        initial = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
        if initial.size != 3 or not np.all(np.isfinite(initial)):
            raise ValueError("x0 must contain three finite state deviations")
        if not np.isfinite(self.discretisation_time) or self.discretisation_time <= 0:
            raise ValueError("dt must be positive and finite")
        if int(number_time_steps) < 1:
            raise ValueError("number_time_steps must be >= 1")
        # Import the stored system
        self.import_linear_system()

        # Store the number of time steps
        self.number_time_steps = int(number_time_steps)
        self.time_step = 0

        # Discretise the system according to the discretisation time
        self.filt_A, self.filt_B, self.filt_C, self.filt_D, _ = cont2discrete(
            (self.A, self.B, self.C, self.D), self.discretisation_time
        )

        self.store_states = np.zeros((self.number_states, self.number_time_steps + 1))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_outputs = np.zeros((self.number_outputs, self.number_time_steps))

        self.x0 = initial.copy()
        self.xt = initial.copy()
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
        command = np.asarray(ut_0, dtype=np.float64).reshape(-1)
        if command.size != self.number_inputs or not np.all(np.isfinite(command)):
            raise ValueError("control must contain one finite value")
        if self.time_step >= self.number_time_steps:
            raise RuntimeError("Simulation horizon exhausted; reinitialize the model")
        previous = (
            self.store_input[:, self.time_step - 1]
            if self.time_step
            else np.array([self.initial_control])
        )
        delta = np.asarray(self.input_rate_limits) * self.discretisation_time
        ut = np.clip(command, previous - delta, previous + delta)
        ut = np.clip(
            ut, -np.asarray(self.input_magnitude_limits), self.input_magnitude_limits
        )
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
            state_name = "theta"
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
            return np.asarray(
                np.rad2deg(self.store_states[index][: self.number_time_steps])
            )
        if to_rad:
            return np.asarray(
                np.deg2rad(self.store_states[index][: self.number_time_steps])
            )
        return np.asarray(self.store_states[index][: self.number_time_steps])

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
            return np.asarray(np.rad2deg(self.store_input[index]))[
                : self.number_time_steps
            ]
        if to_rad:
            return np.asarray(
                np.deg2rad(self.store_input[index][: self.number_time_steps])
            )
        return np.asarray(self.store_input[index][: self.number_time_steps])

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
            return np.asarray(
                np.rad2deg(self.output_history[state_name][: self.time_step - 1])
            )
        if to_rad:
            return np.asarray(
                np.deg2rad(self.output_history[state_name][: self.time_step - 1])
            )
        return np.asarray(self.output_history[state_name][: self.time_step - 1])

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
