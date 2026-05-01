"""Base module for aerospace models.

This module contains the base class ModelBase, which serves as the foundation for all
aerospace models in the TensorAeroSpace library. The class provides common
functionality for modeling aircraft dynamics, including state management,
simulation history, visualization and data analysis.

Main capabilities:
    - State and control signal management
    - Simulation history tracking
    - Results visualization
    - Simulation data analysis
    - Support for various data formats
"""

import warnings

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from .f16.nonlinear.utils import control2dict, state2dict
from .utils.constant import (
    control_to_latex_eng,
    control_to_latex_rus,
    ref_state_to_latex_eng,
    ref_state_to_latex_rus,
    state_to_latex_eng,
    state_to_latex_rus,
)


class ModelBase:
    """Base class for models.

    Args:
        dt: Discretization step.
        selected_state_output: Selected states for working with the system.
        t0: Initial time.
        x0: Initial state.

    Internal variables:
        time_step: Simulation step.
        u_history: All control signals during simulation.
        x_history: All states during simulation.
        state_history: All states during simulation in dict format (for convenient work with plots).
        control_history: All control signals during simulation in dict format (for convenient work with plots).
        list_state: List of all control object states.
        control_list: List of all control object control signals.
        dt: Discretization step.
    """

    def __init__(self, x0, selected_state_output=None, t0=0, dt: float = 0.01):
        """Initialize ModelBase instance.

        Args:
            x0: Initial state.
            selected_state_output: Selected states for working with the system.
            t0: Initial time. Defaults to 0.
            dt: Discretization step. Defaults to 0.01.
        """
        # Массивы с историей
        self.u_history: list[Any] = []
        self.x_history: list[Any] = []

        # Параметры для модели
        self.dt = dt
        self.time_step = 1  # 1 - потому что матлаб
        self.t0 = t0
        self.x0 = x0
        self.selected_state_output = selected_state_output
        self.number_time_steps = 0
        # Текущие состояния, управляющий сигнал и выход системы
        self.xt: Any = None
        self.xt1: Any = None
        self.store_states: np.ndarray = np.empty((0, 0))
        self.store_input: np.ndarray = np.empty((0, 0))
        self.store_outputs: np.ndarray = np.empty((0, 0))
        self.output_history: dict[str, np.ndarray] = {}

    def _initialize_selected_state_index(self, selected_state_output, list_state):
        """Initialize selected_state_index based on selected_state_output.

        Args:
            selected_state_output: List of selected output states.
            list_state: Complete list of model states.
        """
        if selected_state_output:
            self.selected_state_index = [
                list_state.index(val) for val in selected_state_output
            ]
        else:
            # Если selected_state_output не задан, используем все состояния
            self.selected_state_index = list(range(len(list_state)))
        self.yt = None
        self.ut = None

        # Массивы с обработанными данными
        self.state_history: dict[str, np.ndarray] = {}
        self.control_history: dict[str, np.ndarray] = {}
        self.store_outputs = np.empty((0, 0))

        # Массивы с доступными
        # Пространством состояний и пространством управления
        self.list_state: list[str] = []
        self.control_list: list[str] = []

    def run_step(self, u):
        """Calculate control object state.

        Args:
            u: Control signal for current step.
        """
        pass

    def restart(self):
        """Restart the entire control object.

        Resets all internal variables and state history
        to initial values.
        """
        self.time_step = 1
        self.u_history = []
        self.x_history = [self.x0]
        # Historical public state: callers and tests expect an empty list after
        # restart. get_state/get_control lazily rebuild the dict representation.
        self.state_history = cast(Any, [])
        self.control_history = cast(Any, [])
        self.list_state = []
        self.control_list = []

    def get_state(self, state_name: str, to_deg: bool = False, to_rad: bool = False):
        """Get state array.

        Args:
            state_name: State name.
            to_deg: Convert to degrees.
            to_rad: Convert to radians.

        Returns:
            Selected state history array.

        Example:
            >>> state_hist = model.get_state('alpha', to_deg=True)
        """
        if to_rad and to_deg:
            raise Exception(
                "Invalid formatting specified, choose one type: to_rad or to_deg."
            )
        if state_name not in self.list_state:
            raise Exception(f"{state_name} is not in the states list")
        if not self.state_history:
            self.state_history = state2dict(self.x_history, self.list_state)
        if to_deg:
            return np.rad2deg(self.state_history[state_name][: self.time_step - 1])
        if to_rad:
            return np.deg2rad(self.state_history[state_name][: self.time_step - 1])
        return self.state_history[state_name][: self.time_step - 1]

    def get_control(
        self, control_name: str, to_deg: bool = False, to_rad: bool = False
    ):
        """Get control signal array.

        Args:
            control_name: Control signal name.
            to_deg: Convert to degrees.
            to_rad: Convert to radians.

        Returns:
            Selected control signal history array.

        Example:
            >>> state_hist = model.get_control('stab', to_deg=True)
        """
        if to_rad and to_deg:
            raise Exception(
                "Invalid formatting specified, choose one type: to_rad or to_deg."
            )
        if control_name not in self.control_list:
            raise Exception(f"{control_name} is not in the control signals list")
        if not self.control_history:
            self.control_history = control2dict(self.u_history, self.control_list)
        if to_deg:
            return np.rad2deg(self.control_history[control_name][: self.time_step - 1])
        if to_rad:
            return np.deg2rad(self.control_history[control_name][: self.time_step - 1])
        return self.control_history[control_name][: self.time_step - 1]

    def plot_state(
        self,
        state_name: str,
        time: np.ndarray,
        lang: Any = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
        close: bool = False,
    ):
        """Plot control object states.

        Args:
            state_name (str): State name.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.
            time (np.ndarray): Time array for plotting.
            lang (str): Label language. Defaults to "rus".
            figsize (tuple): Figure size. Defaults to (10, 10).
            close (bool): If True, close the figure with ``plt.close(fig)``
                before returning (useful for batch processing to avoid
                memory accumulation). Defaults to False.

        Returns:
            plt.Figure: Plot of selected state.

        Example:
            >>> plot = model.plot_by_state('alpha', time, to_deg=True, figsize=(5,4))
        """
        # Backwards-compatible guard:
        # Some older examples passed a reference signal as the 3rd positional argument
        # (e.g. plot_state('wz', tps, reference_signals[0], ...)). The 3rd argument is
        # actually `lang` (a string), so we ignore non-string values and keep defaults.
        if not isinstance(lang, str):
            warnings.warn(
                "`plot_state` expects `lang` (str) as the 3rd argument. "
                "It looks like a reference signal was passed; this value will be ignored. "
                "Use `plot_transient_process(state_name, time, ref_signal, ...)` to plot a reference.",
                UserWarning,
            )
            lang = "rus"

        state_hist = self.get_state(state_name, to_deg, to_rad)
        if lang == "rus":
            label = state_to_latex_rus[state_name]
            label_time = "t, c"
        else:
            label = state_to_latex_eng[state_name]
            label_time = "t, sec."
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            time[: self.time_step - 1], state_hist[: self.time_step - 1], label=label
        )
        ax.legend()
        ax.set_xlabel(label_time)
        ax.set_ylabel(label)
        ax.grid(True)
        if close:
            plt.close(fig)
        return fig

    def plot_error(
        self,
        state_name: str,
        time: np.ndarray,
        ref_signal: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
        xlim: tuple[float, float] = (13, 20),
        ylim: tuple[float, float] = (-3, 3),
        close: bool = False,
    ):
        """Plot control error.

        .. math:: \\epsilon = ref - state

        Args:
            state_name (str): State name.
            time (np.ndarray): Time array for plotting.
            ref_signal (np.ndarray): Reference signal.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.
            lang (str): Label language. Defaults to "rus".
            figsize (tuple): Figure size. Defaults to (10, 10).
            xlim (list): X-axis limits. Defaults to [13, 20].
            ylim (list): Y-axis limits. Defaults to [-3, 3].
            close (bool): If True, close the figure with ``plt.close(fig)``
                before returning (useful for batch processing to avoid
                memory accumulation). Defaults to False.

        Returns:
            plt.Figure: Plot of transient process.

        Example:
            >>> plot = model.plot_error('alpha', time, ref_signal, to_deg=True, figsize=(5,4))
        """
        state_hist = self.get_state(state_name, to_deg, to_rad)
        error = ref_signal[: self.time_step - 1] - state_hist[: self.time_step - 1]
        if lang == "rus":
            label = r"$\varepsilon$, град."
            label_time = "t, c"
        else:
            label = r"$\varepsilon$, deg"
            label_time = "t, sec."
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot(
            time[: self.time_step - 1],
            error[: self.time_step - 1],
            label=label,
            color="red",
        )
        ax.legend()
        ax.set_xlabel(label_time)
        ax.set_ylabel(label)
        ax.grid(True)
        if close:
            plt.close(fig)
        return fig

    def plot_transient_process(
        self,
        state_name: str,
        time: np.ndarray,
        ref_signal: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
        close: bool = False,
    ):
        """Plot transient process.

        Args:
            state_name (str): State name.
            time (np.ndarray): Time array for plotting.
            ref_signal (np.ndarray): Reference signal.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.
            lang (str): Label language. Defaults to "rus".
            figsize (tuple): Figure size. Defaults to (10, 10).
            close (bool): If True, close the figure with ``plt.close(fig)``
                before returning (useful for batch processing to avoid
                memory accumulation). Defaults to False.

        Returns:
            plt.Figure: Plot of transient process.

        Example:
            >>> plot = model.plot_transient_process('alpha', time, ref_signal, to_deg=True, figsize=(5,4))
        """
        state_hist = self.get_state(state_name, to_deg, to_rad)
        if lang == "rus":
            label = state_to_latex_rus[state_name]
            label_ref = ref_state_to_latex_rus[state_name]
            label_time = "t, c"
        else:
            label = state_to_latex_eng[state_name]
            label_ref = ref_state_to_latex_eng[state_name]
            label_time = "t, sec."
        fig, ax = plt.subplots(figsize=figsize)
        if to_deg:
            ax.plot(
                time[: self.time_step - 1],
                np.rad2deg(ref_signal[: self.time_step - 1]),
                label=label_ref,
                color="red",
            )
        else:
            ax.plot(
                time[: self.time_step - 1],
                ref_signal[: self.time_step - 1],
                label=label_ref,
                color="red",
            )
        ax.plot(
            time[: self.time_step - 1], state_hist[: self.time_step - 1], label=label
        )
        ax.legend()
        ax.set_xlabel(label_time)
        ax.set_ylabel(label)
        ax.grid(True)
        if close:
            plt.close(fig)
        return fig

    def plot_control(
        self,
        control_name: str,
        time: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
        close: bool = False,
    ):
        """Plot control signals.

        Args:
            control_name (str): Control signal name.
            to_deg (bool): Convert to degrees. Defaults to False.
            to_rad (bool): Convert to radians. Defaults to False.
            time (np.ndarray): Time array for plotting.
            lang (str): Label language. Defaults to "rus".
            figsize (tuple): Figure size. Defaults to (10, 10).
            close (bool): If True, close the figure with ``plt.close(fig)``
                before returning (useful for batch processing to avoid
                memory accumulation). Defaults to False.

        Returns:
            plt.Figure: Plot of selected control signal.

        Example:
            >>> plot = model.plot_by_control('stab', time, to_deg=True, figsize=(15,4))
        """
        state_hist = self.get_control(control_name, to_deg, to_rad)
        if lang == "rus":
            label = control_to_latex_rus[control_name]
            label_time = "t, c"
        else:
            label = control_to_latex_eng[control_name]
            label_time = "t, sec."
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            time[: self.time_step - 1],
            state_hist[: self.time_step - 1],
            label=label,
            color="green",
        )
        ax.legend()
        ax.set_xlabel(label_time)
        ax.set_ylabel(label)
        ax.grid(True)
        if close:
            plt.close(fig)
        return fig
