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
        self.u_history = []
        self.x_history = []

        # Параметры для модели
        self.dt = dt
        self.time_step = 1  # 1 - потому что матлаб
        self.t0 = t0
        self.x0 = x0
        self.selected_state_output = selected_state_output
        self.number_time_steps = 0
        # Текущие состояния, управляющий сигнал и выход системы
        self.xt = None
        self.xt1 = None

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
        self.state_history = []
        self.control_history = []
        self.store_outputs = []

        # Массивы с доступными
        # Пространством состояний и пространством управления
        self.list_state = []
        self.control_list = []

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
        self.state_history = []
        self.control_history = []
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
        if control_name not in self.list_state:
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
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
    ):
        """
        Графики состояний ОУ

        Args:
            state_name: Название состояния
            to_deg: Конвертировать в градусы
            to_rad: Конвертировать в радианы
            time: Время на графике
            lang: Язык обозначений на осях
            figsize: Размер графика

        Returns:
            График выбранного состояния

        Пример:

        >>> plot = model.plot_by_state('alpha', time, to_deg=True, figsize=(5,4))

        """
        state_hist = self.get_state(state_name, to_deg, to_rad)
        if lang == "rus":
            label = state_to_latex_rus[state_name]
            label_time = "t, c"
        else:
            label = state_to_latex_eng[state_name]
            label_time = "t, sec."
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.plot(
            time[: self.time_step - 1], state_hist[: self.time_step - 1], label=label
        )
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)

    def plot_error(
        self,
        state_name: str,
        time: np.ndarray,
        ref_signal: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
        xlim: list = [13, 20],
        ylim: list = [-3, 3],
    ):
        """
        График ошибки регулирования

        .. math:: \\epsilon = ref - state

        Args:
            state_name: Название состояния
            time: Время на графике
            ref_signal: Заданный сигнал
            to_deg: Конвертировать в градусы
            to_rad: Конвертировать в радианы
            lang: Язык обозначений на осях
            figsize: Размер графика

        Returns:
            График переходного процесса


        Пример:

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
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot(
            time[: self.time_step - 1],
            error[: self.time_step - 1],
            label=label,
            color="red",
        )
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)

    def plot_transient_process(
        self,
        state_name: str,
        time: np.ndarray,
        ref_signal: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
    ):
        """
        Графики переходного процесса

        Args:
            state_name: Название состояния
            time: Время на графике
            ref_signal: Заданный сигнал
            to_deg: Конвертировать в градусы
            to_rad: Конвертировать в радианы
            lang: Язык обозначений на осях
            figsize: Размер графика

        Returns:
            График переходного процесса

        Пример:

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
        fig = plt.figure(figsize=figsize)
        plt.clf()
        if to_deg:
            plt.plot(
                time[: self.time_step - 1],
                np.rad2deg(ref_signal[: self.time_step - 1]),
                label=label_ref,
                color="red",
            )
        else:
            plt.plot(
                time[: self.time_step - 1],
                ref_signal[: self.time_step - 1],
                label=label_ref,
                color="red",
            )
        plt.plot(
            time[: self.time_step - 1], state_hist[: self.time_step - 1], label=label
        )
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)

    def plot_control(
        self,
        control_name: str,
        time: np.ndarray,
        lang: str = "rus",
        to_deg: bool = False,
        to_rad: bool = False,
        figsize: tuple = (10, 10),
    ):
        """
        Графики управляющих сигналов

        Args:
            control_name: Название состояния
            to_deg: Конвертировать в градусы
            to_rad: Конвертировать в радианы
            time: Время на графике
            lang: Язык обозначений на осях
            figsize: Размер графика

        Returns:
            График выбранного состояния

        Пример:

        >>> plot = model.plot_by_control('stab', time, to_deg=True, figsize=(15,4))
        """
        state_hist = self.get_control(control_name, to_deg, to_rad)
        if lang == "rus":
            label = control_to_latex_rus[control_name]
            label_time = "t, c"
        else:
            label = control_to_latex_eng[control_name]
            label_time = "t, sec."
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.legend()
        plt.xlabel(label_time)
        plt.ylabel(label)
        plt.grid(True)
        plt.plot(
            time[: self.time_step - 1],
            state_hist[: self.time_step - 1],
            label=label,
            color="green",
        )
