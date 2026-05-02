"""Module for modeling linear longitudinal dynamics of F-16 aircraft.

This module contains a Gymnasium environment implementation for training agents
to control longitudinal motion of F-16 Fighting Falcon aircraft. The environment uses
a linearized dynamics model to control angle of attack and pitch angular velocity
through elevator control.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.linear.longitudinal.model import LongitudinalF16

# Порядок состояний в модели LongitudinalF16
MODEL_STATE_ORDER = ["theta", "alpha", "q", "ele"]


class LinearLongitudinalF16(gym.Env):
    """Simulation of LongitudinalF16 control object in OpenAI Gym environment for training AI agents.

    Args:
        initial_state: Initial state.
        reference_signal: Reference signal.
        number_time_steps: Number of simulation steps.
        tracking_states: Tracked states.
        state_space: State space.
        control_space: Control space.
        output_space: Full output space (including noise).
        reward_func: Reward function (WIP status).
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: (
            Callable[[np.ndarray, np.ndarray, int], np.ndarray | float] | None
        ) = None,
        use_reward: bool = True,
    ) -> None:
        """Initialize LinearLongitudinalF16 environment.

        Args:
            initial_state (np.ndarray): Initial state.
            reference_signal (np.ndarray): Reference signal.
            number_time_steps (int): Number of simulation steps.
            tracking_states (list): Tracked states. Defaults to ["alpha", "q"].
            state_space (list): State space. Defaults to ["alpha", "q"].
            control_space (list): Control space. Defaults to ["ele"].
            output_space (list): Full output space. Defaults to ["alpha", "q"].
            reward_func (callable): Reward function. Defaults to None.
            use_reward: Whether to use reward. Defaults to True.
        """
        super(LinearLongitudinalF16, self).__init__()

        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.reference_signal = reference_signal
        self.number_time_steps = number_time_steps
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["alpha", "q"]
        )
        self.state_space = state_space if state_space is not None else ["alpha", "q"]
        self.control_space = control_space if control_space is not None else ["ele"]
        self.output_space = output_space if output_space is not None else ["alpha", "q"]
        self.use_reward = use_reward
        self.reward_func = (
            reward_func if reward_func is not None else self.default_reward
        )

        # Построим начальный вектор состояния модели согласно порядку состояний модели
        model_x0 = self._build_model_initial_state(self.initial_state, self.state_space)

        self.init_args = locals()
        self.model = LongitudinalF16(
            model_x0,
            number_time_steps=number_time_steps,
            selected_state_output=self.state_space,
        )

        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-self.max_action_value,
            high=self.max_action_value,
            shape=(len(self.control_space),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_space),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.done = False

    def _build_model_initial_state(
        self, init_state: np.ndarray, state_names: list
    ) -> np.ndarray:
        """Build initial state vector according to MODEL_STATE_ORDER.

        If init_state has smaller dimension (e.g., only for selected state_names),
        missing components are filled with zeros.

        Args:
            init_state (np.ndarray): Initial state values.
            state_names (list): List of state names.

        Returns:
            np.ndarray: Initial state vector matching MODEL_STATE_ORDER.
        """
        # Плоский массив значений начального состояния
        vals = np.array(init_state, dtype=float).reshape(-1)
        x0 = np.zeros(len(MODEL_STATE_ORDER), dtype=float)
        # Сопоставляем значения из state_names в соответствующие позиции модели
        for i, name in enumerate(state_names):
            if name in MODEL_STATE_ORDER:
                x0[MODEL_STATE_ORDER.index(name)] = vals[i] if i < len(vals) else 0.0
        return x0

    def _get_info(self) -> dict[str, float]:
        """Return auxiliary info for Gym API (currently empty)."""
        return {}

    def get_init_args(self) -> dict[str, object]:
        """Get initialization arguments as a dictionary.

        Returns:
            dict: Dictionary of initialization arguments.
        """
        init_args = self.init_args.copy()
        init_args.pop(
            "self", None
        )  # Удаление ссылки на текущий объект из словаря аргументов
        init_args.pop(
            "__class__", None
        )  # Удаление ссылки на класс из словаря аргументов
        init_args.pop("model_x0", None)  # Удаление внутренней переменной model_x0
        return init_args

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        """Execute one simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators.

        Returns:
            tuple: Tuple containing:
                - next_state (np.ndarray): Next state of the control object.
                - reward (np.ndarray): Evaluation of control algorithm actions.
                - done (bool): Simulation status, whether completed or not.
                - truncated (bool): Whether episode was truncated.
                - info (dict): Additional information (not used).
        """
        action = np.asarray(action).reshape(-1)
        action = np.clip(action, -self.max_action_value, self.max_action_value)
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward: np.ndarray | float = 1.0
        if self.use_reward:
            reward = self.reward_func(
                next_state, self.reference_signal, self.current_step
            )
        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()

        reward_value = float(np.asarray(reward, dtype=float).squeeze())

        return (
            np.asarray(next_state).reshape(-1).astype(np.float32),
            reward_value,
            self.done,
            False,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Reset options. Defaults to None.

        Returns:
            tuple: Tuple containing:
                - observation (np.ndarray): Initial observation.
                - info (dict): Additional information.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False

        # Пересобираем начальное состояние под модель
        model_x0 = self._build_model_initial_state(self.initial_state, self.state_space)

        # Constructor already calls initialise_system internally, so no
        # explicit call is needed here.
        self.model = LongitudinalF16(
            model_x0,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.state_space,
        )
        info = self._get_info()

        return (
            np.array(model_x0, dtype=np.float32)[
                self.model.selected_state_index
            ].reshape(-1),
            info,
        )

    def close(self):
        """Release resources (no-op placeholder)."""
        # Implement cleanup logic here
        pass

    # @staticmethod
    # def default_reward(state, ref_signal, ts):
    #     """Оценка упавления

    #     Args:
    #         state (_type_): Текущее состояния
    #         ref_signal (_type_): Заданное состояние
    #         ts (_type_): Временной шаг

    #     Returns:
    #         reward (float): Оценка упавления
    #     """
    #     alpha = state[0]
    #     error = abs(alpha - ref_signal[:, ts])
    #     penalty = error**2  # Квадратичный штраф за ошибку
    #     reward = -penalty
    #     return reward

    @staticmethod
    def default_reward(
        state: np.ndarray, ref_signal: np.ndarray, ts: int
    ) -> np.ndarray:
        """Reward function for RL environment in longitudinal aircraft control.

        Supports variable-length state vectors. The last two elements of the
        flattened state are treated as ``[tracked_angle, angular_rate]`` so the
        reward works for both 2-state ``[alpha, q]`` and 3-state
        ``[theta, alpha, q]`` configurations.

        Args:
            state (np.ndarray): Current aircraft state (at least 2 elements).
            ref_signal (np.ndarray): Target angle trajectory, shape ``(1, T)``.
            ts (int): Current time step index.

        Returns:
            np.ndarray: Reward value for this step.
        """
        state_flat = np.asarray(state, dtype=float).reshape(-1)
        if state_flat.size < 2:
            raise ValueError(
                f"default_reward expects state with >=2 elements, got {state_flat.size}"
            )

        angle = float(state_flat[-2])
        angular_rate = float(state_flat[-1])
        angle_ref = float(np.asarray(ref_signal).reshape(-1)[ts])

        angle_error = abs(angle - angle_ref)
        rate_penalty = abs(angular_rate)
        reward = -angle_error - 0.1 * rate_penalty

        return np.array(reward)

    # @staticmethod
    # def default_reward(state, ref_signal, ts):
    #     """Оценка упавления

    #     Args:
    #         state (_type_): Текущее состояния
    #         ref_signal (_type_): Заданное состояние
    #         ts (_type_): Временной шаг

    #     Returns:
    #         reward (float): Оценка упавления
    #     """
    #     alpha = state[0]
    #     reward_for_perfect_alignment = 1.0
    #     penalty_for_deviation = 0.2  # Штраф за каждую единицу отклонения от целевого угла

    #     # Расчет отклонения от целевого угла атаки
    #     deviation = abs(alpha - ref_signal[:, ts])

    #     # Расчет вознаграждения с учетом отклонения
    #     reward = reward_for_perfect_alignment - (penalty_for_deviation * deviation)

    #     # Гарантия того, что вознаграждение не станет отрицательным
    #     reward = max(reward, 0)
    #     reward = np.array(reward) if not isinstance(reward, np.ndarray) else reward
    #     return reward
