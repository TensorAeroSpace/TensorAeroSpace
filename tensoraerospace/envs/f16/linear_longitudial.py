"""
Модуль для моделирования линейной продольной динамики самолета F-16.

Этот модуль содержит реализацию среды Gymnasium для обучения агентов управления
продольным движением самолета F-16 Fighting Falcon. Среда использует линеаризованную
модель динамики для управления углом атаки и угловой скоростью тангажа через
руль высоты.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel.f16.linear.longitudinal.model import LongitudinalF16

# Порядок состояний в модели LongitudinalF16
MODEL_STATE_ORDER = ["theta", "alpha", "q", "ele"]


class LinearLongitudinalF16(gym.Env):
    """Моделирование объекта управления LongitudinalF16 в среде моделирования OpenAI Gym для обучения агентов с исскуственным интелектом

    Args:
        initial_state (any): Начальное состояние
        reference_signal (any): Заданный сигнал
        number_time_steps (any): Количество шагов моделирования
        tracking_states (any): Отслеживаемые состояния
        state_space (any): Пространства состояний
        control_space (any): Пространство управления
        output_space (any): Пространство полного выхода (с учетом помех)
        reward_func (any): Функция вознаграждения (статус WIP)
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        tracking_states: list = ["alpha", "q"],
        state_space: list = ["alpha", "q"],
        control_space: list = ["ele"],
        output_space: list = ["alpha", "q"],
        reward_func: callable = None,
        use_reward=True,
    ):
        super(LinearLongitudinalF16, self).__init__()

        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.reference_signal = reference_signal
        self.number_time_steps = number_time_steps
        self.tracking_states = tracking_states
        self.state_space = state_space
        self.control_space = control_space
        self.output_space = output_space
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
            state_space.index(tracking_states[i]) for i in range(len(tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(len(control_space), 1), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(state_space), 1), dtype=np.float32
        )

        self.current_step = 0
        self.done = False

    def _build_model_initial_state(
        self, init_state: np.ndarray, state_names: list
    ) -> np.ndarray:
        """Собирает начальный вектор состояния под порядок состояний модели MODEL_STATE_ORDER.

        Если init_state имеет меньшую размерность (например, только для выбранных state_names),
        недостающие компоненты заполняются нулями.
        """
        # Плоский массив значений начального состояния
        vals = np.array(init_state, dtype=float).reshape(-1)
        x0 = np.zeros(len(MODEL_STATE_ORDER), dtype=float)
        # Сопоставляем значения из state_names в соответствующие позиции модели
        for i, name in enumerate(state_names):
            if name in MODEL_STATE_ORDER:
                x0[MODEL_STATE_ORDER.index(name)] = vals[i] if i < len(vals) else 0.0
        return x0

    def _get_info(self):
        return {}

    def get_init_args(self):
        """Получаем аргументы инициализации в виде словаря."""
        init_args = self.init_args.copy()
        init_args.pop(
            "self", None
        )  # Удаление ссылки на текущий объект из словаря аргументов
        init_args.pop(
            "__class__", None
        )  # Удаление ссылки на класс из словаря аргументов
        init_args.pop("model_x0", None)  # Удаление внутренней переменной model_x0
        return init_args

    def step(self, action: np.ndarray):
        """Выполнения шага моделирования

        Args:
            action (np.ndarray): Массив управляющего сигнала по выбранным органам

        Returns:
            next_state (np.ndarray): Следующие состояние объекта управления
            reward (np.ndarray): Оценка действий алгоритма управления
            done (bool): Статус моделирования, завершено или нет
            logging (any): Дополнительная информацию (не используется)
        """
        if action[0] > self.max_action_value:
            action[0] = self.max_action_value
        if action[0] < self.max_action_value * -1:
            action[0] = self.max_action_value * -1
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = 1
        if self.use_reward:
            reward = self.reward_func(
                next_state, self.reference_signal, self.current_step
            )
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()

        return next_state.reshape([-1, 1]), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        """Восстановление среды моделирования в начальные условия"""
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False

        # Пересобираем начальное состояние под модель
        model_x0 = self._build_model_initial_state(self.initial_state, self.state_space)

        self.model = LongitudinalF16(
            model_x0,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.state_space,
        )
        self.model.initialise_system(
            x0=model_x0, number_time_steps=self.number_time_steps
        )
        info = self._get_info()

        return (
            np.array(model_x0, dtype=np.float32)[
                self.model.selected_state_index
            ].reshape([-1, 1]),
            info,
        )

    def close(self):
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
    def default_reward(state, ref_signal, ts):
        """
        Функция вознаграждения для RL среды в продольном управлении летательного аппарата.

        Аргументы:
            state (float): Текущий угол атаки летательного аппарата.
            ref_signal (float): Целевой угол атаки, за которым необходимо следить.
            ts (float): Временной шаг между итерациями обновления состояния.

        Возвращает:
            float: Величина вознаграждения для данного шага.
        """

        # Параметры для настройки функции вознаграждения

        theta, omega_z = state
        theta_ref = ref_signal[:, ts]

        # Расчет ошибки угла атаки
        angle_error = abs(theta - theta_ref)

        # Наказание за высокую угловую скорость
        omega_penalty = abs(omega_z)

        # Вознаграждение как функция ошибки угла и наказания за скорость
        # Можно настроить веса для этих компонентов в зависимости от предпочтений в управлении
        reward = -angle_error - 0.1 * omega_penalty

        # Возвращаем как np.ndarray для совместимости с тестами
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
