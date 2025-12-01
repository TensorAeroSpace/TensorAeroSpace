"""
Модуль для моделирования самолета наблюдения
LAPAN Surveillance Aircraft (LSU)-05 NG.

Этот модуль содержит реализацию среды Gymnasium для обучения
агентов управления продольным движением самолета LAPAN LSU-05 NG.
Среда предоставляет интерфейс для взаимодействия с моделью самолета,
включая управление углом тангажа и угловой скоростью тангажа
через стабилизаторы.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import LAPAN


class LinearLongitudinalLAPAN(gym.Env):
    """Моделирование объекта управления LAPAN Surveillance Aircraft (LSU)-05 NG
    в среде моделирования OpenAI Gym для обучения агентов с искусственным
    интеллектом

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
        initial_state: Any,
        reference_signal: Any,
        number_time_steps: Any,
        tracking_states=None,
        state_space=None,
        control_space=None,
        output_space=None,
        reward_func: Any = None,
    ):
        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.tracking_states = tracking_states or ["theta", "q"]
        self.state_space = state_space or ["theta", "q"]
        self.control_space = control_space or ["stab"]
        self.output_space = output_space or ["theta", "q"]
        self.selected_state_output = self.output_space
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = LAPAN(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
        )
        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-25,
            high=25,
            shape=(len(self.control_space), 1),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_space), 1),
            dtype=np.float32,
        )

        self.ref_signal = reference_signal
        self.model.initialise_system(
            x0=initial_state, number_time_steps=number_time_steps
        )
        self.number_time_steps = number_time_steps
        self.current_step = 0
        self.done = False

    @staticmethod
    def reward(state, ref_signal, ts):
        """Оценка управления

        Args:
            state (_type_): Текущее состояния
            ref_signal (_type_): Заданное состояние
            ts (_type_): Временное шаг

        Returns:
            reward (float): Оценка управления
        """
        return np.abs(state[0] - ref_signal[:, ts])

    def _get_info(self):
        return {}

    def step(self, action: np.ndarray):
        """Выполнения шага моделирования

        Args:
            action (np.ndarray): Массив управляющего сигнала по
                выбранным органам

        Returns:
            next_state (np.ndarray): Следующие состояние объекта управления
            reward (np.ndarray): Оценка действий алгоритма управления
            done (bool): Статус моделирования, завершено или нет
            logging (any): Дополнительная информацию
                (не используется)
        """
        if action[0] > self.max_action_value:
            action[0] = self.max_action_value
        if action[0] < self.max_action_value * -1:
            action[0] = self.max_action_value * -1
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = self.reward_func(
            next_state[self.indices_tracking_states],
            self.ref_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()
        return next_state.reshape([-1, 1]), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        """Восстановление среды моделирования в начальные условия

        Args:
            seed (int, optional): Seed для генератора случайных чисел
            options (dict, optional): Дополнительные опции для инициализации
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.model = LAPAN(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
        )
        self.ref_signal = self.reference_signal
        self.model.initialise_system(
            x0=self.initial_state,
            number_time_steps=self.number_time_steps,
        )
        info = self._get_info()
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape([-1, 1])
        return observation, info

    def render(self):
        """Визуальное отображение действий в среде. В статусе WIP
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()


class ImprovedLAPANEnv(gym.Env):
    """LAPAN env with normalized spaces; internal units are radians."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_elevator_deg: float = 0.0,
        use_initial_action_on_first_step: bool = True,
    ) -> None:
        super().__init__()

        # Physical/normalization limits
        self.max_pitch_rad = np.deg2rad(20.0)
        self.max_pitch_rate_rad_s = np.deg2rad(5.0)
        self.max_elevator_angle_rad = np.deg2rad(25.0)
        self.max_elevator_angle_deg = 25.0

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Simulation parameters
        self.dt = float(dt)
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        self.reference_signal = np.array(reference_signal, dtype=float)
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0
        # LAPAN state order: [u, w, q, theta]
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

        # Initial elevator and action history (normalized)
        self.initial_elevator_deg = float(initial_elevator_deg)
        self.initial_action_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_elevator_angle_deg,
                -1.0,
                1.0,
            )
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0

        # Reward shaping
        self.reward_scale = 0.1
        self.w_pitch = 5.0
        self.w_q = 0.2
        self.w_action = 0.003
        self.w_smooth = 0.01
        self.w_jerk = 0.001

        # Store init args for helpers
        self.init_args = locals()

        # Underlying LAPAN model (keep full state output order)
        self.model = LAPAN(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
        )
        self.model.initialise_system(
            x0=self.initial_state,
            number_time_steps=self.number_time_steps,
        )

    # Helper indices based on LAPAN state order [u, w, q, theta]
    @property
    def _idx_q(self) -> int:
        return 2

    @property
    def _idx_theta(self) -> int:
        return 3

    def _get_obs(self) -> np.ndarray:
        # Model provides theta, q in radians now
        theta_rad = float(self.state[self._idx_theta])
        q_rad_s = float(self.state[self._idx_q])
        idx = int(np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1))
        target_theta = float(self.reference_signal[0, idx])  # radians

        pitch_error = target_theta - theta_rad
        norm_pitch_error = float(np.clip(pitch_error / self.max_pitch_rad, -1.0, 1.0))
        norm_q = float(np.clip(q_rad_s / self.max_pitch_rate_rad_s, -1.0, 1.0))
        norm_theta = float(np.clip(theta_rad / self.max_pitch_rad, -1.0, 1.0))
        norm_prev_action = float(self.previous_action)

        return np.array(
            [norm_pitch_error, norm_q, norm_theta, norm_prev_action],
            dtype=np.float32,
        )

    def get_init_args(self):
        init_args = self.init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        return init_args

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model.initialise_system(
            self.initial_state,
            self.number_time_steps,
        )
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # action in [-1, 1]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Scale to radians (LAPAN input now expects radians), optionally use
        # initial elevator at first step (converted to rad)
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_action_rad = np.array(
                [np.deg2rad(self.initial_elevator_deg)],
                dtype=np.float32,
            )
        else:
            scaled_action_rad = action * self.max_elevator_angle_rad

        # LAPAN model expects radians for elevator input
        self.state = self.model.run_step(scaled_action_rad).reshape(-1)
        self.current_step += 1

        # Reward (model outputs are radians)
        theta_rad = float(self.state[self._idx_theta])
        q_rad_s = float(self.state[self._idx_q])
        idx_safe = int(
            np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta = float(self.reference_signal[0, idx_safe])  # radians

        # Reference derivative for relative q penalty
        if self.current_step > 0:
            idx_prev = int(
                np.clip(
                    self.current_step - 1,
                    0,
                    self.reference_signal.shape[1] - 1,
                )
            )
            ref_theta_prev = float(self.reference_signal[0, idx_prev])
        else:
            ref_theta_prev = target_theta
        ref_theta_dot = float((target_theta - ref_theta_prev) / self.dt)

        e_theta = float((theta_rad - target_theta) / self.max_pitch_rad)
        e_q_rel = float((q_rad_s - ref_theta_dot) / self.max_pitch_rate_rad_s)

        u_applied_norm = float(
            np.asarray(scaled_action_rad).reshape(-1)[0] / self.max_elevator_angle_rad
        )
        u = u_applied_norm
        du = u_applied_norm - float(self.previous_action)
        ddu = (
            u_applied_norm
            - 2.0 * float(self.previous_action)
            + float(self.pre_previous_action)
        )

        cost = (
            self.w_pitch * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action * (u**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )
        reward = float(-cost) * float(self.reward_scale)

        self.pre_previous_action = float(self.previous_action)
        self.previous_action = float(u_applied_norm)
        self._last_reward = float(reward)

        terminated = False
        if abs(theta_rad) > self.max_pitch_rad:
            reward = -100.0
            terminated = True

        truncated = self.current_step >= self.number_time_steps - 2

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    def render(self, mode: str = "human"):
        # Visualization not implemented for LAPAN
        return

    def close(self):
        return
