import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tensoraerospace.aerospacemodel.f16.linear.longitudinal.model import LongitudinalF16

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
    def __init__(self, initial_state: np.ndarray,
                 reference_signal: np.ndarray,
                 number_time_steps: int,
                 tracking_states: list = ['alpha', 'q'],
                 state_space: list = ['alpha', 'q'],
                 control_space: list = ['stab'],
                 output_space: list = ['alpha', 'q'],
                 reward_func: callable = None):
        super(LinearLongitudinalF16, self).__init__()
        
        self.initial_state = initial_state
        self.reference_signal = reference_signal
        self.number_time_steps = number_time_steps
        self.tracking_states = tracking_states
        self.state_space = state_space
        self.control_space = control_space
        self.output_space = output_space
        self.reward_func = reward_func if reward_func is not None else self.default_reward

        self.model = LongitudinalF16(initial_state, number_time_steps=number_time_steps,
                                     selected_state_output=output_space)
        self.indices_tracking_states = [state_space.index(tracking_states[i]) for i in range(len(tracking_states))]

        self.action_space = spaces.Discrete(1,1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(state_space),1), dtype=np.float64)

        self.current_step = 0
        self.done = False
        
    def _get_info(self):
        return {}
    
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
        self.current_step += 1
        next_state = self.model.run_step(action)
        reward = self.reward_func(next_state[self.indices_tracking_states], self.reference_signal, self.current_step)
        self.done = self.current_step >= self.number_time_steps
        info = self._get_info()

        return next_state, reward, self.done, False, info

    def reset(self, seed=None, options=None):
        """Восстановление среды моделирования в начальные условия
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.model = LongitudinalF16(self.initial_state, number_time_steps=self.number_time_steps,
                                     selected_state_output=self.output_space)
        self.model.initialise_system(x0=self.initial_state, number_time_steps=self.number_time_steps)
        info = self._get_info()
        
        return np.array(self.initial_state, dtype=np.float64)[self.model.selected_state_index], info


    def close(self):
        # Implement cleanup logic here
        pass

    @staticmethod
    def default_reward(state, ref_signal, ts):
        """Оценка упавления

        Args:
            state (_type_): Текущее состояния
            ref_signal (_type_): Заданное состояние
            ts (_type_): Временное шаг

        Returns:
            reward (float): Оценка упавления
        """
        return np.abs(state[0] - ref_signal[:, ts])
