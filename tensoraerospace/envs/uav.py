import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import LongitudinalUAV


class LinearLongitudinalUAV(gym.Env):
    """Моделирование объекта управления LongitudinalUAV в среде моделирования OpenAI Gym для обучения агентов с искусственным интеллектом

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
    def __init__(self, initial_state: any,
                 reference_signal,
                 number_time_steps,
                 tracking_states=['theta', 'q'],
                 state_space=['theta', 'q'],
                 control_space=['ele'],
                 output_space=['theta', 'q'],
                 reward_func=None):

        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.selected_state_output = output_space
        self.tracking_states = tracking_states
        self.state_space = state_space
        self.control_space = control_space
        self.output_space = output_space
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward
            
        self.model = LongitudinalUAV(initial_state, number_time_steps=number_time_steps,
                                     selected_state_output=output_space, t0=0)
        self.indices_tracking_states = [state_space.index(tracking_states[i]) for i in range(len(tracking_states))]
        
        self.ref_signal = reference_signal
        self.model.initialise_system(x0=initial_state, number_time_steps=number_time_steps)
        self.number_time_steps = number_time_steps
        self.action_space = spaces.Box(low=-60, high=60, shape=(len(control_space),1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(state_space),1), dtype=np.float32)

        self.current_step = 0
        self.done = False

    def _get_info(self):
        return {}
    
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
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()

        return next_state.reshape([1,-1])[0], reward, self.done, False, info

    def reset(self):
        """Восстановление среды моделирования в начальные условия
        """
        self.model = None
        self.model = LongitudinalUAV(self.initial_state, number_time_steps=self.number_time_steps,
                                     selected_state_output=self.output_space, t0=0)
        self.ref_signal = self.reference_signal
        self.model.initialise_system(x0=self.initial_state, number_time_steps=self.number_time_steps)
        info = self._get_info()
        self.current_step = 0
        return np.array(self.initial_state, dtype=np.float64)[self.model.selected_state_index].reshape([1,-1])[0], info

    def render(self):
        """Визуальное отображение действий в среде. В статусе WIP
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()