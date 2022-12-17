import gym
import numpy as np
from gym import error, spaces
from gym.utils import seeding, EzPickle
from tensorairspace.aerospacemodel import ELVRocket



class LinearLongitudinalELVRocket(gym.Env, EzPickle):
    """Моделирование объекта управления LongitudinalB747 в среде моделирования OpenAI Gym для обучения агентов с исскуственным интелектом

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
                 control_space=['stab'],
                 output_space=['theta', 'q'],
                 reward_func=None):

        EzPickle.__init__(self)
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
            
        self.model = ELVRocket(initial_state, number_time_steps=number_time_steps,
                                     selected_state_output=output_space, t0=0)
        self.indices_tracking_states = [state_space.index(tracking_states[i]) for i in range(len(tracking_states))]
        
        self.ref_signal = reference_signal
        self.model.initialise_system(x0=initial_state, number_time_steps=number_time_steps)
        self.number_time_steps = number_time_steps
    
    @staticmethod
    def reward(state, ref_signal, ts):
        """Оценка упавления

        Args:
            state (_type_): Текущее состояния
            ref_signal (_type_): Заданное состояние
            ts (_type_): Временное шаг

        Returns:
            reward (float): Оценка упавления
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
        next_state = self.model.run_step(action)
        reward = self.reward_func(next_state[self.indices_tracking_states], self.ref_signal, self.model.time_step)
        if self.model.time_step == self.number_time_steps:
            return next_state, reward, True, {}
        return next_state, reward, False, {}

    def reset(self):
        """Восстановление среды моделирования в начальные условия
        """
        self.model = None
        self.model = LongitudinalB747(self.initial_state, number_time_steps=self.number_time_steps,
                                     selected_state_output=self.output_space, t0=0)
        self.ref_signal = self.reference_signal
        self.model.initialise_system(x0=self.initial_state, number_time_steps=self.number_time_steps)

    def render(self):
        """Визуальное отображение действий в среде. В статусе WIP
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()