import gym
import numpy as np
from gym import error, spaces
from gym.utils import seeding, EzPickle
from tensorairspace.aircraftmodel.model.f16.linear.longitudinal.model import LongitudinalF16


class LinearLongitudinalF16(gym.Env, EzPickle):
    def __init__(self, initial_state: any,
                 reference_signal,
                 number_time_steps,
                 tracking_states=['alpha', 'q'],
                 state_space=['alpha', 'q'],
                 control_space=['stab'],
                 output_space=['alpha', 'q'],
                 return_reward=False):
        """
            initial_state - начальное состояние
            reference_signal - заданный сигнал
            tracking_state - отслеживаемое состояние
            state_space - пространство состояний
            control_space - пространство управления
            output_space - пространство полного выхода (с учетом помех)
        """
        EzPickle.__init__(self)
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.selected_state_output = output_space
        self.tracking_states = tracking_states
        self.state_space = state_space
        self.control_space = control_space
        self.output_space = output_space
        self.reference_signal = reference_signal

        self.model = LongitudinalF16(initial_state, number_time_steps=number_time_steps,
                                     selected_state_output=output_space, t0=0)
        self.indices_tracking_states = [state_space.index(tracking_states[i]) for i in range(len(tracking_states))]
        self.return_reward = return_reward
        self.ref_signal = reference_signal
        self.model.initialise_system(x0=initial_state, number_time_steps=number_time_steps)
        self.number_time_steps = number_time_steps

    def step(self, action: np.ndarray):
        next_state = self.model.run_step(action)
        reward = next_state[self.indices_tracking_states][0] - self.ref_signal[:, self.model.time_step]
        if self.model.time_step == self.number_time_steps:
            return next_state, reward, True, {}
        return next_state, reward, False, {}

    def reset(self):
        self.model = None
        self.model = LongitudinalF16(self.initial_state, number_time_steps=self.number_time_steps,
                                     selected_state_output=self.output_space, t0=0)
        self.ref_signal = self.reference_signal
        self.model.initialise_system(x0=self.initial_state, number_time_steps=self.number_time_steps)

    def render(self):
        print("Not implimented")
