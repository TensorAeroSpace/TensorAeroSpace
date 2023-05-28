from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import gym
from gym.spaces.discrete import Discrete
import numpy as np


def get_plane_env(env_path="", server=False, worker=0):
    """Функция которая возвращает gym обёртку для нашей юнити среды

    Returns:
         env --> wrapped unity environment
    """

    unity_env = UnityEnvironment(env_path, worker_id=worker, no_graphics=server)
    
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    return env


class unity_discrete_env(gym.Wrapper):
    """Дискретная обёртка для нашей юнити среды
    """

    def __init__(self, env_path="", server=False):
        super().__init__(gym.Wrapper)
        self.action_space = Discrete(3 ** 7)
        self.env = get_plane_env(env_path=env_path, server=server)

    def reset(self):
        """Функция которая перезагружает unity среду и возвращает первое наблюдение после перезагрузки

        Returns:
            obs (_type_): первое наблюдение среды
        """
        return self.env.reset()

    def step(self, action):
        """Функция которая переводит дискретное действие алгоритма dqn в непрерывное действие unity среды

        Args:
            action (int): дискретное действие алгоритма dqn
        Returns:
            transition (_type_): переход, который возвращает unity среда
        """
        actions = np.array([0.0] * 7, dtype=np.float32)
        for i in range(7):
            actions[i] = (action // 3 ** i) % 3 - 1.0
        return self.env.step(actions)

    def close(self):
        """Функция которая закрывает unity среду
        """
        self.env.close()