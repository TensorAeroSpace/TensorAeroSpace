from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def get_plane_env():

    """
    Обёртка для юнити среды

    :return: env --> wrapped unity environment
    """

    unity_env = UnityEnvironment("./PlaneEnv")
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    return env