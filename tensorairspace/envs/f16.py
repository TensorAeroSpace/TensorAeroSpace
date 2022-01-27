import gym
from gym import error, spaces
from gym.utils import seeding, EzPickle


class LongitudalSuperSonic(gym.Env, EzPickle):
    def __init__(self, continuous: bool = False):
        pass