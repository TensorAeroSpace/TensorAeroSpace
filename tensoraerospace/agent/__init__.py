"""Reinforcement learning agents module for aerospace system control.

This module provides various reinforcement learning algorithms,
including A2C, A3C, DQN, IHDP, MPC, PPO, DDPS, GAIL, SAC, specially adapted
for aircraft and space system control tasks.
"""

from .a2c.model import A2C as A2C  # noqa: F401
from .a3c import Agent as Agent  # noqa: F401
from .a3c import setup_global_params as setup_global_params  # noqa: F401
from .ddpg.model import DDPG as DDPG  # noqa: F401
from .dqn.model import DQNAgent as DQNAgent  # noqa: F401
from .dqn.model import Model as Model  # noqa: F401
from .gail.model import GAIL as GAIL  # noqa: F401
from .ihdp.model import IHDPAgent as IHDPAgent  # noqa: F401
from .mpc.base import AircraftMPC as AircraftMPC  # noqa: F401
from .mpc.dynamics import DynamicsNN as DynamicsNN  # noqa: F401
from .ppo.model import PPO as PPO  # noqa: F401
from .sac.sac import SAC as SAC  # noqa: F401
