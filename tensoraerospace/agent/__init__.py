"""Reinforcement learning agents module for aerospace system control.

This module provides various reinforcement learning algorithms,
including A2C, A3C, DQN, IHDP, MPC, PPO, DDPS, GAIL, SAC, specially adapted
for aircraft and space system control tasks.
"""

from .a2c.model import A2C as A2C
from .a3c.model import Agent as Agent
from .a3c.model import setup_global_params as setup_global_params
from .ddpg.model import DDPG as DDPG
from .dqn.model import DQNAgent as DQNAgent
from .dqn.model import Model as Model
from .gail.model import GAIL as GAIL
from .ihdp.model import IHDPAgent as IHDPAgent
from .mpc.base import AircraftMPC as AircraftMPC
from .mpc.dynamics import DynamicsNN as DynamicsNN
from .ppo.model import PPO as PPO
from .sac.sac import SAC as SAC
