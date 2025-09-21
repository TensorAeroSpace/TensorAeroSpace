"""
Модуль агентов обучения с подкреплением для управления аэрокосмическими системами.

Этот модуль предоставляет различные алгоритмы обучения с подкреплением,
включая A2C, A3C, DQN, IHDP, MPC, PPO, DDPS, GAIL, SAC, специально адаптированные
для задач управления летательными аппаратами и космическими системами.
"""

from .a2c.model import A2C as A2C
from .a3c.model import Agent as Agent
from .a3c.model import setup_global_params as setup_global_params
from .ddpg.model import DDPG as DDPG
from .dqn.model import Model as Model
from .dqn.model import PERAgent as PERAgent
from .gail.model import GAIL as GAIL
from .ihdp.model import IHDPAgent as IHDPAgent
from .mpc.base import AircraftMPC as AircraftMPC
from .mpc.dynamics import DynamicsNN as DynamicsNN
from .ppo.model import PPO as PPO
from .sac.sac import SAC as SAC
