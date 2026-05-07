"""Reinforcement learning agents module for aerospace system control.

This module provides various reinforcement learning algorithms,
including A2C, A3C, DQN, IHDP, MPC, PPO, DDPS, GAIL, SAC, specially adapted
for aircraft and space system control tasks.
"""

from .a2c.model import A2C as A2C  # noqa: F401
from .a3c import Agent as Agent  # noqa: F401
from .a3c import setup_global_params as setup_global_params  # noqa: F401

# Canonical ADHDP (standalone)
from .adhdp.model import ADHDP as ADHDP  # noqa: F401

# Adaptive Critic / ADP (Prokhorov & Wunsch 1995 inspired)
from .adp.adp import ADP as ADP  # noqa: F401

# AIDI — Adaptive Incremental Dynamic Inversion (Ul Haq et al. 2026)
from .aidi.model import AIDIAgent as AIDIAgent  # noqa: F401
from .aidi.model import AIDIConfig as AIDIConfig  # noqa: F401
from .ddpg.model import DDPG as DDPG  # noqa: F401
from .dqn.model import DQNAgent as DQNAgent  # noqa: F401
from .dqn.model import Model as Model  # noqa: F401
from .dsac.dsac import DSAC as DSAC  # noqa: F401

# Event-triggered DHP (Sun et al., CEAS EuroGNC 2022)
from .et_dhp.model import ETDHPAgent as ETDHPAgent  # noqa: F401
from .et_dhp.model import ETDHPConfig as ETDHPConfig  # noqa: F401
from .gail.model import GAIL as GAIL  # noqa: F401

# Model-based HDP (standalone wrapper)
from .hdp.model import HDP as HDP  # noqa: F401
from .ihdp.model import IHDPAgent as IHDPAgent  # noqa: F401

# Incremental Model-based GDHP (partial observability)
from .im_gdhp.model import IMGDHPAgent as IMGDHPAgent  # noqa: F401
from .im_gdhp.model import IMGDHPConfig as IMGDHPConfig  # noqa: F401

# Backward compatibility alias
from .mpc.mpc import MPC as MPC  # noqa: F401
from .mpc.mpc import MPCAgent as MPCAgent  # noqa: F401
from .ppo.model import PPO as PPO  # noqa: F401
from .sac.sac import SAC as SAC  # noqa: F401

TorchMPCAgent = MPCAgent

# Unified Fault-Tolerant Control (UFTC) — Phase 1 MVP
from .uftc import UFTCConfig as UFTCConfig  # noqa: F401
from .uftc import UFTCController as UFTCController  # noqa: F401
