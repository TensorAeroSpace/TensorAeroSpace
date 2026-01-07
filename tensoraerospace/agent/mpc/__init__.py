"""Model Predictive Control (MPC) agents.

This package contains MPC-related agents and neural-network dynamics models
used for control in TensorAeroSpace environments.
"""

from .base import AircraftMPC as AircraftMPC
from .dynamics import DynamicsNN as DynamicsNN
from .gradient import MPCOptimizationAgent as MPCOptimizationAgent
from .narx import NARX as NARX
from .narx import NARXDynamicsModel as NARXDynamicsModel
from .stochastic import MPCAgent as MPCAgent
from .torch_mpc import OneStepMLP as OneStepMLP
from .torch_mpc import TorchMPC as TorchMPC
from .torch_mpc import TorchMPCAgent as TorchMPCAgent
from .torch_mpc import TorchMPCConstraints as TorchMPCConstraints
from .torch_mpc import TorchMPCSolveResult as TorchMPCSolveResult
from .torch_mpc import (
    TorchMPCStepResponseExtraCostConfig as TorchMPCStepResponseExtraCostConfig,
)
from .torch_mpc import (
    TorchMPCTrackingExtraCostConfig as TorchMPCTrackingExtraCostConfig,
)
from .torch_mpc import TorchMPCWeights as TorchMPCWeights
from .transformers import TransformerDynamicsModel as TransformerDynamicsModel
