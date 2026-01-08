"""Model Predictive Control (MPC) agents.

This package contains MPC-related agents and neural-network dynamics models
used for control in TensorAeroSpace environments.
"""

from .mpc import MPC as MPC
from .mpc import MPCAgent as MPCAgent
from .mpc import MPCConstraints as MPCConstraints
from .mpc import MPCSolveResult as MPCSolveResult
from .mpc import MPCStandardScaler as MPCStandardScaler
from .mpc import MPCStepResponseExtraCostConfig as MPCStepResponseExtraCostConfig
from .mpc import MPCTrackingExtraCostConfig as MPCTrackingExtraCostConfig
from .mpc import MPCWeights as MPCWeights
from .mpc import OneStepMLP as OneStepMLP
from .narx import NARX as NARX
from .narx import NARXDynamicsModel as NARXDynamicsModel
from .transformers import TransformerDynamicsModel as TransformerDynamicsModel

# Backward compatibility aliases (deprecated)
TorchMPC = MPC
TorchMPCAgent = MPCAgent
TorchMPCConstraints = MPCConstraints
TorchMPCSolveResult = MPCSolveResult
TorchMPCStandardScaler = MPCStandardScaler
TorchMPCStepResponseExtraCostConfig = MPCStepResponseExtraCostConfig
TorchMPCTrackingExtraCostConfig = MPCTrackingExtraCostConfig
TorchMPCWeights = MPCWeights
