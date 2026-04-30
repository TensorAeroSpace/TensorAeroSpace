"""Adaptive Incremental Dynamic Inversion agent (AIDI).

Reference:
    Ul Haq, Atmaca, van Kampen, "Adaptive Incremental Dynamic Inversion for
    Fault-tolerant Flight Control of a Flying Wing", AIAA SciTech 2026,
    AIAA 2026-1744 — https://doi.org/10.2514/6.2026-1744
"""

from .allocator import MoorePenroseAllocator as MoorePenroseAllocator
from .model import AIDIAgent as AIDIAgent
from .model import AIDIConfig as AIDIConfig
from .onboard_ce import F16NonlinearOnboardCE as F16NonlinearOnboardCE
from .onboard_ce import LinearOnboardCE as LinearOnboardCE
from .onboard_ce import OnboardCEModel as OnboardCEModel
from .pch import PseudoControlHedge as PseudoControlHedge
from .ref_models import CStarController as CStarController
from .ref_models import LinearController as LinearController
from .ref_models import RollReferenceModel as RollReferenceModel
from .ref_models import SideslipCompensator as SideslipCompensator
from .ref_models import SpeedController as SpeedController
from .scaling_rls import ScalingRLS as ScalingRLS
from .utils import reconstruct_n_z as reconstruct_n_z

__all__ = [
    "AIDIAgent",
    "AIDIConfig",
    "MoorePenroseAllocator",
    "OnboardCEModel",
    "LinearOnboardCE",
    "F16NonlinearOnboardCE",
    "PseudoControlHedge",
    "ScalingRLS",
    "CStarController",
    "RollReferenceModel",
    "SideslipCompensator",
    "SpeedController",
    "LinearController",
    "reconstruct_n_z",
]
