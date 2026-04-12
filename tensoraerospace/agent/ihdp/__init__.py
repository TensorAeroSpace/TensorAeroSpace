"""IHDP agent components.

This package provides building blocks for an Incremental Heuristic Dynamic
Programming (IHDP) agent implementation.

Note: Requires TensorFlow. Install with ``pip install tensoraerospace[tensorflow]``.
"""

try:
    from .Actor import Actor as Actor
    from .Critic import Critic as Critic
    from .Incremental_model import IncrementalModel as IncrementalModel
    from .model import IHDPAgent as IHDPAgent
except ImportError:
    pass  # TensorFlow not installed
