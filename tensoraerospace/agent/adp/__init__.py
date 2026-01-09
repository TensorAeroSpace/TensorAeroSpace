"""Adaptive Critic Design / Approximate Dynamic Programming (ADP) agents.

This subpackage provides a lightweight Adaptive Critic Design (ACD) style
actor-critic agent inspired by:

Prokhorov D.V., Wunsch D.C. “Adaptive critic designs: A case study for
neurocontrol.” Neural Networks, 8(9), 1995, pp. 1367–1372.
"""

from .adp import ADP as ADP  # noqa: F401

__all__ = ["ADP"]
