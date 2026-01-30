"""Canonical ADHDP (Action-Dependent Heuristic Dynamic Programming) agent.

This package provides a standalone implementation of ADHDP inspired by
Prokhorov & Wunsch, "Adaptive Critic Designs" (1997).
"""

from .model import ADHDP as ADHDP  # noqa: F401

__all__ = ["ADHDP"]
