"""UFTC Phase 3 — L4 Distributional SAC outer-loop planner."""
from __future__ import annotations

from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .dsac import DSACConfig

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "GaussianActor",
    "QRDistCritic",
    "cvar_alpha_fn",
    "qr_huber_loss",
    "risk_gate",
    "soft_update",
]
