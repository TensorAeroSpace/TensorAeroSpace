"""Incremental Approximate Dynamic Programming (iADP).

Online adaptive flight-control agent combining

* **Incremental plant identification** — a fixed-forgetting RLS tracks
  ``Θ̃ = [F̃; G̃]^T`` of the locally linearised model
  ``ΔX_{t+1} ≈ F̃ ΔX_t + G̃ Δδ_t``.
* **Approximate Dynamic Programming** — a quadratic cost-to-go
  ``V_π(X_t) = X_t^T P̃ X_t`` is fit by batch least-squares to the
  Bellman residuals on a sliding window of recent transitions.
* **Closed-form policy improvement** — the optimal incremental control
  minimising the cost-to-go is derived analytically (paper eq. (11)).

Reference:
    Konatala, Milz, Weiser, Looye, van Kampen,
    *"Flight Testing Reinforcement Learning based Online Adaptive
    Flight Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024,
    DOI: 10.2514/6.2024-2402. Flight-tested on the TU Delft /
    NLR Cessna Citation II (PH-LAB).
"""

from .model import IADPAgent as IADPAgent
from .model import IADPConfig as IADPConfig
from .rls import IncrementalRLS as IncrementalRLS

__all__ = [
    "IADPAgent",
    "IADPConfig",
    "IncrementalRLS",
]
