"""Event-triggered Dual Heuristic Programming (ET-DHP) agent.

Reference:
    Bo Sun, Cheng Liu, Killian Dally, Erik-Jan van Kampen.
    "Intelligent Aircraft Stabilization Control with Event-Triggered Scheme".
    CEAS EuroGNC 2022.
    https://sunbo.me/publication/cf_2022_caes_eurognc/

The controller couples a neural plant model, a bounded-action actor
``u = u_b · tanh(D(x))`` and a costate critic ``λ(x) = ∂J/∂x``. An
event-triggered supervisor defers actor/critic re-training (and even
control recomputation) until the deviation of the measured state from
the last triggered state exceeds a Lipschitz-style threshold, which in
turn caps the sampling rate without hurting tracking quality.
"""

from .event_trigger import EventTrigger
from .model import ETDHPAgent, ETDHPConfig
from .networks import ETDHPActor, ETDHPCritic, PlantModelNN

__all__ = [
    "ETDHPAgent",
    "ETDHPConfig",
    "ETDHPActor",
    "ETDHPCritic",
    "PlantModelNN",
    "EventTrigger",
]
