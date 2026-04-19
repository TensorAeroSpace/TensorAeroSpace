"""Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI).

Fault-tolerant flight-control agent that combines

* an **INDI control law** (incremental nonlinear dynamic inversion with a
  reference model) for the nominal tracking loop,
* **Variable-Forgetting-Factor RLS** for online estimation of the
  control-effectiveness matrix ``G``, so the controller adapts quickly
  to actuator faults, and
* a light-weight **sensor-filter surrogate** (low-pass differentiator +
  exponential bias estimator) standing in for the OTSEKF-HOSM branch of
  the TU Delft AA-INDI publication.

Reference:
    Sun et al., *"Active Incremental Nonlinear Dynamic Inversion for
    Sensor and Actuator Fault Diagnosis and Fault-Tolerant Flight
    Control"*, TU Delft,
    https://research.tudelft.nl/en/publications/
    active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/
"""

from .model import AAINDIAgent as AAINDIAgent
from .model import AAINDIConfig as AAINDIConfig
from .sensor_filter import BiasEstimator as BiasEstimator
from .sensor_filter import LowPassDerivative as LowPassDerivative
from .vff_rls import VFFRLSEstimator as VFFRLSEstimator

__all__ = [
    "AAINDIAgent",
    "AAINDIConfig",
    "VFFRLSEstimator",
    "LowPassDerivative",
    "BiasEstimator",
]
