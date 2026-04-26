"""Unified TensorBoard metrics for TensorAeroSpace RL agents.

See ``docs/superpowers/specs/2026-04-19-unified-tensorboard-metrics-design.md``
for the full schema.
"""

from . import schema
from .contract import MANDATORY_METRICS, MetricsContractError, check_contract
from .writer import MetricWriter, TorchSummaryWriter, create_metric_writer

__all__ = [
    "schema",
    "MetricWriter",
    "TorchSummaryWriter",
    "create_metric_writer",
    "MANDATORY_METRICS",
    "MetricsContractError",
    "check_contract",
]
