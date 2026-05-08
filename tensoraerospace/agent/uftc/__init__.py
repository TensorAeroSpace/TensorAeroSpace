"""Unified Fault-Tolerant Control (UFTC) — Phase 1 MVP orchestrator."""
from __future__ import annotations

from .controller import UFTCConfig, UFTCController
from .fdd.detector import FDDConfig, FDDDetector, FDDOutput
from .fdd.glr import GLRConfig, GLRDetector
from .inner import ModeSwitcher, SuperTwistingObserver, WrappedAAINDI
from .l1 import (
    ConformalMargin,
    ConformalMarginConfig,
    DeepReachConfig,
    DeepReachValueFn,
    HJReachabilityShield,
    HJShieldConfig,
)
from .l4 import (
    DSACConfig,
    DSACOuter,
    LongitudinalTrimFreeConfig,
    LongitudinalTrimFreeWrapper,
)
from .middle import IADPMiddle, RLSResetPolicy
from .monitor import (
    AlarmStateMachine,
    CertificateReport,
    CompositeLyapunovMonitor,
    MacroAction,
    MacroActionDispatcher,
    MonitorConfig,
    MonitorOutput,
    VState,
    run_certificate,
)

__all__ = [
    "AlarmStateMachine",
    "CertificateReport",
    "CompositeLyapunovMonitor",
    "ConformalMargin",
    "ConformalMarginConfig",
    "DSACConfig",
    "DSACOuter",
    "DeepReachConfig",
    "DeepReachValueFn",
    "FDDConfig",
    "FDDDetector",
    "FDDOutput",
    "GLRConfig",
    "GLRDetector",
    "HJReachabilityShield",
    "HJShieldConfig",
    "IADPMiddle",
    "LongitudinalTrimFreeConfig",
    "LongitudinalTrimFreeWrapper",
    "MacroAction",
    "MacroActionDispatcher",
    "ModeSwitcher",
    "MonitorConfig",
    "MonitorOutput",
    "RLSResetPolicy",
    "SuperTwistingObserver",
    "UFTCConfig",
    "UFTCController",
    "VState",
    "WrappedAAINDI",
    "run_certificate",
]
