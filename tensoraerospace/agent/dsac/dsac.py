"""DSAC compatibility wrapper.

TensorAeroSpace exposes DSAC as `tensoraerospace.agent.DSAC`.

The implementation is ported from the `dsac-flight` project and lives in
`tensoraerospace/agent/dsac/dsac_flight.py`.
"""

from __future__ import annotations

from .dsac_flight import DSAC

__all__ = ["DSAC"]
