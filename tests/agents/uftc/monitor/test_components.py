"""Component extractors are NaN-guarded; collect_vstate composes them."""
from __future__ import annotations

import math

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.monitor.components import (
    _safe,
    extract_v_fdd,
    extract_v_indi,
    extract_v_iadp,
)


def test_safe_drops_nan_and_inf() -> None:
    assert _safe(0.5) == 0.5
    assert _safe(float("nan")) == 0.0
    assert _safe(float("inf")) == 0.0
    assert _safe(None) == 0.0


def test_extract_v_indi_from_omega_pair() -> None:
    omega = np.array([0.1, -0.2, 0.05])
    omega_ref = np.array([0.0, 0.0, 0.0])
    v = extract_v_indi(omega=omega, omega_ref=omega_ref)
    assert v >= 0.0
    assert abs(v - 0.5 * float(np.linalg.norm(omega - omega_ref) ** 2)) < 1e-12


def test_extract_v_iadp_from_state_error_and_pcritic() -> None:
    err = np.array([0.1, -0.05, 0.0])
    P = np.eye(3) * 2.0
    v = extract_v_iadp(state_error=err, P_critic=P)
    assert abs(v - 0.5 * float(err @ P @ err)) < 1e-12


def test_extract_v_fdd_from_severities() -> None:
    fdd = FDDOutput(False, severity=0.0, confidence=0.0,
                    innovation_norm=0.0, time_since_event=0.0,
                    fault_kind="none",
                    severity_abrupt=0.3, severity_gradual=0.4)
    v = extract_v_fdd(fdd)
    assert abs(v - (0.3 + 0.4)) < 1e-12
