"""FDDDetector composition with optional GLR; FDDOutput extension."""

from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.change_point import ChangePointDetector
from tensoraerospace.agent.uftc.fdd.detector import (
    FDDConfig,
    FDDDetector,
    FDDOutput,
)
from tensoraerospace.agent.uftc.fdd.glr import GLRConfig, GLRDetector
from tensoraerospace.agent.uftc.fdd.kalman_3step import NominalKalman


def _build_detector(*, with_glr: bool) -> FDDDetector:
    n = 2
    F = np.eye(n) * 0.0  # incremental form
    G = np.zeros((n, 1))
    Q = np.eye(n) * 1e-3
    R = np.eye(n) * 1e-2
    kalman = NominalKalman(F_nominal=F, G_nominal=G, Q=Q, R=R)
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0, cooldown_steps=200)
    glr = (
        GLRDetector(
            n_dim=n, cfg=GLRConfig(window=100, h_alarm=30.0, cooldown_steps=200)
        )
        if with_glr
        else None
    )
    return FDDDetector(n_state=n, n_control=1, kalman=kalman, cpd=cpd, glr=glr, dt=0.01)


def test_extended_fields_default_for_clean_input() -> None:
    rng = np.random.default_rng(0)
    det = _build_detector(with_glr=True)
    x = np.zeros(2)
    u = np.zeros(1)
    out = det.step(x + rng.standard_normal(2) * 0.05, u)
    assert isinstance(out, FDDOutput)
    assert out.fault_kind in ("none", "abrupt", "gradual", "compound")
    assert hasattr(out, "severity_abrupt")
    assert hasattr(out, "severity_gradual")
    assert out.severity == max(out.severity_abrupt, out.severity_gradual)


def test_phase1_compatibility_when_glr_disabled() -> None:
    det = _build_detector(with_glr=False)
    out = det.step(np.zeros(2), np.zeros(1))
    assert out.severity_gradual == 0.0
    # Phase 1 consumers reading FDDOutput.severity see CUSUM-only severity.
    assert abs(out.severity - out.severity_abrupt) < 1e-12


def test_compound_when_both_channels_alarm() -> None:
    rng = np.random.default_rng(1)
    det = _build_detector(with_glr=True)

    # Burn in clean dynamics.
    for _ in range(500):
        det.step(rng.standard_normal(2) * 0.05, np.zeros(1))

    # Inject a sustained large-mean drift to fire both CUSUM and GLR.
    seen_kinds: set[str] = set()
    for _ in range(2000):
        x = rng.standard_normal(2) * 0.05 + np.array([4.0, 0.0])
        out = det.step(x, np.zeros(1))
        seen_kinds.add(out.fault_kind)
    assert (
        "compound" in seen_kinds or "abrupt" in seen_kinds
    )  # at least abrupt; compound when GLR catches up
    assert "compound" in seen_kinds  # both must fire eventually under sustained step
