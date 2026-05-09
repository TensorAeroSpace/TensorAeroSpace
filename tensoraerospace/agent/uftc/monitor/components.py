"""NaN-guarded extractors of the five composite-Lyapunov components.

Each extractor returns 0.0 on missing/NaN/inf input rather than
crashing the controller. ``collect_vstate(controller)`` is a one-shot
composer used by ``UFTCController.learn()``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .composite import VState

if TYPE_CHECKING:  # pragma: no cover
    from tensoraerospace.agent.uftc.controller import UFTCController


def _safe(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def extract_v_hj(*, value_fn_value: float | None, conformal_eps: float | None) -> float:
    v = _safe(value_fn_value)
    eps = _safe(conformal_eps)
    return max(0.0, eps - v)


def extract_v_indi(*, omega: np.ndarray | None, omega_ref: np.ndarray | None) -> float:
    if omega is None or omega_ref is None:
        return 0.0
    err = np.asarray(omega, dtype=np.float64) - np.asarray(omega_ref, dtype=np.float64)
    return float(0.5 * np.dot(err, err))


def extract_v_iadp(
    *, state_error: np.ndarray | None, P_critic: np.ndarray | None
) -> float:
    if state_error is None or P_critic is None:
        return 0.0
    e = np.asarray(state_error, dtype=np.float64)
    P = np.asarray(P_critic, dtype=np.float64)
    # Allow augmented (2n x 2n) P_critic by slicing to match state_error dim.
    n = e.shape[0]
    if P.ndim == 2 and P.shape[0] >= n and P.shape[1] >= n:
        P = P[:n, :n]
    else:
        return 0.0
    return float(0.5 * (e @ (P @ e)))


def extract_v_dsac(*, z_quantiles: np.ndarray | None, var_target: float = 0.5) -> float:
    if z_quantiles is None:
        return 0.0
    var = float(np.asarray(z_quantiles, dtype=np.float64).var())
    return max(0.0, var - var_target)


def extract_v_fdd(fdd: FDDOutput | None) -> float:
    if fdd is None:
        return 0.0
    return _safe(getattr(fdd, "severity_abrupt", 0.0)) + _safe(
        getattr(fdd, "severity_gradual", 0.0)
    )


def collect_vstate(controller: "UFTCController") -> VState:
    """Centralised V-state collector. Layers expose their own
    ``last_*`` properties; missing properties degrade to 0.0."""
    cfg = controller.cfg
    fdd = getattr(controller, "_last_fdd", None)

    v_hj = (
        extract_v_hj(
            value_fn_value=getattr(controller.l1, "_last_v_x", None),
            conformal_eps=getattr(controller.l1, "_last_eps", None),
        )
        if getattr(cfg, "enable_l1_shield", False)
        and getattr(controller, "l1", None) is not None
        else 0.0
    )
    v_indi = extract_v_indi(
        omega=getattr(controller.inner, "_last_omega_meas", None),
        omega_ref=getattr(controller.inner, "_last_omega_ref", None),
    )
    v_iadp = extract_v_iadp(
        state_error=getattr(controller.middle, "_last_state_error", None),
        P_critic=(
            getattr(controller.middle.base, "P_critic", None)
            if hasattr(controller.middle, "base")
            else None
        ),
    )
    v_dsac = (
        extract_v_dsac(z_quantiles=getattr(controller.l4, "_last_z", None))
        if getattr(cfg, "enable_l4_outer", False)
        and getattr(controller, "l4", None) is not None
        else 0.0
    )
    v_fdd = extract_v_fdd(fdd)
    return VState(
        V_hj=v_hj,
        V_indi=v_indi,
        V_iadp=v_iadp,
        V_dsac=v_dsac,
        V_fdd=v_fdd,
        timestamp=float(getattr(controller, "_step", 0)) * float(cfg.dt),
    )
