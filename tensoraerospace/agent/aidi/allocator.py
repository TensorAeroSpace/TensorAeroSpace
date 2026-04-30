"""Moore-Penrose pseudoinverse-based control allocator.

Used by the AIDI inner loop to map a virtual-control demand
``ν − ω̇_meas`` to a control increment ``Δu``. Falls back to zero on
ill-conditioning (a numerical guard during the RLS warm-up).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MoorePenroseAllocator:
    """Minimum-norm control allocation via :func:`numpy.linalg.pinv`.

    Args:
        rcond: Cut-off for small singular values, passed to
            :func:`numpy.linalg.pinv`.
        cond_threshold: When ``cond(G)`` exceeds this value, ``allocate``
            returns ``Δu = 0`` and emits a warning instead of inverting.
    """

    def __init__(self, rcond: float = 1e-8, cond_threshold: float = 1e8) -> None:
        if rcond <= 0.0 or cond_threshold <= 0.0:
            raise ValueError("rcond and cond_threshold must be positive")
        self.rcond = float(rcond)
        self.cond_threshold = float(cond_threshold)

    def allocate(
        self,
        G_eff: np.ndarray,
        nu_des: np.ndarray,
        omega_dot_meas: np.ndarray,
    ) -> np.ndarray:
        """Compute ``Δu = G⁺ · (ν_des − ω̇_meas)``.

        Args:
            G_eff: Scaled control-effectiveness matrix ``G̃``, shape
                ``(n_y, n_u)``.
            nu_des: Virtual control vector, shape ``(n_y,)``.
            omega_dot_meas: Measured angular acceleration, shape ``(n_y,)``.

        Returns:
            Control increment ``Δu`` of shape ``(n_u,)``. Zero when
            ``G_eff`` is too ill-conditioned to invert.
        """
        G = np.asarray(G_eff, dtype=np.float64)
        nu = np.asarray(nu_des, dtype=np.float64).reshape(-1)
        omd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        if G.ndim != 2:
            raise ValueError(f"G_eff must be 2-D; got shape {G.shape}")
        if nu.size != G.shape[0]:
            raise ValueError(
                f"nu_des must have length {G.shape[0]}, got {nu.size}"
            )
        if omd.size != G.shape[0]:
            raise ValueError(
                f"omega_dot_meas must have length {G.shape[0]}, got {omd.size}"
            )
        try:
            cond = float(np.linalg.cond(G))
        except np.linalg.LinAlgError:
            cond = float("inf")
        if not np.isfinite(cond) or cond > self.cond_threshold:
            logger.warning(
                "AIDI allocator: G is ill-conditioned (cond=%.3g); returning Δu=0",
                cond,
            )
            return np.zeros(G.shape[1], dtype=np.float64)
        G_pinv = np.linalg.pinv(G, rcond=self.rcond)
        return G_pinv @ (nu - omd)
