"""Trim-free longitudinal reference wrapper for L4 D-SAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LongitudinalTrimFreeConfig:
    V_idx: int
    gamma_idx: int
    alpha_idx: int
    q_idx: int
    enabled: bool = False


class LongitudinalTrimFreeWrapper:
    """Replace ``alpha_target`` and ``q_target`` in the base reference with
    actor output. Pilot-supplied ``V_target`` and ``gamma_target`` are
    preserved verbatim.
    """

    def __init__(self, cfg: LongitudinalTrimFreeConfig) -> None:
        self.cfg = cfg

    def apply(
        self,
        r_tilde_actor: np.ndarray,
        *,
        x_obs: np.ndarray,
        base_reference: np.ndarray,
    ) -> np.ndarray:
        if not self.cfg.enabled:
            return np.asarray(base_reference, dtype=np.float64).copy()
        out = np.asarray(base_reference, dtype=np.float64).copy()
        out[self.cfg.alpha_idx] = float(r_tilde_actor[0])
        out[self.cfg.q_idx] = float(r_tilde_actor[1])
        return out
