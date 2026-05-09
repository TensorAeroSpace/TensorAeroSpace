"""CVaRₐ tail-mean and risk-gate β_t."""

from __future__ import annotations

import math

import numpy as np
import torch


def cvar_alpha_fn(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the lowest α-fraction of quantiles."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    n = int(z.shape[-1])
    k = max(1, int(math.floor(alpha * n)))
    z_sorted, _ = torch.sort(z, dim=-1)
    return z_sorted[..., :k].mean(dim=-1)


_ALARM = {"OK": 0.0, "WARN": 0.5, "CRITICAL": 1.0}


def risk_gate(
    z_quantiles: torch.Tensor,
    *,
    fdd_severity: float,
    monitor_alarm: str = "OK",
    var_target: float = 0.5,
    k_fdd: float = 0.4,
) -> float:
    var_z = float(z_quantiles.var(dim=-1).mean().item())
    g_var = float(torch.sigmoid(torch.tensor((var_z - var_target) * 5.0)).item())
    g_fdd = float(np.clip(k_fdd * float(fdd_severity), 0.0, 1.0))
    g_alarm = _ALARM.get(str(monitor_alarm), 0.0)
    return float(min(1.0, max(g_var, g_fdd, g_alarm)))
