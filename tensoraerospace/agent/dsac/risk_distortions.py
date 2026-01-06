"""Risk distortion functions copied from dsac-flight."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch

DistortionFn = Callable[[torch.Tensor, float], torch.Tensor]


def normal_cdf(tau: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """CDF of the normal distribution."""
    return 0.5 * (1 + torch.erf((tau - mean) / std / np.sqrt(2)))


def normal_inverse_cdf(
    tau: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """Inverse CDF of the normal distribution."""
    return mean + std * torch.erfinv(2 * tau - 1) * np.sqrt(2)


def neutral(tau: torch.Tensor, _xi: float) -> torch.Tensor:
    """Neutral distortion returns the original quantiles."""
    return tau


def cvar(tau: torch.Tensor, xi: float) -> torch.Tensor:
    """Conditional value at risk distortion (clamped to [0,1])."""
    return torch.clamp(tau * xi, min=0.0, max=1.0)


def cpw(tau: torch.Tensor, xi: float) -> torch.Tensor:
    """Cumulative probability weighting."""
    tau_pow_xi = torch.pow(tau, xi)
    denom = torch.pow((tau_pow_xi + torch.pow(1.0 - tau, xi)), (1.0 / xi))
    return tau_pow_xi / denom


def wang(tau: torch.Tensor, xi: float) -> torch.Tensor:
    """Wang transform."""
    return normal_cdf(normal_inverse_cdf(tau) + xi)


distortion_functions: Dict[str, DistortionFn] = {
    "neutral": neutral,
    "cvar": cvar,
    "cpw": cpw,
    "wang": wang,
}
