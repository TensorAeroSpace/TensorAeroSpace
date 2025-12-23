"""Utility functions for Soft Actor-Critic (SAC).

This module contains small math helpers used by the SAC implementation.
"""

import numpy as np
import torch


def create_log_gaussian(mean, log_std, t):
    """Compute log probability density of a Gaussian.

    Args:
        mean (torch.Tensor): Mean of the Gaussian.
        log_std (torch.Tensor): Log standard deviation.
        t (torch.Tensor): Value to evaluate.

    Returns:
        torch.Tensor: Log probability.
    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    length = mean.shape
    log_z = log_std
    two_pi = torch.tensor(2 * np.pi, dtype=log_std.dtype, device=log_std.device)
    z = length[-1] * torch.log(two_pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    """Compute log(sum(exp(inputs))) in a numerically stable way.

    Args:
        inputs (torch.Tensor): Input tensor.
        dim (int, optional): Dimension to reduce. If None, reduces over all
            elements.
        keepdim (bool): Whether to keep reduced dimension(s).

    Returns:
        torch.Tensor: Result tensor.
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    """Soft-update target parameters towards source parameters.

    Args:
        target (torch.nn.Module): Target model to update.
        source (torch.nn.Module): Source model to track.
        tau (float): Interpolation factor in [0, 1]. ``tau=1`` copies weights.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """Copy parameters from source model to target model."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
