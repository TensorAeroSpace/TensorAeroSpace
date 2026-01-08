"""Neural network components for the ADP (adaptive critic) agent."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Sequence[int],
    *,
    activation: type[nn.Module] = nn.Tanh,
    out_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden_sizes:
        h = int(h)
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, int(out_dim)))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Deterministic actor with tanh output scaled to env action bounds."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._body = _mlp(
            obs_dim,
            act_dim,
            hidden_sizes,
            activation=nn.Tanh,
            out_activation=nn.Tanh,
        )

        # Buffers follow the module device.
        if action_low is None or action_high is None:
            low = np.full((act_dim,), -1.0, dtype=np.float32)
            high = np.full((act_dim,), 1.0, dtype=np.float32)
        else:
            low = np.asarray(action_low, dtype=np.float32).reshape(-1)
            high = np.asarray(action_high, dtype=np.float32).reshape(-1)
            if low.shape[0] != act_dim or high.shape[0] != act_dim:
                raise ValueError(
                    f"action_low/high must have shape ({act_dim},). "
                    f"Got low={low.shape}, high={high.shape}"
                )

        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = self._body(obs)
        return a * self.action_scale + self.action_bias


class QCritic(nn.Module):
    """Critic approximating cost-to-go Q(s, a) (adaptive critic)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self._q = _mlp(
            obs_dim + act_dim,
            1,
            hidden_sizes,
            activation=activation,
            out_activation=None,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self._q(x)


def polyak_update(
    target: nn.Module, source: nn.Module, *, tau: float, params: Iterable[str] | None = None
) -> None:
    """Polyak averaging: target = (1-tau)*target + tau*source."""

    tau = float(tau)
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1].")

    with torch.no_grad():
        if params is None:
            for p_t, p in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        else:
            src = dict(source.named_parameters())
            tgt = dict(target.named_parameters())
            for name in params:
                if name not in src or name not in tgt:
                    continue
                tgt[name].data.mul_(1.0 - tau).add_(src[name].data, alpha=tau)


