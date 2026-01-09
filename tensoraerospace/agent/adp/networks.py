"""Neural network components for the ADP (adaptive critic) agent."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

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


def _mlp_body(
    in_dim: int,
    hidden_sizes: Sequence[int],
    *,
    activation: type[nn.Module] = nn.Tanh,
) -> Tuple[nn.Module, int]:
    """Create an MLP trunk (no output layer) and return (module, last_dim)."""
    prev = int(in_dim)
    if len(hidden_sizes) == 0:
        return nn.Identity(), prev

    layers: list[nn.Module] = []
    for h in hidden_sizes:
        h = int(h)
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    return nn.Sequential(*layers), prev


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
        self.register_buffer(
            "action_scale", torch.as_tensor(scale, dtype=torch.float32)
        )
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


class JCritic(nn.Module):
    """Critic approximating cost-to-go J(R) (HDP-style scalar critic)."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        trunk, last = _mlp_body(int(input_dim), hidden_sizes, activation=activation)
        self._trunk = trunk
        self._j = nn.Linear(int(last), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._trunk(x)
        return self._j(h)


class LambdaCritic(nn.Module):
    """Critic estimating lambda = dJ/dx (DHP-style).

    Input can be an observable vector R(t) (e.g., state concatenated with
    exogenous reference signals). Output is lambda w.r.t the *plant state* x.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        input_dim: int | None = None,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        in_dim = int(input_dim) if input_dim is not None else int(state_dim)
        self._net = _mlp(
            in_dim,
            int(state_dim),
            hidden_sizes,
            activation=activation,
            out_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class JLambdaCritic(nn.Module):
    """Critic approximating both J(R) and lambda_R=dJ/dR (GDHP-style, Fig. 5).

    This is the "straightforward" GDHP critic: a shared trunk with two heads:
      - scalar J
      - vector lambda_R
    """

    def __init__(
        self,
        *,
        input_dim: int,
        r_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        trunk, last = _mlp_body(int(input_dim), hidden_sizes, activation=activation)
        self._trunk = trunk
        self._j = nn.Linear(int(last), 1)
        self._lam_r = nn.Linear(int(last), int(r_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk(x)
        return self._j(h), self._lam_r(h)


class JLambdaActionCritic(nn.Module):
    """Critic for ADGDHP: outputs J and two gradient vectors (Fig. 7).

    Outputs:
      - J(R,A) scalar
      - J_R = dJ/dR  (vector)
      - J_A = dJ/dA  (vector)
    """

    def __init__(
        self,
        *,
        r_dim: int,
        a_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        in_dim = int(r_dim) + int(a_dim)
        trunk, last = _mlp_body(in_dim, hidden_sizes, activation=activation)
        self._trunk = trunk
        self._j = nn.Linear(int(last), 1)
        self._jr = nn.Linear(int(last), int(r_dim))
        self._ja = nn.Linear(int(last), int(a_dim))

    def forward(
        self, r: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([r, a], dim=-1)
        h = self._trunk(x)
        return self._j(h), self._jr(h), self._ja(h)


def polyak_update(
    target: nn.Module,
    source: nn.Module,
    *,
    tau: float,
    params: Iterable[str] | None = None,
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
