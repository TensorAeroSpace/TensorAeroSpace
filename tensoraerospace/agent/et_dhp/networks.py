"""Actor, critic and plant-model networks for ET-DHP.

All three networks follow the recipe from Sun et al. (2022): two tanh
hidden layers, **no bias terms** (so the fixed point ``x = 0`` produces
``u = 0``), and weights initialised with ``U(-0.5, 0.5)``. The actor
output is passed through a saturating ``u_b · tanh`` layer so the
policy respects a per-channel actuator bound even during the early
random-search phase.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _build_mlp_no_bias(
    in_features: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    init_scale: float = 0.5,
) -> nn.Sequential:
    """Build a tanh MLP with no bias, uniform-init weights."""
    layers: list[nn.Module] = []
    last = in_features
    for h in hidden_sizes:
        linear = nn.Linear(last, h, bias=False)
        nn.init.uniform_(linear.weight, -init_scale, init_scale)
        layers.append(linear)
        layers.append(activation())
        last = h
    return nn.Sequential(*layers)


class PlantModelNN(nn.Module):
    """One-step predictor ``x_{k+1} = f(x_k, u_k)`` for ET-DHP.

    The network realises a discrete-time approximation of the plant
    dynamics. It is queried twice during each learning sweep:

    1. **Forward** — predict the next state.
    2. **Backward** — extract the Jacobians ``F = ∂f/∂x`` and
       ``G = ∂f/∂u`` via autograd; these feed directly into the critic
       target and the optimal-control formula used by the actor.

    Args:
        n_state: Length of the state vector ``x``.
        n_control: Length of the control vector ``u``.
        hidden_sizes: Sizes of the tanh hidden layers. Defaults to
            ``(10, 10)`` to match the reference implementation.
        init_scale: Uniform-init bound for each weight (no bias). The
            reference uses ``0.5``.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        hidden_sizes: Sequence[int] = (10, 10),
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.backbone = _build_mlp_no_bias(
            self.n_state + self.n_control, hidden_sizes, nn.Tanh, init_scale
        )
        last = hidden_sizes[-1] if hidden_sizes else self.n_state + self.n_control
        self.head = nn.Linear(last, self.n_state, bias=False)
        nn.init.uniform_(self.head.weight, -init_scale, init_scale)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        """Return the predicted next state from a concatenated ``[x; u]``."""
        return self.head(self.backbone(xu))


class ETDHPActor(nn.Module):
    """Bounded deterministic policy ``u = u_b · tanh(D_nn(x))``.

    The ``D_nn`` pre-bound output is returned alongside the bounded
    control because the ET-DHP reward contains a "bounding" term that
    depends on the raw logits — see :func:`bounded_integral_cost` in
    :mod:`tensoraerospace.agent.et_dhp.model`.

    Args:
        n_state: Length of the state vector (actor input).
        n_control: Number of actuators.
        hidden_sizes: Tanh hidden-layer sizes.
        u_bound: Per-channel absolute bound on the output. The bounding
            layer rescales ``tanh(D_nn)`` by this value.
        init_scale: Uniform-init bound for each weight.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        hidden_sizes: Sequence[int] = (10, 10),
        u_bound: float = 1.0,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.u_bound = float(u_bound)
        self.backbone = _build_mlp_no_bias(
            self.n_state, hidden_sizes, nn.Tanh, init_scale
        )
        last = hidden_sizes[-1] if hidden_sizes else self.n_state
        self.head = nn.Linear(last, self.n_control, bias=False)
        nn.init.uniform_(self.head.weight, -init_scale, init_scale)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u_bounded, D_nn)``.

        ``D_nn`` is the pre-saturation output of the last linear layer.
        The bounded control is ``u_bound · tanh(D_nn)``.
        """
        h = self.backbone(state)
        d_nn = self.head(h)
        u = self.u_bound * torch.tanh(d_nn)
        return u, d_nn


class ETDHPCritic(nn.Module):
    """Costate critic ``λ(x) = ∂J/∂x`` for DHP.

    Directly regresses the partial derivative of the cost-to-go with
    respect to the state (same dimensionality as ``x``) so the actor
    update can take a plain matrix-vector form and skip the scalar
    ``J``-head altogether — this is the original DHP formulation from
    Werbos/Prokhorov.

    Args:
        n_state: Length of the state vector ``x``.
        hidden_sizes: Tanh hidden-layer sizes.
        init_scale: Uniform-init bound for each weight.
    """

    def __init__(
        self,
        n_state: int,
        hidden_sizes: Sequence[int] = (10, 10),
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.backbone = _build_mlp_no_bias(
            self.n_state, hidden_sizes, nn.Tanh, init_scale
        )
        last = hidden_sizes[-1] if hidden_sizes else self.n_state
        self.head = nn.Linear(last, self.n_state, bias=False)
        nn.init.uniform_(self.head.weight, -init_scale, init_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``λ(x)`` of shape ``(n_state,)`` (or batched)."""
        return self.head(self.backbone(state))
