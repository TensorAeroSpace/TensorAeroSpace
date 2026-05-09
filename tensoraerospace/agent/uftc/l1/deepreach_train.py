"""DeepReach offline training of V_θ on the HJI residual.

Minimal Phase 2 implementation: PINN-residual + boundary loss.
The full curriculum (smoothness regulariser, multi-stage scheduling,
near-boundary sampling) is documented in the spec but not required
for Phase 2 to land. The CLI script wraps this function with argparse.

References: Bansal & Tomlin (2021), DeepReach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .value_fn import _MLP, DeepReachConfig, DeepReachValueFn


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 4096
    lr: float = 1e-3
    n_state: int = 0
    n_control: int = 0
    u_low: Optional[np.ndarray] = None
    u_high: Optional[np.ndarray] = None
    disturbance_low: Optional[np.ndarray] = None
    disturbance_high: Optional[np.ndarray] = None
    boundary_weight: float = 1.0
    seed: int = 0


def train_value_fn(
    value_cfg: DeepReachConfig,
    train_cfg: TrainingConfig,
    *,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    safe_set: Callable[[np.ndarray], float],
) -> tuple[DeepReachValueFn, dict[str, list[float]]]:
    import torch

    torch.manual_seed(int(train_cfg.seed))
    rng = np.random.default_rng(int(train_cfg.seed))
    model = _MLP.build(value_cfg, seed=int(train_cfg.seed))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    bounds = (
        np.asarray(value_cfg.state_bounds, dtype=np.float64)
        if value_cfg.state_bounds is not None
        else np.repeat([[-1.0, 1.0]], value_cfg.n_state, axis=0)
    )

    history: dict[str, list[float]] = {"loss": []}
    for epoch in range(int(train_cfg.epochs)):
        x = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(int(train_cfg.batch_size), value_cfg.n_state),
        )
        t = rng.uniform(
            0.0, value_cfg.time_horizon, size=(int(train_cfg.batch_size), 1)
        )
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        t_t = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        inp = torch.cat([x_t, t_t], dim=-1)
        v = model(inp).squeeze(-1)

        grads = torch.autograd.grad(v.sum(), [x_t, t_t], create_graph=True)
        dV_dx, dV_dt = grads[0], grads[1].squeeze(-1)

        # Min-max over u (control) and d (disturbance) computed analytically
        # for affine-in-u dynamics is hard for arbitrary callables; sample
        # finitely-many candidates and take the worst.
        u_samples = _sample_box(
            train_cfg.u_low,
            train_cfg.u_high,
            n=8,
            n_state=int(value_cfg.n_state),
            rng=rng,
        )
        d_samples = _sample_box(
            train_cfg.disturbance_low,
            train_cfg.disturbance_high,
            n=4,
            n_state=int(value_cfg.n_state),
            rng=rng,
        )
        worst = None
        for u_s in u_samples:
            for d_s in d_samples:
                f_xs = np.stack([dynamics(xi, u_s) + d_s for xi in x], axis=0)
                f_t = torch.tensor(f_xs, dtype=torch.float64)
                hji = dV_dt + (dV_dx * f_t).sum(dim=-1)
                if worst is None:
                    worst = hji
                else:
                    worst = torch.minimum(worst, hji)
        loss_hji = (worst**2).mean()

        # Boundary condition at t = T.
        x_b = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(int(train_cfg.batch_size), value_cfg.n_state),
        )
        t_b = np.full((int(train_cfg.batch_size), 1), value_cfg.time_horizon)
        inp_b = torch.tensor(np.concatenate([x_b, t_b], axis=-1), dtype=torch.float64)
        v_b = model(inp_b).squeeze(-1)
        l_b = torch.tensor([safe_set(xi) for xi in x_b], dtype=torch.float64)
        loss_bdy = ((v_b - l_b) ** 2).mean()

        loss = loss_hji + train_cfg.boundary_weight * loss_bdy
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))

    model.eval()
    return DeepReachValueFn(value_cfg, model), history


def _sample_box(
    low: Optional[np.ndarray],
    high: Optional[np.ndarray],
    *,
    n: int,
    n_state: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if low is None or high is None:
        return [np.zeros(n_state)]
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    return [rng.uniform(low, high) for _ in range(int(n))]
