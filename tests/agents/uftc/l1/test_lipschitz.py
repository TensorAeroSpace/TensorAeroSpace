"""Power-iteration Lipschitz upper bound on a torch nn.Module."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l1.lipschitz import power_iteration_lipschitz


def test_linear_layer_returns_operator_norm() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 3))
    b = rng.standard_normal(4)

    class Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(torch.tensor(W, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.tensor(b, dtype=torch.float64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.W.t() + self.b

    model = Linear().eval()

    def sample() -> np.ndarray:
        return rng.standard_normal(3)

    L = power_iteration_lipschitz(model, sample, n_iter=200, n_starts=4,
                                  dtype=torch.float64)
    expected = float(np.linalg.norm(W, ord=2))
    assert abs(L - expected) / expected < 0.05


def test_returns_finite_positive_float() -> None:
    class Tanh(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x)

    rng = np.random.default_rng(1)
    L = power_iteration_lipschitz(
        Tanh().eval(), lambda: rng.standard_normal(2),
        n_iter=50, n_starts=2, dtype=torch.float64,
    )
    assert isinstance(L, float)
    assert 0.0 < L <= 1.0 + 1e-6  # tanh derivative is bounded by 1
