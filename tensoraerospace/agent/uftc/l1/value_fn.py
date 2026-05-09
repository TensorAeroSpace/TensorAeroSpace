"""HJ value-function protocol used by the L1 shield.

A value function ``V(x)`` is *non-positive inside* the safe set, *zero on the
boundary*, and *positive outside*. The shield uses ``V`` and ``∇V`` to enforce
forward-invariance of the safe set under a CBF-style QP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HJValueFunction(Protocol):
    """Minimal contract any L1 value function must satisfy."""

    def value(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def lipschitz_const(self) -> float: ...


@dataclass
class DeepReachConfig:
    """Hyper-parameters for :class:`DeepReachValueFn`."""

    n_state: int
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "tanh"
    state_bounds: list[list[float]] | None = None
    time_horizon: float = 5.0
    safe_set_fn_name: str = "alpha_envelope"
    dt: float = 0.01
    lipschitz_n_starts: int = 8
    lipschitz_n_iter: int = 200


class _MLP:
    @staticmethod
    def build(cfg: DeepReachConfig, seed: int = 0):
        import torch
        from torch import nn

        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        in_dim = int(cfg.n_state) + 1
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh() if cfg.activation == "tanh" else nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).double()


class DeepReachValueFn:
    """Torch-MLP value function ``V_θ(x, t)`` with ``t`` fixed to ``time_horizon``."""

    def __init__(self, cfg: DeepReachConfig, model: "torch.nn.Module") -> None:
        self.cfg = cfg
        self._model = model

    @classmethod
    def from_config(cls, cfg: DeepReachConfig, *, seed: int = 0) -> "DeepReachValueFn":
        model = _MLP.build(cfg, seed=seed)
        model.eval()
        return cls(cfg, model)

    def value(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            inp = self._make_input(x)
            return float(self._model(inp).squeeze(-1).item())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        import torch

        inp = self._make_input(x).requires_grad_(True)
        v = self._model(inp).squeeze(-1)
        (g,) = torch.autograd.grad(v, inp)
        return g.detach().cpu().numpy()[: int(self.cfg.n_state)]

    def lipschitz_const(self) -> float:
        from .lipschitz import power_iteration_lipschitz

        rng = np.random.default_rng(0)
        n = int(self.cfg.n_state)
        bounds = (
            np.asarray(self.cfg.state_bounds, dtype=np.float64)
            if self.cfg.state_bounds is not None
            else np.repeat([[-1.0, 1.0]], n, axis=0)
        )

        def sample() -> np.ndarray:
            return rng.uniform(bounds[:, 0], bounds[:, 1])

        import torch

        cfg = self.cfg
        model = self._model

        class _StateOnly(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                t = torch.full((1,), float(cfg.time_horizon), dtype=x.dtype)
                return model(torch.cat([x, t], dim=-1)).squeeze(-1)

        return power_iteration_lipschitz(
            _StateOnly().eval(),
            sample,
            n_iter=cfg.lipschitz_n_iter,
            n_starts=cfg.lipschitz_n_starts,
            dtype=torch.float64,
        )

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        meta = asdict(self.cfg)
        Path(path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DeepReachValueFn":
        import torch

        path = Path(path)
        meta = json.loads(Path(path.with_suffix(".json")).read_text())
        meta["hidden_sizes"] = tuple(meta["hidden_sizes"])
        cfg = DeepReachConfig(**meta)
        model = _MLP.build(cfg, seed=0)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return cls(cfg, model)

    def _make_input(self, x: np.ndarray) -> "torch.Tensor":
        import torch

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != int(self.cfg.n_state):
            raise ValueError(f"expected x of size {self.cfg.n_state}, got {x.size}")
        t = np.array([float(self.cfg.time_horizon)], dtype=np.float64)
        inp = torch.tensor(np.concatenate([x, t]), dtype=torch.float64)
        return inp
