"""Differentiable MPC utilities (torch/autograd).

This module provides a small, dependency-free MPC solver that works with:
- Linear discrete dynamics (via matrices), and
- Learned torch dynamics models f(x, u) -> x_next.

The focus is correctness, predictable shapes, and compatibility with
TensorAeroSpace environments (notably the B747 family).
"""

from __future__ import annotations

import copy
import datetime
import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm  # type: ignore[import-untyped]

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)
from ..sac.replay_memory import ReplayMemory

TensorLike = Union[np.ndarray, torch.Tensor]
ExtraCostFn = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
]


@dataclass(frozen=True)
class MPCTrackingExtraCostConfig:
    """Extra cost for generic reference tracking.

    This is applied in addition to the quadratic MPC objective.
    Values are dimensionless weights.
    """

    w_du: float = 0.0
    w_jerk: float = 0.0


@dataclass(frozen=True)
class MPCStepResponseExtraCostConfig:
    """Extra cost tuned for step response (overshoot/settling/osc/jerk).

    All thresholds must be in the SAME UNITS as the tracked state component
    inside x_seq/x_ref (for B747 internal model this is radians).
    """

    tracked_idx: int = -1
    rate_idx: int | None = None

    dt: float = 0.1

    ref_change_threshold: float = float(np.deg2rad(0.10))
    min_step_amp: float = float(np.deg2rad(0.50))

    overshoot_limit: float = float(np.deg2rad(0.05))
    settle_band: float = float(np.deg2rad(0.10))
    settle_band_min: float = float(np.deg2rad(0.05))
    settle_band_ratio: float = 0.01
    settle_time_target_s: float = 1.0

    rate_settle: float = float(np.deg2rad(0.25))

    w_overshoot: float = 8_000.0
    w_time: float = 800.0
    w_settle: float = 8_000.0
    w_sse_steady: float = 40_000.0
    w_osc: float = 500.0
    w_jerk: float = 50.0

    w_du_steady: float = 80.0
    w_jerk_steady: float = 800.0

    @classmethod
    def from_degrees(
        cls,
        *,
        tracked_idx: int = -1,
        rate_idx: int | None = None,
        dt: float = 0.1,
        ref_change_threshold_deg: float = 0.10,
        min_step_amp_deg: float = 0.50,
        overshoot_limit_deg: float = 0.05,
        settle_band_deg: float = 0.10,
        settle_band_min_deg: float = 0.05,
        settle_band_ratio: float = 0.01,
        settle_time_target_s: float = 1.0,
        rate_settle_deg_s: float = 0.25,
        w_overshoot: float = 8_000.0,
        w_time: float = 800.0,
        w_settle: float = 8_000.0,
        w_sse_steady: float = 40_000.0,
        w_osc: float = 500.0,
        w_jerk: float = 50.0,
        w_du_steady: float = 80.0,
        w_jerk_steady: float = 800.0,
    ) -> "MPCStepResponseExtraCostConfig":
        return cls(
            tracked_idx=int(tracked_idx),
            rate_idx=None if rate_idx is None else int(rate_idx),
            dt=float(dt),
            ref_change_threshold=float(np.deg2rad(ref_change_threshold_deg)),
            min_step_amp=float(np.deg2rad(min_step_amp_deg)),
            overshoot_limit=float(np.deg2rad(overshoot_limit_deg)),
            settle_band=float(np.deg2rad(settle_band_deg)),
            settle_band_min=float(np.deg2rad(settle_band_min_deg)),
            settle_band_ratio=float(settle_band_ratio),
            settle_time_target_s=float(settle_time_target_s),
            rate_settle=float(np.deg2rad(rate_settle_deg_s)),
            w_overshoot=float(w_overshoot),
            w_time=float(w_time),
            w_settle=float(w_settle),
            w_sse_steady=float(w_sse_steady),
            w_osc=float(w_osc),
            w_jerk=float(w_jerk),
            w_du_steady=float(w_du_steady),
            w_jerk_steady=float(w_jerk_steady),
        )


@dataclass(frozen=True)
class MPCWeights:
    """Quadratic weights for the standard MPC objective.

    The objective is:
        sum_{t=0..N-1} ||x_{t+1} - x_ref_{t+1}||_Q^2
                      + ||u_t||_R^2
                      + ||u_t - u_{t-1}||_S^2
        + terminal_weight * ||x_N - x_ref_N||_Q^2

    Q, R, S are interpreted as diagonal weights (vectors). This keeps the API
    simple and fast enough for small horizons without extra dependencies.
    """

    Q_diag: TensorLike
    R_diag: TensorLike
    S_diag: TensorLike | None = None
    terminal_weight: float = 1.0


@dataclass(frozen=True)
class MPCConstraints:
    """Box constraints for control and rate limits.

    All bounds are interpreted element-wise (per control dimension).
    """

    u_min: TensorLike | None = None
    u_max: TensorLike | None = None
    du_min: TensorLike | None = None
    du_max: TensorLike | None = None


@dataclass(frozen=True)
class MPCSolveResult:
    """Result bundle for MPC solve."""

    u0: np.ndarray  # (action_dim,)
    u_seq: np.ndarray  # (horizon, action_dim)
    x_seq: np.ndarray  # (horizon + 1, state_dim)
    final_cost: float
    iters: int


def _to_serializable(obj: Any) -> Any:
    """Recursively move tensors/arrays to CPU lists for JSON friendliness."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _to_cpu_detached(obj: Any) -> Any:
    """Recursively detach tensors to CPU (used for safe checkpointing)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu_detached(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_detached(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(v) for v in obj)
    return obj


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state_dict to CPU for portable saving."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _optimizer_state_dict_cpu(opt: torch.optim.Optimizer) -> dict[str, Any]:
    """Deepcopy optimizer state and move all tensors to CPU."""
    sd = copy.deepcopy(opt.state_dict())
    sd["state"] = _to_cpu_detached(sd.get("state", {}))
    return sd


def _dtype_from_string(
    name: str | None, default: torch.dtype = torch.float32
) -> torch.dtype:
    """Convert torch dtype name string back to dtype object."""
    if name is None:
        return default
    try:
        attr = name.split(".")[-1]
        dt = getattr(torch, attr)
        if isinstance(dt, torch.dtype):
            return dt
    except Exception:
        pass
    return default


def _safe_device_str(device_str: str | torch.device | None) -> str:
    """Return a device string downgraded to CPU when CUDA/MPS is unavailable."""
    if device_str is None:
        return "cpu"
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if dev.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return "cpu"
    return dev.type if dev.index is None else f"{dev.type}:{dev.index}"


def _to_2d(x: TensorLike, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    xt = torch.as_tensor(x, dtype=dtype, device=device)
    if xt.ndim == 1:
        return xt.unsqueeze(0)
    if xt.ndim == 2:
        return xt
    # Flatten any extra dims into features (batch, features)
    return xt.reshape(xt.shape[0], -1)


def _to_1d(x: TensorLike, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    xt = torch.as_tensor(x, dtype=dtype, device=device)
    return xt.reshape(-1)


class MPC:
    """Projected-gradient MPC over a differentiable dynamics model.

    This solver optimizes a control sequence U using torch/autograd and applies
    hard constraints via projection (clamp + sequential rate limiting).
    """

    def __init__(
        self,
        *,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state_dim: int,
        action_dim: int,
        horizon: int = 20,
        weights: MPCWeights,
        constraints: MPCConstraints | None = None,
        extra_cost_fn: ExtraCostFn | None = None,
        iters: int = 60,
        lr: float = 0.05,
        optimizer: Literal["adam", "sgd"] = "adam",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        warm_start: bool = True,
        track_best: bool = True,
        best_check_every: int = 1,
        compile_dynamics: bool = False,
        compile_mode: str = "reduce-overhead",
        seed: int | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        self.weights = weights
        self.constraints = constraints or MPCConstraints()
        self.extra_cost_fn = extra_cost_fn
        self.iters = int(iters)
        self.lr = float(lr)
        self.optimizer = optimizer
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = dtype
        self.warm_start = bool(warm_start)
        self.track_best = bool(track_best)
        self.best_check_every = int(best_check_every)
        if self.best_check_every < 1:
            raise ValueError("best_check_every must be >= 1")
        self.compile_dynamics = bool(compile_dynamics)
        self.compile_mode = str(compile_mode)

        if seed is not None:
            torch.manual_seed(int(seed))

        if self.compile_dynamics:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "compile_dynamics=True requires torch.compile (PyTorch 2.x)."
                )
            try:
                compiled_dyn = torch.compile(
                    self.dynamics, mode=self.compile_mode
                )  # type: ignore[attr-defined]

                # IMPORTANT: compiled CUDA graphs can reuse output buffers across
                # invocations. MPC rollouts keep intermediate states, so we must
                # clone to avoid "output overwritten" errors.
                def _dyn_wrapped(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
                    out = compiled_dyn(x, u)
                    if out.is_cuda:
                        return out.clone()
                    return out

                self.dynamics = _dyn_wrapped
            except Exception as e:  # pragma: no cover - depends on torch backend
                raise RuntimeError(
                    f"torch.compile failed for dynamics (mode={self.compile_mode!r}). "
                    "Try compile_dynamics=False or a different compile_mode."
                ) from e

        # Warm start buffer in *projected* space (horizon, action_dim)
        self._u_warm: torch.Tensor | None = None

        # Pre-pack weights to tensors
        self._Q = _to_1d(weights.Q_diag, dtype=self.dtype, device=self.device)
        self._R = _to_1d(weights.R_diag, dtype=self.dtype, device=self.device)
        self._S = (
            None
            if weights.S_diag is None
            else _to_1d(weights.S_diag, dtype=self.dtype, device=self.device)
        )
        self._Q_row = self._Q.reshape(1, -1)
        self._R_row = self._R.reshape(1, -1)
        self._S_row = None if self._S is None else self._S.reshape(1, -1)
        self._terminal_weight = float(weights.terminal_weight)

        if self._Q.numel() != self.state_dim:
            raise ValueError(
                f"Q_diag must have length state_dim={self.state_dim}, "
                f"got {self._Q.numel()}"
            )
        if self._R.numel() != self.action_dim:
            raise ValueError(
                f"R_diag must have length action_dim={self.action_dim}, "
                f"got {self._R.numel()}"
            )
        if self._S is not None and self._S.numel() != self.action_dim:
            raise ValueError(
                f"S_diag must have length action_dim={self.action_dim}, "
                f"got {self._S.numel()}"
            )

        # Constraints as tensors (broadcast-friendly)
        self._u_min = (
            None
            if self.constraints.u_min is None
            else _to_1d(self.constraints.u_min, dtype=self.dtype, device=self.device)
        )
        self._u_max = (
            None
            if self.constraints.u_max is None
            else _to_1d(self.constraints.u_max, dtype=self.dtype, device=self.device)
        )
        self._du_min = (
            None
            if self.constraints.du_min is None
            else _to_1d(self.constraints.du_min, dtype=self.dtype, device=self.device)
        )
        self._du_max = (
            None
            if self.constraints.du_max is None
            else _to_1d(self.constraints.du_max, dtype=self.dtype, device=self.device)
        )

        for name, bound in [
            ("u_min", self._u_min),
            ("u_max", self._u_max),
            ("du_min", self._du_min),
            ("du_max", self._du_max),
        ]:
            if bound is not None and bound.numel() not in (1, self.action_dim):
                raise ValueError(
                    f"{name} must be scalar or length "
                    f"action_dim={self.action_dim}, "
                    f"got {bound.numel()}"
                )

    def reset(self) -> None:
        """Reset warm-start state."""

        self._u_warm = None

    def _project_u(
        self, u_raw: torch.Tensor, u_prev: torch.Tensor | None
    ) -> torch.Tensor:
        """Project action sequence into constraints.

        Args:
            u_raw: (horizon, action_dim)
            u_prev: (action_dim,) or None
        """

        u = u_raw

        # Magnitude bounds
        if self._u_min is not None or self._u_max is not None:
            u = torch.clamp(u, min=self._u_min, max=self._u_max)

        # Rate bounds (sequentially)
        if (
            self._du_min is not None or self._du_max is not None
        ) and u_prev is not None:
            prev = u_prev.reshape(-1)
            seq: list[torch.Tensor] = []
            for t in range(self.horizon):
                ut = u[t]
                du = torch.clamp(ut - prev, min=self._du_min, max=self._du_max)
                ut_proj = prev + du
                # Re-apply magnitude bounds after rate projection (important)
                if self._u_min is not None or self._u_max is not None:
                    ut_proj = torch.clamp(ut_proj, min=self._u_min, max=self._u_max)
                seq.append(ut_proj)
                prev = ut_proj
            u = torch.stack(seq, dim=0)

        return u

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """Roll out dynamics for a single trajectory.

        Args:
            x0: (1, state_dim)
            u_seq: (horizon, action_dim)

        Returns:
            x_seq: (horizon + 1, state_dim) including x0.
        """

        x = x0
        xs = [x0.reshape(-1)]
        for t in range(self.horizon):
            ut = u_seq[t].reshape(1, -1)
            x = self.dynamics(x, ut)
            xs.append(x.reshape(-1))
        return torch.stack(xs, dim=0)

    def _compute_cost(
        self,
        *,
        x_seq: torch.Tensor,
        u_seq: torch.Tensor,
        x_ref: torch.Tensor | None,
        u_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute MPC objective for an already-rolled-out trajectory."""

        cost_t = u_seq.new_zeros(())

        # stage cost for x_{t+1}
        if x_ref is not None:
            err = x_seq[1:] - x_ref[1:]
            cost_t = cost_t + (err.pow(2) * self._Q_row).sum()

        # control effort
        cost_t = cost_t + (u_seq.pow(2) * self._R_row).sum()

        # delta-u penalty
        if self._S_row is not None:
            if u_prev is None:
                du = u_seq[1:] - u_seq[:-1]
                cost_t = cost_t + (du.pow(2) * self._S_row).sum()
            else:
                u_prev_row = u_prev.reshape(1, -1)
                du0 = u_seq[0:1] - u_prev_row
                du_rest = u_seq[1:] - u_seq[:-1]
                cost_t = cost_t + (du0.pow(2) * self._S_row).sum()
                cost_t = cost_t + (du_rest.pow(2) * self._S_row).sum()

        # terminal cost (same Q diag, scaled)
        if x_ref is not None and float(self._terminal_weight) != 0.0:
            terr = x_seq[-1] - x_ref[-1]
            cost_t = cost_t + float(self._terminal_weight) * (
                (terr.pow(2) * self._Q).sum()
            )

        # Optional extra cost (e.g., step-response constraints)
        if self.extra_cost_fn is not None:
            extra = self.extra_cost_fn(x_seq, u_seq, x_ref)
            extra_t = torch.as_tensor(
                extra, dtype=self.dtype, device=self.device
            ).mean()
            cost_t = cost_t + extra_t

        return cost_t

    def solve(
        self,
        *,
        x0: TensorLike,
        x_ref: TensorLike | None = None,
        u_prev: TensorLike | None = None,
    ) -> MPCSolveResult:
        """Solve MPC problem for the current state.

        Args:
            x0: Current state, shape (state_dim,) or (state_dim, 1) or
                (1, state_dim).
            x_ref: Optional reference trajectory for states.
                Expected shape is (horizon+1, state_dim) or
                (horizon, state_dim).
                If provided as (horizon, state_dim), it is interpreted
                as targets for x_{t+1} and a terminal target is appended by
                repeating the last row.
            u_prev: Optional previous control input, shape (action_dim,).

        Returns:
            MPCSolveResult: contains the first control input and the
                predicted trajectory.
        """

        x0_t = _to_2d(x0, dtype=self.dtype, device=self.device)
        x0_t = x0_t.reshape(1, -1)
        if x0_t.shape[1] != self.state_dim:
            raise ValueError(
                f"x0 must have state_dim={self.state_dim} features, "
                f"got shape {tuple(x0_t.shape)}"
            )

        u_prev_t = None
        if u_prev is not None:
            u_prev_t = _to_1d(u_prev, dtype=self.dtype, device=self.device)
            if u_prev_t.numel() != self.action_dim:
                raise ValueError(
                    f"u_prev must have action_dim={self.action_dim} elements, "
                    f"got {u_prev_t.numel()}"
                )

        x_ref_t = None
        if x_ref is not None:
            xr = _to_2d(x_ref, dtype=self.dtype, device=self.device)
            if xr.shape[1] != self.state_dim:
                raise ValueError(
                    f"x_ref must have state_dim={self.state_dim} features, "
                    f"got {tuple(xr.shape)}"
                )
            if xr.shape[0] == self.horizon:
                xr = torch.cat([xr, xr[-1:].clone()], dim=0)
            if xr.shape[0] != self.horizon + 1:
                raise ValueError(
                    f"x_ref must have length horizon+1={self.horizon+1} "
                    "(or horizon), "
                    f"got {xr.shape[0]}"
                )
            x_ref_t = xr

        # Initialize control sequence
        if self.warm_start and self._u_warm is not None:
            u0 = torch.cat([self._u_warm[1:], self._u_warm[-1:]], dim=0)
        else:
            u0 = torch.zeros(
                (self.horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )

        u_param = torch.nn.Parameter(u0.clone())
        opt: torch.optim.Optimizer
        if self.optimizer == "adam":
            opt = torch.optim.Adam([u_param], lr=self.lr)
        elif self.optimizer == "sgd":
            opt = torch.optim.SGD([u_param], lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}")

        best_u: torch.Tensor | None = None
        best_cost: float = float("inf")

        for i in range(self.iters):
            opt.zero_grad(set_to_none=True)

            u_proj = self._project_u(u_param, u_prev_t)
            x_seq = self._rollout(x0_t, u_proj)

            cost_t = self._compute_cost(
                x_seq=x_seq, u_seq=u_proj, x_ref=x_ref_t, u_prev=u_prev_t
            )

            # IMPORTANT: we only need gradients w.r.t. u_param
            # (control sequence),
            # not w.r.t. dynamics model parameters. Using autograd.grad avoids
            # accumulating gradients in the dynamics model and is faster.
            grad_u = torch.autograd.grad(
                cost_t,
                u_param,
                retain_graph=False,
                create_graph=False,
            )[0]
            u_param.grad = grad_u
            opt.step()

            if self.track_best and (
                i % self.best_check_every == 0 or i == self.iters - 1
            ):
                with torch.no_grad():
                    cost_val = float(cost_t.detach().item())
                    if cost_val < best_cost:
                        best_cost = cost_val
                        best_u = u_proj.detach().clone()

        if best_u is None:
            # Track-best disabled (or nothing checked): use final iterate.
            with torch.no_grad():
                best_u = self._project_u(u_param, u_prev_t).detach().clone()
                best_cost_t = self._compute_cost(
                    x_seq=self._rollout(x0_t, best_u),
                    u_seq=best_u,
                    x_ref=x_ref_t,
                    u_prev=u_prev_t,
                )
                best_cost = float(best_cost_t.detach().item())

        with torch.no_grad():
            best_x = self._rollout(x0_t, best_u).detach().clone()

        self._u_warm = best_u.detach().clone()

        u0_np = best_u[0].detach().cpu().numpy().reshape(-1)
        return MPCSolveResult(
            u0=u0_np.astype(np.float32),
            u_seq=best_u.detach().cpu().numpy().astype(np.float32),
            x_seq=best_x.detach().cpu().numpy().astype(np.float32),
            final_cost=float(best_cost),
            iters=int(self.iters),
        )


@dataclass
class MPCStandardScaler:
    """Simple per-feature standardization helper (mean/std).

    The scaler lives on a torch device and is used both for training and for
    MPC rollouts through a learned dynamics model.
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def identity(
        cls, dim: int, *, device: torch.device, dtype: torch.dtype
    ) -> "MPCStandardScaler":
        mean = torch.zeros((int(dim),), device=device, dtype=dtype)
        std = torch.ones((int(dim),), device=device, dtype=dtype)
        return cls(mean=mean, std=std)

    @classmethod
    def fit(
        cls,
        x: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> "MPCStandardScaler":
        if x.ndim != 2:
            raise ValueError(f"fit expects a 2-D tensor (N, D). Got {tuple(x.shape)}")
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        std = torch.where(std < float(eps), torch.ones_like(std), std)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.reshape(1, -1) + self.mean.reshape(1, -1)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "MPCStandardScaler":
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        return self


class OneStepMLP(nn.Module):
    """A small MLP for one-step dynamics learning.

    Expected IO:
        in:  concatenated [x, u] of shape (B, state_dim + action_dim)
        out: either delta-x or x_next of shape (B, state_dim)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        activation: Literal["relu", "tanh", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim/output_dim must be positive")

        if len(hidden_layers) < 1:
            raise ValueError("hidden_layers must be non-empty")
        hs = [int(h) for h in hidden_layers]
        if any(h <= 0 for h in hs):
            raise ValueError(f"hidden_layers must be positive, got {hidden_layers}")

        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

        self.hidden_layers = hs
        self.activation_name = activation

        layers: list[nn.Module] = []
        d = input_dim
        for h in hs:
            layers.append(nn.Linear(d, h))
            layers.append(act)
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass."""
        if xu.ndim != 2:
            xu = xu.view(xu.shape[0], -1)
        return cast(torch.Tensor, self.net(xu))


class MPCAgent(BaseRLModel):
    """DSAC-like wrapper around `MPC` with learned dynamics.

    Goals:
    - Work with different Gymnasium-like environments (infer dims and bounds)
    - Accept a neural dynamics model at init (or build a default MLP)
    - Provide DSAC-ish ergonomics: buffer, collect_data(), train_dynamics(),
      select_action()
    """

    def __init__(
        self,
        env: Any,
        *,
        state_dim: int | None = None,
        action_dim: int | None = None,
        # --- MPC config ---
        horizon: int = 20,
        weights: MPCWeights | None = None,
        constraints: MPCConstraints | None = None,
        tracking_type: Literal["tracking", "step_response"] = "tracking",
        tracking_config: MPCTrackingExtraCostConfig | None = None,
        step_response_config: MPCStepResponseExtraCostConfig | None = None,
        extra_cost_fn: ExtraCostFn | None = None,
        iters: int = 60,
        mpc_lr: float = 0.05,
        mpc_optimizer: Literal["adam", "sgd"] = "adam",
        warm_start: bool = True,
        mpc_track_best: bool = True,
        mpc_best_check_every: int = 1,
        mpc_compile_dynamics: bool = False,
        mpc_compile_mode: str = "reduce-overhead",
        # --- Learned dynamics config ---
        model: nn.Module | None = None,
        model_predict_delta: bool = True,
        hidden_layers: Sequence[int] = (256, 256),
        activation: Literal["relu", "tanh", "gelu"] = "relu",
        normalize: bool = True,
        dynamics_lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = 1.0,
        # --- Data / buffer ---
        memory_capacity: int = 200_000,
        # --- Env adapters (optional) ---
        obs_to_state: Callable[[Any, Any], np.ndarray] | None = None,
        action_to_env: Callable[[np.ndarray], np.ndarray] | None = None,
        action_from_env: Callable[[np.ndarray], np.ndarray] | None = None,
        # --- Global ---
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.env = env
        self.seed = int(seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device(device)
        self.dtype = dtype

        # Keep MPC config to allow rebuilding (e.g., for device switches)
        self._mpc_horizon = int(horizon)
        self._mpc_weights: MPCWeights | None = None
        self._mpc_constraints: MPCConstraints | None = None
        self._mpc_extra_cost_fn: ExtraCostFn | None = None
        self._mpc_iters = int(iters)
        self._mpc_lr = float(mpc_lr)
        self._mpc_optimizer: Literal["adam", "sgd"] = mpc_optimizer
        self._mpc_warm_start = bool(warm_start)
        self._mpc_track_best = bool(mpc_track_best)
        self._mpc_best_check_every = int(mpc_best_check_every)
        if self._mpc_best_check_every < 1:
            raise ValueError("mpc_best_check_every must be >= 1")
        self._mpc_compile_dynamics = bool(mpc_compile_dynamics)
        self._mpc_compile_mode = str(mpc_compile_mode)

        # Tracking mode configuration
        self.tracking_type: Literal["tracking", "step_response"] = tracking_type
        self.tracking_config = (
            MPCTrackingExtraCostConfig() if tracking_config is None else tracking_config
        )
        self.step_response_config = step_response_config
        self._u_lim: torch.Tensor | None = None
        self._user_extra_cost_fn = extra_cost_fn

        # --- Infer dims ---
        env_obs_dim = 0
        try:
            obs_shape = getattr(self.env.observation_space, "shape", (0,))
            env_obs_dim = int(np.prod(obs_shape))
        except (AttributeError, TypeError, ValueError):
            env_obs_dim = 0

        env_act_dim = 0
        try:
            act_shape = getattr(self.env.action_space, "shape", (0,))
            env_act_dim = int(np.prod(act_shape))
        except (AttributeError, TypeError, ValueError):
            env_act_dim = 0

        def _vec_len(v: Any) -> int:
            if torch.is_tensor(v):
                return int(v.numel())
            return int(np.asarray(v).reshape(-1).size)

        if state_dim is not None:
            self.state_dim = int(state_dim)
        elif weights is not None:
            self.state_dim = _vec_len(weights.Q_diag)
        else:
            self.state_dim = int(env_obs_dim)

        if action_dim is not None:
            self.action_dim = int(action_dim)
        elif weights is not None:
            self.action_dim = _vec_len(weights.R_diag)
        elif constraints is not None:
            ref = (
                constraints.u_min
                if constraints.u_min is not None
                else constraints.u_max
            )
            self.action_dim = 0 if ref is None else _vec_len(ref)
        else:
            self.action_dim = int(env_act_dim)

        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError(
                "MPCAgent could not infer state/action dims. "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}. "
                "Pass state_dim=... and action_dim=... explicitly."
            )

        if env_obs_dim > 0 and self.state_dim != env_obs_dim and obs_to_state is None:
            raise ValueError(
                "state_dim differs from env.observation_space. "
                f"state_dim={self.state_dim}, env_obs_dim={env_obs_dim}. "
                "Provide obs_to_state=... (to use env internal state) or "
                "set state_dim to match env observation."
            )

        if (
            env_act_dim > 0
            and self.action_dim != env_act_dim
            and (action_to_env is None or action_from_env is None)
        ):
            raise ValueError(
                "action_dim differs from env.action_space. "
                f"action_dim={self.action_dim}, env_act_dim={env_act_dim}. "
                "Provide action_to_env/action_from_env adapters or "
                "set action_dim to match env action."
            )

        # --- Adapters ---
        self._obs_to_state = obs_to_state
        self._action_to_env = action_to_env
        self._action_from_env = action_from_env

        # --- Action bounds (env units) ---
        self._a_low_env: np.ndarray | None = None
        self._a_high_env: np.ndarray | None = None
        try:
            low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
            if low.size == self.action_dim and high.size == self.action_dim:
                self._a_low_env = low
                self._a_high_env = high
        except (AttributeError, TypeError, ValueError):
            self._a_low_env = None
            self._a_high_env = None

        # --- Learned model ---
        self.normalize = bool(normalize)
        self.model_predict_delta = bool(model_predict_delta)
        self._hidden_layers_cfg: Sequence[int] = tuple(int(h) for h in hidden_layers)
        self._activation_name: Literal["relu", "tanh", "gelu"] = activation

        if model is None:
            model = OneStepMLP(
                input_dim=int(self.state_dim + self.action_dim),
                output_dim=int(self.state_dim),
                hidden_layers=self._hidden_layers_cfg,
                activation=self._activation_name,
            )
        self.model: nn.Module = model.to(self.device, dtype=self.dtype)

        self.model_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=float(dynamics_lr),
            weight_decay=float(weight_decay),
        )
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

        # --- Normalizers (identity until fitted) ---
        self.x_scaler = MPCStandardScaler.identity(
            self.state_dim, device=self.device, dtype=self.dtype
        )
        self.u_scaler = MPCStandardScaler.identity(
            self.action_dim, device=self.device, dtype=self.dtype
        )
        self.y_scaler = MPCStandardScaler.identity(
            self.state_dim, device=self.device, dtype=self.dtype
        )

        # --- Buffer (DSAC-style) ---
        self.memory = ReplayMemory(int(memory_capacity), seed=self.seed)

        # --- MPC weights/constraints defaults ---
        if weights is None:
            weights = MPCWeights(
                Q_diag=np.ones((self.state_dim,), dtype=np.float32),
                R_diag=np.full((self.action_dim,), 1e-2, dtype=np.float32),
                S_diag=np.full((self.action_dim,), 1e-1, dtype=np.float32),
                terminal_weight=1.0,
            )
        if (
            constraints is None
            and self._a_low_env is not None
            and self._a_high_env is not None
        ):
            constraints = MPCConstraints(
                u_min=self._a_low_env.copy(),
                u_max=self._a_high_env.copy(),
            )

        self._u_prev: np.ndarray | None = None

        # Save final weights/constraints and build MPC
        self._mpc_weights = weights
        self._mpc_constraints = constraints

        # Prepare limits for extra costs and pick the extra_cost_fn.
        self._update_u_limit_tensor()

        if self.step_response_config is None:
            # Reasonable default for many aerospace states: last is angle,
            # previous is rate (B747: [u, w, q, theta]).
            dt0 = float(
                getattr(
                    getattr(self.env, "unwrapped", self.env),
                    "dt",
                    getattr(self.env, "dt", 0.1),
                )
            )
            tracked_idx = int(self.state_dim - 1)
            rate_idx = int(self.state_dim - 2) if self.state_dim >= 3 else None
            self.step_response_config = MPCStepResponseExtraCostConfig(
                tracked_idx=tracked_idx,
                rate_idx=rate_idx,
                dt=float(dt0),
            )

        # If user provided a custom extra_cost_fn, keep it.
        # Otherwise select by type.
        if self._user_extra_cost_fn is not None:
            self._mpc_extra_cost_fn = self._user_extra_cost_fn
        else:
            self._mpc_extra_cost_fn = self._make_extra_cost_fn()

        self._rebuild_mpc()

    def _rebuild_mpc(self) -> None:
        """(Re)build internal MPC with current device/dtype."""

        if self._mpc_weights is None:
            raise RuntimeError("Internal error: MPC weights are not set")

        # Differentiable dynamics wrapper around the NN (+ normalization)
        def dyn(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            # x: (B, state_dim), u: (B, action_dim)
            if x.ndim != 2:
                x = x.view(x.shape[0], -1)
            if u.ndim != 2:
                u = u.view(u.shape[0], -1)

            if self.normalize:
                xn = self.x_scaler.transform(x)
                un = self.u_scaler.transform(u)
                xu = torch.cat([xn, un], dim=-1)
                y_hat_n = cast(torch.Tensor, self.model(xu))
                y_hat = self.y_scaler.inverse(y_hat_n)
            else:
                xu = torch.cat([x, u], dim=-1)
                y_hat = cast(torch.Tensor, self.model(xu))

            if self.model_predict_delta:
                return x + y_hat
            return y_hat

        self.mpc = MPC(
            dynamics=dyn,
            state_dim=int(self.state_dim),
            action_dim=int(self.action_dim),
            horizon=int(self._mpc_horizon),
            weights=self._mpc_weights,
            constraints=self._mpc_constraints,
            extra_cost_fn=self._mpc_extra_cost_fn,
            iters=int(self._mpc_iters),
            lr=float(self._mpc_lr),
            optimizer=self._mpc_optimizer,
            device=self.device,
            dtype=self.dtype,
            warm_start=bool(self._mpc_warm_start),
            track_best=bool(self._mpc_track_best),
            best_check_every=int(self._mpc_best_check_every),
            compile_dynamics=bool(self._mpc_compile_dynamics),
            compile_mode=str(self._mpc_compile_mode),
            seed=self.seed,
        )

    def _update_u_limit_tensor(self) -> None:
        """Update internal u_limit tensor.

        Used for normalized smoothness costs.
        """

        u_lim = torch.ones(
            (int(self.action_dim),), device=self.device, dtype=self.dtype
        )
        c = self._mpc_constraints
        if c is None:
            self._u_lim = u_lim
            return

        def _as_bound(x: TensorLike | None) -> torch.Tensor | None:
            if x is None:
                return None
            t = torch.as_tensor(x, device=self.device, dtype=self.dtype).reshape(-1)
            if t.numel() == 1:
                t = t.repeat(int(self.action_dim))
            return t

        u_min_t = _as_bound(c.u_min)
        u_max_t = _as_bound(c.u_max)

        if u_min_t is None and u_max_t is None:
            self._u_lim = u_lim
            return

        if u_min_t is None:
            assert u_max_t is not None
            u_lim = u_max_t.abs()
        elif u_max_t is None:
            assert u_min_t is not None
            u_lim = u_min_t.abs()
        else:
            u_lim = torch.maximum(u_min_t.abs(), u_max_t.abs())

        u_lim = torch.where(u_lim < 1e-6, torch.ones_like(u_lim), u_lim)
        self._u_lim = u_lim

    def _resolve_state_idx(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx = int(self.state_dim) + idx
        if idx < 0 or idx >= int(self.state_dim):
            raise ValueError(
                f"Invalid state index {idx} for state_dim={self.state_dim}"
            )
        return idx

    def _make_extra_cost_fn(self) -> ExtraCostFn | None:
        if self.tracking_type == "step_response":
            assert self.step_response_config is not None
            return self._make_step_response_extra_cost(self.step_response_config)
        return self._make_tracking_extra_cost(self.tracking_config)

    def set_tracking_type(
        self,
        tracking_type: Literal["tracking", "step_response"],
        *,
        tracking_config: MPCTrackingExtraCostConfig | None = None,
        step_response_config: MPCStepResponseExtraCostConfig | None = None,
    ) -> None:
        """Switch extra-cost mode (tracking vs step_response)."""

        self.tracking_type = tracking_type
        if tracking_config is not None:
            self.tracking_config = tracking_config
        if step_response_config is not None:
            self.step_response_config = step_response_config
        if self.step_response_config is None:
            rate_idx = int(self.state_dim - 2) if self.state_dim >= 3 else None
            self.step_response_config = MPCStepResponseExtraCostConfig(
                tracked_idx=int(self.state_dim - 1),
                rate_idx=rate_idx,
                dt=float(
                    getattr(
                        getattr(self.env, "unwrapped", self.env),
                        "dt",
                        getattr(self.env, "dt", 0.1),
                    )
                ),
            )

        self._update_u_limit_tensor()
        self._mpc_extra_cost_fn = self._make_extra_cost_fn()
        self.mpc.extra_cost_fn = self._mpc_extra_cost_fn

    def _make_tracking_extra_cost(self, cfg: MPCTrackingExtraCostConfig) -> ExtraCostFn:
        w_du = float(cfg.w_du)
        w_jerk = float(cfg.w_jerk)

        def extra(
            x_seq: torch.Tensor,
            u_seq: torch.Tensor,
            x_ref: torch.Tensor | None,
        ) -> torch.Tensor:
            _ = x_seq
            _ = x_ref
            if u_seq.numel() == 0 or self._u_lim is None:
                return u_seq.new_tensor(0.0)

            u_lim = self._u_lim.reshape(1, -1)
            u_norm = torch.clamp(u_seq / u_lim, -1.0, 1.0)
            cost = u_seq.new_tensor(0.0)

            if w_du != 0.0 and u_norm.shape[0] >= 2:
                du = u_norm[1:] - u_norm[:-1]
                cost = cost + float(w_du) * du.pow(2).sum()

            if w_jerk != 0.0 and u_norm.shape[0] >= 3:
                du = u_norm[1:] - u_norm[:-1]
                ddu = du[1:] - du[:-1]
                cost = cost + float(w_jerk) * ddu.pow(2).sum()

            return cost

        return extra

    def _make_step_response_extra_cost(
        self, cfg: MPCStepResponseExtraCostConfig
    ) -> ExtraCostFn:
        tracked_idx = self._resolve_state_idx(cfg.tracked_idx)
        rate_idx = None
        if cfg.rate_idx is not None:
            rate_idx = self._resolve_state_idx(cfg.rate_idx)

        dt = max(1e-6, float(cfg.dt))
        settle_steps = int(np.ceil(float(cfg.settle_time_target_s) / dt))

        def extra(
            x_seq: torch.Tensor,
            u_seq: torch.Tensor,
            x_ref: torch.Tensor | None,
        ) -> torch.Tensor:
            # Always include jerk (it helps even without a detected step).
            if u_seq.numel() == 0 or self._u_lim is None:
                return u_seq.new_tensor(0.0)

            u_lim = self._u_lim.reshape(1, -1)
            u_norm = torch.clamp(u_seq / u_lim, -1.0, 1.0)

            cost = u_seq.new_tensor(0.0)
            if float(cfg.w_jerk) != 0.0 and u_norm.shape[0] >= 3:
                du = u_norm[1:] - u_norm[:-1]
                ddu = du[1:] - du[:-1]
                cost = cost + float(cfg.w_jerk) * ddu.pow(2).sum()

            if x_ref is None:
                return cost

            theta_ref = x_ref[:, tracked_idx]
            baseline = theta_ref[0]
            changed = torch.abs(theta_ref - baseline) > float(cfg.ref_change_threshold)
            if not torch.any(changed):
                return cost

            step_idx = int(torch.argmax(changed.to(torch.int64)).item())
            target = theta_ref[-1]

            theta = x_seq[:, tracked_idx]
            err = theta - target

            t = torch.arange(theta.shape[0], device=x_seq.device)
            post_step = t >= step_idx

            amp_abs = torch.abs(target - baseline)
            band = torch.maximum(
                x_seq.new_tensor(float(cfg.settle_band)),
                amp_abs * float(cfg.settle_band_ratio),
            )
            band = torch.maximum(band, x_seq.new_tensor(float(cfg.settle_band_min)))

            theta_exceed = torch.relu(torch.abs(err) - band) * post_step.to(x_seq.dtype)
            deadline = step_idx + int(settle_steps)

            w_settle = x_seq.new_tensor(float(cfg.w_settle))
            w_time = x_seq.new_tensor(float(cfg.w_time))
            w_zero = x_seq.new_tensor(0.0)
            w_theta = torch.where(
                t >= deadline,
                w_settle,
                torch.where(post_step, w_time, w_zero),
            )
            cost = cost + (w_theta * theta_exceed.pow(2)).sum()

            if rate_idx is not None:
                rate = x_seq[:, rate_idx]
                rate_exceed = torch.relu(
                    torch.abs(rate) - float(cfg.rate_settle)
                ) * post_step.to(x_seq.dtype)
                w_rate = (t >= deadline).to(x_seq.dtype) * float(cfg.w_settle)
                cost = cost + (w_rate * rate_exceed.pow(2)).sum()

            w_sse = (t >= deadline).to(x_seq.dtype) * float(cfg.w_sse_steady)
            cost = cost + (w_sse * err.pow(2)).sum()

            # Oscillation penalty: sign changes of error outside band
            if err.numel() >= 2:
                err_post = err[step_idx:]
                if err_post.numel() >= 2:
                    prod = err_post[1:] * err_post[:-1]
                    sign_change = torch.relu(-prod) / (band.pow(2) + 1e-9)
                    out_w = torch.sigmoid(20.0 * (torch.abs(err_post) - band))
                    out_pair = out_w[1:] * out_w[:-1]
                    cost = cost + float(cfg.w_osc) * (sign_change * out_pair).sum()

            # Overshoot penalty (only if step is "big enough")
            if float(amp_abs.detach().item()) >= float(cfg.min_step_amp):
                seg_sign = torch.sign(target - baseline)
                seg_sign = torch.where(
                    seg_sign == 0, torch.ones_like(seg_sign), seg_sign
                )
                err_dir = (theta - target) * seg_sign
                overshoot_excess = torch.relu(
                    err_dir - float(cfg.overshoot_limit)
                ) * post_step.to(x_seq.dtype)
                cost = cost + float(cfg.w_overshoot) * overshoot_excess.pow(2).sum()

            # Extra smoothing after "deadline" (steady-state)
            deadline_u = int(max(0, int(deadline) - 1))
            if u_norm.shape[0] >= 2:
                du_u = u_norm[1:] - u_norm[:-1]
                tu = torch.arange(u_norm.shape[0], device=x_seq.device)
                w_du = (tu[1:] >= deadline_u).to(x_seq.dtype).reshape(-1, 1)
                cost = cost + float(cfg.w_du_steady) * (w_du * du_u.pow(2)).sum()
                if du_u.shape[0] >= 2:
                    ddu_u = du_u[1:] - du_u[:-1]
                    w_ddu = (tu[2:] >= deadline_u).to(x_seq.dtype).reshape(-1, 1)
                    cost = (
                        cost + float(cfg.w_jerk_steady) * (w_ddu * ddu_u.pow(2)).sum()
                    )

            return cost

        return extra

    def to_device(self, device: str | torch.device) -> "MPCAgent":
        """Move model, normalizers, and MPC to a new device (DSAC-style)."""

        new_device = torch.device(device)
        if new_device == self.device:
            return self

        self.device = new_device

        # Move model
        self.model = self.model.to(self.device, dtype=self.dtype)

        # Move optimizer state
        for state in self.model_opt.state.values():
            if isinstance(state, dict):
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

        # Move scalers
        self.x_scaler.to(device=self.device, dtype=self.dtype)
        self.u_scaler.to(device=self.device, dtype=self.dtype)
        self.y_scaler.to(device=self.device, dtype=self.dtype)

        # Rebuild MPC tensors on the new device and reset warm-start
        self._u_prev = None
        self._update_u_limit_tensor()
        self._rebuild_mpc()
        self.mpc.reset()
        return self

    # -----------------------------
    # BaseRLModel compatibility
    # -----------------------------
    def get_env(self):  # noqa: D401
        """Return current env."""
        return self.env

    def predict(
        self,
        state: np.ndarray | None = None,
        *,
        x_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        if state is None:
            raise ValueError("predict() requires an explicit `state` array.")
        return self.select_action(state, x_ref=x_ref)

    def get_param_env(self) -> dict[str, dict[str, Any]]:
        """Return env/policy metadata for HuggingFace-style checkpoints."""

        env_obj = getattr(self.env, "unwrapped", self.env)
        env_name = f"{env_obj.__class__.__module__}.{env_obj.__class__.__name__}"
        env_params: dict[str, Any] = {}
        if "tensoraerospace" in env_name:
            try:
                env_params = serialize_env(self.env)
            except Exception:
                env_params = {}

        def _serialize_dc(dc: Any) -> Any:
            if dc is None:
                return None
            return _to_serializable(dc)

        policy_params: dict[str, Any] = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "horizon": int(self._mpc_horizon),
            "weights": _serialize_dc(self._mpc_weights),
            "constraints": _serialize_dc(self._mpc_constraints),
            "tracking_type": self.tracking_type,
            "tracking_config": _serialize_dc(self.tracking_config),
            "step_response_config": _serialize_dc(self.step_response_config),
            "iters": int(self._mpc_iters),
            "mpc_lr": float(self._mpc_lr),
            "mpc_optimizer": self._mpc_optimizer,
            "warm_start": bool(self._mpc_warm_start),
            "mpc_track_best": bool(self._mpc_track_best),
            "mpc_best_check_every": int(self._mpc_best_check_every),
            "mpc_compile_dynamics": bool(self._mpc_compile_dynamics),
            "mpc_compile_mode": str(self._mpc_compile_mode),
            "model_predict_delta": bool(self.model_predict_delta),
            "hidden_layers": (
                list(getattr(self.model, "hidden_layers", []))
                if hasattr(self.model, "hidden_layers")
                else list(getattr(self, "_hidden_layers_cfg", []))
            ),
            "activation": getattr(self, "_activation_name", "relu"),
            "normalize": bool(self.normalize),
            "dynamics_lr": float(self.model_opt.defaults.get("lr", 0.0)),
            "weight_decay": float(self.model_opt.defaults.get("weight_decay", 0.0)),
            "grad_clip_norm": self.grad_clip_norm,
            "memory_capacity": int(getattr(self.memory, "capacity", 0)),
            "device": _safe_device_str(self.device),
            "dtype": str(self.dtype),
            "seed": int(self.seed),
            "model_class": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}",
        }

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {
                "name": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "params": policy_params,
            },
        }

    # -----------------------------
    # Adapters
    # -----------------------------
    def _state_from_obs(self, obs: Any) -> np.ndarray:
        if self._obs_to_state is not None:
            x = self._obs_to_state(self.env, obs)
        else:
            x = np.asarray(obs, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.state_dim:
            raise ValueError(
                f"obs_to_state produced state of size {x.size}, "
                f"expected {self.state_dim}"
            )
        return x

    def _action_env_from_internal(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        if u.size != self.action_dim:
            raise ValueError(f"Action must have size {self.action_dim}, got {u.size}")
        if self._action_to_env is not None:
            a = self._action_to_env(u)
        else:
            a = u
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(
                f"action_to_env returned size {a.size}, " f"expected {self.action_dim}"
            )
        if self._a_low_env is not None and self._a_high_env is not None:
            a = np.clip(a, self._a_low_env, self._a_high_env)
        return a.astype(np.float32, copy=False)

    def _action_internal_from_env(self, a_env: np.ndarray) -> np.ndarray:
        a_env = np.asarray(a_env, dtype=np.float32).reshape(-1)
        if a_env.size != self.action_dim:
            raise ValueError(
                f"Env action must have size {self.action_dim}, " f"got {a_env.size}"
            )
        if self._action_from_env is not None:
            u = self._action_from_env(a_env)
        else:
            u = a_env
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        if u.size != self.action_dim:
            raise ValueError(
                f"action_from_env returned size {u.size}, "
                f"expected {self.action_dim}"
            )
        return u.astype(np.float32, copy=False)

    # -----------------------------
    # MPC policy
    # -----------------------------
    def reset(self) -> None:
        """Reset MPC warm-start and previous action memory."""
        self._u_prev = None
        self.mpc.reset()

    def select_action(
        self, state: np.ndarray, *, x_ref: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute action (env units) using MPC over learned dynamics."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(f"state must have size {self.state_dim}, got {state.size}")

        u_prev = None if self._u_prev is None else self._u_prev
        res = self.mpc.solve(x0=state, x_ref=x_ref, u_prev=u_prev)
        u0 = res.u0.reshape(-1).astype(np.float32, copy=False)
        self._u_prev = u0.copy()
        return self._action_env_from_internal(u0)

    # -----------------------------
    # Data collection
    # -----------------------------
    def collect_data(
        self,
        *,
        num_episodes: int = 10,
        max_steps: int | None = None,
        exploration: Literal["random", "signals"] = "random",
        signal_kinds: Sequence[str] | None = None,
        dt: float | None = None,
        action_amplitude_frac: float = 0.8,
    ) -> None:
        """Collect (x, u, x_next) transitions into `self.memory`.

        Notes:
            - For `exploration="random"` uses env.action_space.sample().
            - For `exploration="signals"` uses `tensoraerospace.signals` to
              generate time-series actions (works for continuous Box actions).
        """

        num_episodes = int(num_episodes)
        if num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")

        if max_steps is None:
            # Common TensorAeroSpace envs expose a fixed horizon
            max_steps = int(
                getattr(
                    self.env.unwrapped,
                    "number_time_steps",
                    getattr(self.env, "number_time_steps", 0),
                )
            )
            if max_steps <= 0:
                max_steps = 10_000  # fallback: until done
        max_steps = int(max_steps)
        if max_steps < 2:
            raise ValueError("max_steps must be >= 2")

        if exploration == "signals":
            from tensoraerospace import signals as ta_signals  # lazy import

            if dt is None:
                dt = float(
                    getattr(self.env.unwrapped, "dt", getattr(self.env, "dt", 0.01))
                )
            dt = float(dt)
            tp = (np.arange(max_steps, dtype=np.float32) * dt).astype(np.float32)

            if signal_kinds is None:
                signal_kinds = (
                    "random_steps",
                    "unit_step",
                    "multi_step",
                    "ramp",
                    "sinusoid",
                    "multisine",
                    "chirp",
                    "square_wave",
                    "triangular_wave",
                    "sawtooth",
                    "doublet",
                    "pulse",
                    "gaussian_pulse",
                    "exponential",
                    "damped_sinusoid",
                )
            kinds = [str(k).lower().strip() for k in signal_kinds]
            if len(kinds) < 1:
                raise ValueError(
                    "signal_kinds must be non-empty for exploration='signals'"
                )

            # Use env bounds if available; otherwise default to [-1, 1]
            if self._a_low_env is None or self._a_high_env is None:
                low = -np.ones((self.action_dim,), dtype=np.float32)
                high = np.ones((self.action_dim,), dtype=np.float32)
            else:
                low = self._a_low_env
                high = self._a_high_env

            amp = float(np.max(np.abs(np.concatenate([low, high])))) * float(
                action_amplitude_frac
            )
            amp = max(1e-6, amp)

            def gen_1d(kind: str) -> np.ndarray:
                kind = str(kind).lower().strip()
                t_end = float(tp[-1]) if tp.size else 0.0

                if kind == "random_steps":
                    base = float(dt) * float(max(1, int(max_steps // 20)))
                    sd = (0.5 * base, 2.5 * base)
                    return ta_signals.full_random_signal(
                        t0=0.0, dt=dt, tn=t_end, sd=sd, sv=(-amp, amp)
                    ).astype(np.float32)

                if kind == "unit_step":
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    time_step = float(np.random.uniform(0.05, 0.35)) * max(dt, t_end)
                    return ta_signals.unit_step(
                        tp, degree=a, time_step=time_step, output_rad=False
                    ).astype(np.float32)

                if kind == "multi_step":
                    n_steps = int(np.random.randint(2, 7))
                    times = np.sort(
                        np.random.uniform(0.05 * t_end, 0.95 * t_end, size=(n_steps,))
                    ).tolist()
                    inc = (
                        np.random.uniform(-1.0, 1.0, size=(n_steps,))
                        .astype(np.float32)
                        .tolist()
                    )
                    u = ta_signals.multi_step(
                        tp, step_times=times, step_values=inc
                    ).astype(np.float32)
                    m = float(np.max(np.abs(u))) if u.size else 1.0
                    u = u / max(1e-6, m)
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    return (a * u).astype(np.float32)

                if kind == "ramp":
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    time_start = float(np.random.uniform(0.0, 0.3 * t_end))
                    dur = max(dt, t_end - time_start)
                    slope = float(a) / float(dur)
                    return ta_signals.ramp(
                        tp, slope=slope, time_start=time_start
                    ).astype(np.float32)

                if kind == "sinusoid":
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.2, 1.0))
                        * amp
                    )
                    f = float(np.random.uniform(0.01, 2.2))
                    return ta_signals.sinusoid_vertical_shift(
                        tp, frequency=f, amplitude=a, vertical_shift=0.0
                    ).astype(np.float32)

                if kind == "multisine":
                    n_comp = int(np.random.randint(1, 8))
                    freqs = [float(np.random.uniform(0.01, 2.2)) for _ in range(n_comp)]
                    amps = [float(np.random.uniform(0.2, 1.0)) for _ in range(n_comp)]
                    phases = [
                        float(np.random.uniform(0.0, 2.0 * np.pi))
                        for _ in range(n_comp)
                    ]
                    u = ta_signals.multisine(
                        tp, frequencies=freqs, amplitudes=amps, phases=phases
                    ).astype(np.float32)
                    m = float(np.max(np.abs(u))) if u.size else 1.0
                    u = u / max(1e-6, m)
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    return (a * u).astype(np.float32)

                if kind == "chirp":
                    f0 = float(np.random.uniform(0.01, 1.0))
                    f1 = float(np.random.uniform(f0, 1.5))
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    method = str(np.random.choice(["linear", "exponential"]))
                    u = ta_signals.chirp(
                        tp, f0=f0, f1=f1, amplitude=abs(a), method=method
                    ).astype(np.float32)
                    return (float(np.sign(a)) * u).astype(np.float32)

                if kind == "square_wave":
                    f = float(np.random.uniform(0.01, 1.8))
                    duty = float(np.random.uniform(0.2, 0.8))
                    u01 = ta_signals.square_wave(
                        tp, frequency=f, amplitude=amp, duty_cycle=duty
                    ).astype(np.float32)
                    # 0/amp -> -amp/+amp
                    return (2.0 * u01 - amp).astype(np.float32)

                if kind == "triangular_wave":
                    f = float(np.random.uniform(0.01, 1.8))
                    return ta_signals.triangular_wave(
                        tp, frequency=f, amplitude=amp
                    ).astype(np.float32)

                if kind == "sawtooth":
                    f = float(np.random.uniform(0.01, 1.8))
                    return ta_signals.sawtooth(tp, frequency=f, amplitude=amp).astype(
                        np.float32
                    )

                if kind == "doublet":
                    width = float(np.random.uniform(1.0, 25.0)) * dt
                    time_start = float(np.random.uniform(0.05 * t_end, 0.35 * t_end))
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.5, 1.0))
                        * amp
                    )
                    return ta_signals.doublet(
                        tp, amplitude=a, time_start=time_start, width=width
                    ).astype(np.float32)

                if kind == "pulse":
                    width = float(np.random.uniform(1.0, 20.0)) * dt
                    time_start = float(np.random.uniform(0.05 * t_end, 0.8 * t_end))
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    return ta_signals.pulse(
                        tp, amplitude=a, time_start=time_start, width=width
                    ).astype(np.float32)

                if kind == "gaussian_pulse":
                    center = float(np.random.uniform(0.05 * t_end, 0.95 * t_end))
                    width = float(np.random.uniform(0.05, 0.6))
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    return ta_signals.gaussian_pulse(
                        tp, amplitude=a, center=center, width=width
                    ).astype(np.float32)

                if kind == "exponential":
                    tau = float(np.random.uniform(0.15, 3.0))
                    time_start = float(np.random.uniform(0.0, 0.35 * t_end))
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    return ta_signals.exponential(
                        tp,
                        amplitude=a,
                        time_constant=tau,
                        time_start=time_start,
                    ).astype(np.float32)

                if kind == "damped_sinusoid":
                    f = float(np.random.uniform(0.01, 2.2))
                    damping = float(np.random.uniform(0.05, 0.6))
                    time_start = float(np.random.uniform(0.0, 0.25)) * t_end
                    a = (
                        float(np.random.choice([-1.0, 1.0]))
                        * float(np.random.uniform(0.3, 1.0))
                        * amp
                    )
                    u = ta_signals.damped_sinusoid(
                        tp,
                        frequency=f,
                        amplitude=abs(a),
                        damping=damping,
                        time_start=time_start,
                    ).astype(np.float32)
                    return (float(np.sign(a)) * u).astype(np.float32)

                raise ValueError(f"Unknown signal kind: {kind!r}")

        for _ in tqdm(range(num_episodes), desc="Collect episodes"):
            obs, _info = self.env.reset()
            state = self._state_from_obs(obs)
            done = False

            if exploration == "signals":
                # one signal per action dimension (independent)
                u_seq = np.zeros((max_steps, self.action_dim), dtype=np.float32)
                for j in range(self.action_dim):
                    kind = str(np.random.choice(kinds))
                    u_seq[:, j] = gen_1d(kind)[:max_steps]
                # clip to env bounds if known
                if self._a_low_env is not None and self._a_high_env is not None:
                    u_seq = np.clip(
                        u_seq,
                        self._a_low_env.reshape(1, -1),
                        self._a_high_env.reshape(1, -1),
                    )
            else:
                u_seq = None

            t = 0
            while not done and t < max_steps - 1:
                if exploration == "random":
                    a_env = np.asarray(
                        self.env.action_space.sample(), dtype=np.float32
                    ).reshape(-1)
                elif exploration == "signals":
                    assert u_seq is not None
                    a_env = u_seq[t].reshape(-1)
                else:
                    raise ValueError(f"Unknown exploration: {exploration!r}")

                # store internal action for training/MPC
                # (may differ from env action)
                u_internal = self._action_internal_from_env(a_env)

                next_obs, _reward, terminated, truncated, _info = self.env.step(
                    self._action_env_from_internal(u_internal)
                )
                next_state = self._state_from_obs(next_obs)

                done_env = bool(terminated or truncated)
                done_bootstrap = float(bool(terminated))
                self.memory.push(state, u_internal, 0.0, next_state, done_bootstrap)

                state = next_state
                done = done_env
                t += 1

    # -----------------------------
    # Dynamics training
    # -----------------------------
    def fit_normalizers(self, *, num_samples: int = 50_000) -> None:
        """Fit x/u/y normalizers from a random subset of the replay buffer."""
        if len(self.memory) < 2:
            raise ValueError("Replay buffer is empty; collect_data() first.")
        n = int(min(len(self.memory), int(num_samples)))
        if n < 2:
            raise ValueError("Not enough samples to fit normalizers.")
        s, a, _r, ns, _d = self.memory.sample(batch_size=n)

        s_t = torch.as_tensor(s, device=self.device, dtype=self.dtype).view(n, -1)
        a_t = torch.as_tensor(a, device=self.device, dtype=self.dtype).view(n, -1)
        ns_t = torch.as_tensor(ns, device=self.device, dtype=self.dtype).view(n, -1)

        if s_t.shape[1] != self.state_dim or a_t.shape[1] != self.action_dim:
            raise ValueError(
                f"Bad sample shapes from memory: s={tuple(s_t.shape)}, "
                f"a={tuple(a_t.shape)}"
            )

        y = ns_t - s_t if self.model_predict_delta else ns_t

        self.x_scaler = MPCStandardScaler.fit(s_t).to(
            device=self.device, dtype=self.dtype
        )
        self.u_scaler = MPCStandardScaler.fit(a_t).to(
            device=self.device, dtype=self.dtype
        )
        self.y_scaler = MPCStandardScaler.fit(y).to(
            device=self.device, dtype=self.dtype
        )

    def train_dynamics(
        self,
        *,
        epochs: int = 5,
        batch_size: int = 1024,
        steps_per_epoch: int | None = None,
        loss: Literal["mse", "huber"] = "mse",
    ) -> dict[str, float]:
        """Train the dynamics model on transitions stored in the replay buffer.

        The model is trained on samples from `self.memory`.
        """

        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: have {len(self.memory)}, "
                f"need {batch_size}"
            )

        # Fit scalers once (unless normalize disabled)
        if self.normalize:
            self.fit_normalizers()

        if steps_per_epoch is None:
            steps_per_epoch = max(1, int(len(self.memory) // batch_size))
        steps_per_epoch = int(steps_per_epoch)

        if loss == "mse":
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss()
        elif loss == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss!r}")

        self.model.train()
        total_steps = int(epochs) * int(steps_per_epoch)
        pbar = tqdm(range(total_steps), desc="Train dynamics", unit="step")

        running = 0.0
        for _ in pbar:
            s, a, _r, ns, _d = self.memory.sample(batch_size=batch_size)
            s_t = torch.as_tensor(s, device=self.device, dtype=self.dtype).view(
                batch_size, -1
            )
            a_t = torch.as_tensor(a, device=self.device, dtype=self.dtype).view(
                batch_size, -1
            )
            ns_t = torch.as_tensor(ns, device=self.device, dtype=self.dtype).view(
                batch_size, -1
            )

            if self.model_predict_delta:
                y_t = ns_t - s_t
            else:
                y_t = ns_t

            if self.normalize:
                xn = self.x_scaler.transform(s_t)
                un = self.u_scaler.transform(a_t)
                yn = self.y_scaler.transform(y_t)
                xu = torch.cat([xn, un], dim=-1)
                y_hat_n = cast(torch.Tensor, self.model(xu))
                train_loss = loss_fn(y_hat_n, yn)
            else:
                xu = torch.cat([s_t, a_t], dim=-1)
                y_hat = cast(torch.Tensor, self.model(xu))
                train_loss = loss_fn(y_hat, y_t)

            self.model_opt.zero_grad(set_to_none=True)
            train_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
            self.model_opt.step()

            running = 0.98 * running + 0.02 * float(train_loss.detach().cpu().item())
            pbar.set_postfix(loss=float(running))

        self.model.eval()
        return {"loss": float(running)}

    # -----------------------------
    # Save / load (HuggingFace-style, like PPO/DSAC)
    # -----------------------------
    @staticmethod
    def _filter_kwargs_for_init(
        env_cls: type, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Drop unexpected kwargs so env construction is robust to config drift."""
        try:
            sig = inspect.signature(env_cls.__init__)
        except (TypeError, ValueError):
            return kwargs

        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return kwargs

        allowed: set[str] = set()
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                allowed.add(name)
        return {k: v for k, v in kwargs.items() if k in allowed}

    def save(self, path: str | Path | None = None, save_gradients: bool = True) -> Path:
        """Save MPC agent in HuggingFace-style layout (config + weights)."""

        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        model_path = run_dir / "dynamics_model.pth"
        optim_path = run_dir / "dynamics_optim.pth"
        norms_path = run_dir / "normalizers.npz"

        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        torch.save(_state_dict_cpu(self.model), model_path)
        if save_gradients:
            torch.save(_optimizer_state_dict_cpu(self.model_opt), optim_path)

        np.savez(
            norms_path,
            x_mean=self.x_scaler.mean.detach().cpu().numpy(),
            x_std=self.x_scaler.std.detach().cpu().numpy(),
            u_mean=self.u_scaler.mean.detach().cpu().numpy(),
            u_std=self.u_scaler.std.detach().cpu().numpy(),
            y_mean=self.y_scaler.mean.detach().cpu().numpy(),
            y_std=self.y_scaler.std.detach().cpu().numpy(),
        )
        return run_dir

    @classmethod
    def __load(cls, path: str | Path, load_gradients: bool = False) -> "MPCAgent":
        path = Path(path)
        config_path = path / "config.json"
        model_path = path / "dynamics_model.pth"
        optim_path = path / "dynamics_optim.pth"
        norms_path = path / "normalizers.npz"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        agent_name = f"{cls.__module__}.{cls.__name__}"
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch

        # --- rebuild env
        env_name = config["env"]["name"]
        raw_env_params = dict(config["env"].get("params", {}) or {})
        if "tensoraerospace" in env_name:
            env_cls = get_class_from_string(env_name)
            env_params = deserialize_env_params(raw_env_params)
            env_params = cls._filter_kwargs_for_init(env_cls, env_params)
            if "device" in env_params:
                env_params["device"] = _safe_device_str(env_params["device"])
            env = env_cls(**env_params)
        else:
            env = get_class_from_string(env_name)()

        # --- rebuild policy params
        policy_params = dict(config["policy"]["params"])
        policy_params["device"] = _safe_device_str(policy_params.get("device"))
        policy_params["dtype"] = _dtype_from_string(policy_params.get("dtype"))
        if policy_params.get("mpc_compile_dynamics") and not hasattr(torch, "compile"):
            policy_params["mpc_compile_dynamics"] = False
        for key in [
            "state_dim",
            "action_dim",
            "horizon",
            "iters",
            "mpc_best_check_every",
            "memory_capacity",
            "seed",
        ]:
            if key in policy_params:
                policy_params[key] = int(policy_params[key])

        weights_cfg = policy_params.pop("weights", None)
        constraints_cfg = policy_params.pop("constraints", None)
        tracking_cfg = policy_params.pop("tracking_config", None)
        step_cfg = policy_params.pop("step_response_config", None)
        model_class_path = policy_params.pop("model_class", None)

        def _maybe_arr(x: Any) -> np.ndarray | None:
            if x is None:
                return None
            return np.asarray(x, dtype=np.float32)

        if policy_params.get("grad_clip_norm") is not None:
            policy_params["grad_clip_norm"] = float(policy_params["grad_clip_norm"])
        if "mpc_lr" in policy_params:
            policy_params["mpc_lr"] = float(policy_params["mpc_lr"])
        if "dynamics_lr" in policy_params:
            policy_params["dynamics_lr"] = float(policy_params["dynamics_lr"])

        if weights_cfg is not None:
            policy_params["weights"] = MPCWeights(
                Q_diag=_maybe_arr(weights_cfg.get("Q_diag")),
                R_diag=_maybe_arr(weights_cfg.get("R_diag")),
                S_diag=_maybe_arr(weights_cfg.get("S_diag")),
                terminal_weight=float(weights_cfg.get("terminal_weight", 1.0)),
            )
        if constraints_cfg is not None:
            policy_params["constraints"] = MPCConstraints(
                u_min=_maybe_arr(constraints_cfg.get("u_min")),
                u_max=_maybe_arr(constraints_cfg.get("u_max")),
                du_min=_maybe_arr(constraints_cfg.get("du_min")),
                du_max=_maybe_arr(constraints_cfg.get("du_max")),
            )
        if tracking_cfg is not None:
            policy_params["tracking_config"] = MPCTrackingExtraCostConfig(
                **{
                    k: v
                    for k, v in tracking_cfg.items()
                    if k in MPCTrackingExtraCostConfig.__annotations__
                }
            )
        if step_cfg is not None:
            policy_params["step_response_config"] = MPCStepResponseExtraCostConfig(
                **{
                    k: v
                    for k, v in step_cfg.items()
                    if k in MPCStepResponseExtraCostConfig.__annotations__
                }
            )

        if isinstance(policy_params.get("hidden_layers"), (list, tuple)):
            policy_params["hidden_layers"] = tuple(
                int(h) for h in policy_params["hidden_layers"]
            )
        else:
            policy_params["hidden_layers"] = (256, 256)

        if model_class_path and "OneStepMLP" not in model_class_path:
            print(
                f"Warning: custom model {model_class_path} not automatically rebuilt; "
                "using default OneStepMLP."
            )

        new_agent = cls(env=env, **policy_params)

        # --- load weights/optim
        state = torch.load(
            model_path, map_location=new_agent.device, weights_only=False
        )
        new_agent.model.load_state_dict(state)

        if load_gradients and optim_path.exists():
            opt_state = torch.load(
                optim_path, map_location=new_agent.device, weights_only=False
            )
            new_agent.model_opt.load_state_dict(opt_state)

        if norms_path.exists():
            data = np.load(norms_path)
            new_agent.x_scaler.mean = torch.as_tensor(
                data["x_mean"], device=new_agent.device, dtype=new_agent.dtype
            )
            new_agent.x_scaler.std = torch.as_tensor(
                data["x_std"], device=new_agent.device, dtype=new_agent.dtype
            )
            new_agent.u_scaler.mean = torch.as_tensor(
                data["u_mean"], device=new_agent.device, dtype=new_agent.dtype
            )
            new_agent.u_scaler.std = torch.as_tensor(
                data["u_std"], device=new_agent.device, dtype=new_agent.dtype
            )
            new_agent.y_scaler.mean = torch.as_tensor(
                data["y_mean"], device=new_agent.device, dtype=new_agent.dtype
            )
            new_agent.y_scaler.std = torch.as_tensor(
                data["y_std"], device=new_agent.device, dtype=new_agent.dtype
            )

        new_agent.model.eval()
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "MPCAgent":
        """Load checkpoint from local dir or HuggingFace Hub."""

        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(
                    f"Local directory not found: '{repo_name}'. Please check the path."
                )
            return cls.__load(p, load_gradients=load_gradients)

        folder_path = BaseRLModel.from_pretrained(
            repo_name, access_token=access_token, version=version
        )
        return cls.__load(folder_path, load_gradients=load_gradients)

    def push_to_hub(
        self,
        repo_name: str,
        access_token: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        include_gradients: bool = False,
    ) -> str:
        """Save checkpoint and upload it to HuggingFace Hub."""
        base_path = Path.cwd() if save_path is None else Path(str(save_path))
        base_path.mkdir(parents=True, exist_ok=True)

        run_dir = self.save(path=base_path, save_gradients=include_gradients)
        BaseRLModel().publish_to_hub(
            repo_name=repo_name,
            folder_path=str(run_dir),
            access_token=access_token,
        )
        return str(run_dir)
