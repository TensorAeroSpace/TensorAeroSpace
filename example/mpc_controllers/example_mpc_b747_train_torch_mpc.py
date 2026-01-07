"""Train a one-step B747 dynamics model and control it with differentiable MPC.

This example is intentionally lightweight
(no extra solver deps like cvxpy/osqp).

Run from repo root:
    poetry run python \\
        example/mpc_controllers/example_mpc_b747_train_torch_mpc.py

What it does:
1) Create LinearLongitudinalB747-v0 with a pitch step reference (radians).
2) Collect (x_t, u_t) -> x_{t+1} transitions from random actions.
3) Train a small torch MLP dynamics model in the model's native units (rad).
4) Run MPC using the learned model and plot theta tracking + elevator actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tensoraerospace.agent.mpc.torch_mpc import (
    TorchMPC,
    TorchMPCConstraints,
    TorchMPCWeights,
)
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period


def _slice_and_pad_1d(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    start = int(max(0, start))
    end = start + int(length)
    chunk = arr[start:end]
    if chunk.size >= length:
        return chunk.astype(np.float32, copy=False)
    pad_val = float(
        chunk[-1] if chunk.size > 0 else (arr[-1] if arr.size else 0.0)
    )
    pad = np.full((length - chunk.size,), pad_val, dtype=np.float32)
    return np.concatenate([chunk.astype(np.float32, copy=False), pad], axis=0)


@dataclass(frozen=True)
class Args:
    dt: float = 0.1
    tn: int = 20
    step_deg: float = 5.0
    step_time_s: float = 5.0

    # data collection
    collect_episodes: int = 20
    collect_max_steps: int = 250
    action_range_deg: float = 10.0

    # training
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    hidden: int = 128

    # MPC
    horizon: int = 25
    mpc_iters: int = 80
    mpc_lr: float = 0.08
    w_u: float = 0.05
    w_w: float = 0.05
    w_q: float = 1.0
    w_theta: float = 120.0
    w_action: float = 0.5
    w_delta_action: float = 2.0
    terminal_weight: float = 3.0


class MLPDynamics(nn.Module):
    def __init__(
        self, state_dim: int = 4, action_dim: int = 1, hidden: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2:
            u = u.reshape(u.shape[0], -1)
        xu = torch.cat([x, u], dim=-1)
        return cast(torch.Tensor, self.net(xu))


def make_env(dt: float, tn: int, step_deg: float, step_time_s: float):
    tp = generate_time_period(tn=tn, dt=dt)
    tps = np.array(convert_tp_to_sec_tp(tp, dt=dt), dtype=np.float32)
    number_time_steps = len(tp)
    reference_signal = unit_step(
        tp=tps,
        degree=int(round(step_deg)),
        time_step=int(round(step_time_s)),
        output_rad=True,
    ).reshape(1, -1)
    env = gym.make(
        "LinearLongitudinalB747-v0",
        number_time_steps=number_time_steps,
        initial_state=np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        reference_signal=reference_signal,
        dt=float(dt),
    )
    return env


def collect_transitions(
    env, *, episodes: int, max_steps: int, action_range_deg: float
):
    xs: list[np.ndarray] = []
    us: list[float] = []
    xns: list[np.ndarray] = []

    for _ in range(int(episodes)):
        env.reset()
        for _step in range(int(max_steps)):
            x_t = np.asarray(
                env.unwrapped.model.xt, dtype=np.float32
            ).reshape(-1)
            u_deg = float(
                np.random.uniform(-action_range_deg, action_range_deg)
            )
            action = np.array([u_deg], dtype=np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            x_tp1 = np.asarray(
                env.unwrapped.model.xt, dtype=np.float32
            ).reshape(-1)

            xs.append(x_t)
            us.append(float(np.deg2rad(u_deg)))
            xns.append(x_tp1)

            if terminated or truncated:
                break

    X = np.asarray(xs, dtype=np.float32)
    U = np.asarray(us, dtype=np.float32).reshape(-1, 1)
    XN = np.asarray(xns, dtype=np.float32)
    return X, U, XN


def train_dynamics(
    model: nn.Module,
    X: np.ndarray,
    U: np.ndarray,
    XN: np.ndarray,
    *,
    args: Args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    # train/val split
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_t = torch.tensor(X[train_idx], dtype=torch.float32)
    U_t = torch.tensor(U[train_idx], dtype=torch.float32)
    XN_t = torch.tensor(XN[train_idx], dtype=torch.float32)
    X_v = torch.tensor(X[val_idx], dtype=torch.float32)
    U_v = torch.tensor(U[val_idx], dtype=torch.float32)
    XN_v = torch.tensor(XN[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_t, U_t, XN_t),
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    for epoch in range(int(args.epochs)):
        model.train()
        total = 0.0
        for xb, ub, ynb in train_loader:
            xb = xb.to(device)
            ub = ub.to(device)
            ynb = ynb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, ub)
            loss = loss_fn(pred, ynb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())

        model.eval()
        with torch.no_grad():
            pred_v = model(X_v.to(device), U_v.to(device))
            val_loss = float(
                loss_fn(pred_v, XN_v.to(device)).detach().cpu().item()
            )
        if epoch % 5 == 0 or epoch == int(args.epochs) - 1:
            print(
                "epoch="
                f"{epoch:03d} "
                f"train_loss={total/len(train_loader):.6f} "
                f"val_loss={val_loss:.6f}"
            )

    return model.to("cpu")


def run_mpc(env, model: nn.Module, *, args: Args):
    model.eval()

    # Pull constraints from the underlying model (radians)
    b747_model = env.unwrapped.model
    u_lim = float(np.asarray(b747_model.input_magnitude_limits).reshape(-1)[0])
    dt = float(env.unwrapped.dt)
    rate_lim = float(np.asarray(b747_model.input_rate_limits).reshape(-1)[0])
    du_max = rate_lim * dt

    weights = TorchMPCWeights(
        Q_diag=np.array(
            [args.w_u, args.w_w, args.w_q, args.w_theta], dtype=np.float32
        ),
        R_diag=np.array([args.w_action], dtype=np.float32),
        S_diag=np.array([args.w_delta_action], dtype=np.float32),
        terminal_weight=float(args.terminal_weight),
    )
    constraints = TorchMPCConstraints(
        u_min=np.array([-u_lim], dtype=np.float32),
        u_max=np.array([u_lim], dtype=np.float32),
        du_min=np.array([-du_max], dtype=np.float32),
        du_max=np.array([du_max], dtype=np.float32),
    )

    def dyn(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, model(x, u))

    mpc = TorchMPC(
        dynamics=dyn,
        state_dim=4,
        action_dim=1,
        horizon=int(args.horizon),
        weights=weights,
        constraints=constraints,
        iters=int(args.mpc_iters),
        lr=float(args.mpc_lr),
        optimizer="adam",
        device="cpu",
        warm_start=True,
    )

    # Rollout with MPC
    obs, _ = env.reset()
    _ = obs  # ignore observation shape quirks; we use model.xt (full state)
    u_prev = None

    hist_theta_deg: list[float] = []
    hist_ref_deg: list[float] = []
    hist_u_deg: list[float] = []

    max_steps = int(env.unwrapped.number_time_steps - 2)
    ref = np.asarray(env.unwrapped.reference_signal, dtype=np.float32).reshape(
        1, -1
    )

    for step in range(max_steps):
        x0 = np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)
        theta_ref = _slice_and_pad_1d(
            ref[0], start=step, length=int(args.horizon) + 1
        )
        x_ref = np.zeros((int(args.horizon) + 1, 4), dtype=np.float32)
        x_ref[:, 3] = theta_ref

        res = mpc.solve(x0=x0, x_ref=x_ref, u_prev=u_prev)
        u0_rad = float(res.u0[0])
        u_prev = np.array([u0_rad], dtype=np.float32)
        u0_deg = float(np.rad2deg(u0_rad))
        action = np.array([u0_deg], dtype=np.float32)

        _obs, _reward, terminated, truncated, _info = env.step(action)

        theta_deg = float(np.rad2deg(env.unwrapped.model.xt.reshape(-1)[3]))
        ref_deg = float(np.rad2deg(ref[0, min(step, ref.shape[1] - 1)]))
        hist_theta_deg.append(theta_deg)
        hist_ref_deg.append(ref_deg)
        hist_u_deg.append(float(action[0]))

        if terminated or truncated:
            break

    return hist_theta_deg, hist_ref_deg, hist_u_deg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=Args.dt)
    p.add_argument("--tn", type=int, default=Args.tn)
    p.add_argument(
        "--collect-episodes", type=int, default=Args.collect_episodes
    )
    p.add_argument(
        "--collect-max-steps", type=int, default=Args.collect_max_steps
    )
    p.add_argument("--epochs", type=int, default=Args.epochs)
    p.add_argument("--horizon", type=int, default=Args.horizon)
    ns = p.parse_args()

    args = Args(
        dt=ns.dt,
        tn=ns.tn,
        collect_episodes=ns.collect_episodes,
        collect_max_steps=ns.collect_max_steps,
        epochs=ns.epochs,
        horizon=ns.horizon,
    )

    np.random.seed(0)
    torch.manual_seed(0)

    env = make_env(
        dt=args.dt,
        tn=args.tn,
        step_deg=args.step_deg,
        step_time_s=args.step_time_s,
    )
    print("Collecting data...")
    X, U, XN = collect_transitions(
        env,
        episodes=args.collect_episodes,
        max_steps=args.collect_max_steps,
        action_range_deg=args.action_range_deg,
    )
    print(f"Dataset: X={X.shape} U={U.shape} XN={XN.shape}")

    print("Training dynamics model...")
    model = MLPDynamics(state_dim=4, action_dim=1, hidden=args.hidden)
    model = train_dynamics(model, X, U, XN, args=args)

    print("Running MPC with learned dynamics...")
    theta_deg, ref_deg, u_deg = run_mpc(env, model, args=args)

    t = np.arange(len(theta_deg)) * float(args.dt)
    _, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, ref_deg, label="theta_ref (deg)")
    axes[0].plot(t, theta_deg, label="theta (deg)")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, u_deg, label="elevator (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("u (deg)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
