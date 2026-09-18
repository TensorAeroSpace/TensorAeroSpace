"""Learn LAPAN one-step dynamics with standalone NARX, then check free rollouts.

This is system identification, not a trained control policy. The independent
coefficient table is Septiyana et al. (2020), p.86. A scalar-reference mode also
works on the old implementation whose batched forward raises an exception.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scalar-reference", action="store_true")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from scipy.signal import cont2discrete

    from tensoraerospace.agent.narx.model import NARX

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(20260918)
    a = np.array(
        [
            [-0.00271615, 0.248462, 0, -9.81],
            [-0.257616, -11.3097, 68.9497, 0],
            [0.0576336, -7.23232, -11.3237, 0],
            [0, 0, 1, 0],
        ]
    )
    b = np.array([[1.959083], [-73.99448], [-188.4752], [0]])
    dt = 0.02
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(4), np.zeros((4, 1))), dt)
    scale = np.array([0.5, 0.1, np.deg2rad(2), np.deg2rad(2)])
    control_scale = np.deg2rad(0.5)
    # Independent uniform states and controls keep training coverage explicit.
    x = rng.uniform(-1, 1, (12000, 4))
    u = rng.uniform(-1, 1, (12000, 1))
    y = ((x * scale) @ ad.T + (u * control_scale) @ bd.T) / scale
    states, controls, targets = [
        torch.tensor(v, dtype=torch.float32) for v in (x, u, y)
    ]
    net = NARX(1, 32, 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        repo=str(args.repo.resolve()),
        seed=args.seed,
        scalar_reference=args.scalar_reference,
        config=dict(
            dt=dt,
            updates=args.updates,
            batch_size=128,
            hidden_size=32,
            lr=0.01,
            train_count=10000,
            validation_count=2000,
            state_scale=scale.tolist(),
            elevator_scale_rad=float(control_scale),
        ),
    )
    try:
        prediction = net(controls[:2], states[:2])
        result["batch_forward"] = dict(supported=True, shape=list(prediction.shape))
    except RuntimeError as exc:
        result["batch_forward"] = dict(supported=False, error=str(exc))
        if not args.scalar_reference and args.checkpoint is None:
            raise

    def batch_forward(u, x):
        return (
            torch.stack([net(ui, xi) for ui, xi in zip(u, x)])
            if args.scalar_reference
            else net(u, x)
        )

    losses, curves = [], []
    if args.checkpoint:
        net.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
    else:
        generator = torch.Generator().manual_seed(args.seed + 100)
        for update in range(args.updates):
            indices = torch.randint(10000, (128,), generator=generator)
            loss = net.train(
                batch_forward(controls[indices], states[indices]), targets[indices]
            )
            if not np.isfinite(loss) or not all(
                torch.isfinite(p).all() for p in net.parameters()
            ):
                raise FloatingPointError("Nonfinite NARX learning")
            losses.append(loss)
            if (update + 1) % 200 == 0:
                with torch.no_grad():
                    mse = float(
                        (
                            batch_forward(controls[10000:], states[10000:])
                            - targets[10000:]
                        )
                        .square()
                        .mean()
                    )
                curves.append(dict(updates=update + 1, validation_mse=mse))
        torch.save(net.state_dict(), args.output.with_suffix(".pt"))
    with torch.no_grad():
        errors = (
            batch_forward(controls[10000:], states[10000:]) - targets[10000:]
        ).numpy() * scale
    result["one_step_rmse"] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
    result["one_step_pitch_rmse_deg"] = float(np.rad2deg(result["one_step_rmse"][3]))
    result["learning_curve"] = curves
    result["max_training_loss"] = max(losses) if losses else None
    result["parameters_finite"] = all(
        bool(torch.isfinite(p).all()) for p in net.parameters()
    )
    result["free_rollout"] = []
    # Frozen recursive prediction: no teacher forcing and no retraining.
    for initial_pitch in [-1.0, 0.0, 1.0]:
        state = np.array([0.0, 0.0, 0.0, np.deg2rad(initial_pitch)])
        predicted = state.copy()
        errors, trace = [], []
        for i in range(500):
            command = np.deg2rad(0.2) * np.sin(2 * np.pi * 0.4 * i * dt)
            state = ad @ state + bd[:, 0] * command
            with torch.no_grad():
                predicted = (
                    net(
                        torch.tensor([command / control_scale], dtype=torch.float32),
                        torch.tensor(predicted / scale, dtype=torch.float32),
                    ).numpy()
                    * scale
                )
            if not np.all(np.isfinite(predicted)):
                raise FloatingPointError("Nonfinite free prediction")
            errors.append(predicted - state)
            trace.append(
                [
                    (i + 1) * dt,
                    float(np.rad2deg(state[3])),
                    float(np.rad2deg(predicted[3])),
                ]
            )
        errors = np.array(errors)
        result["free_rollout"].append(
            dict(
                initial_pitch_deg=initial_pitch,
                rmse_state_2s=np.sqrt(np.mean(errors[:100] ** 2, axis=0)).tolist(),
                rmse_state_10s=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                max_pitch_error_deg=float(np.rad2deg(np.max(abs(errors[:, 3])))),
                trace=trace[::5],
            )
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("free_rollout",)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
