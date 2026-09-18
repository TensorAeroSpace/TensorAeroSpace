"""Train MPCAgent dynamics and compare MPC revisions on the same independent plant.

A learned linear layer is sufficient for this published linear model. This checks
library model training and MPC together, not a general nonlinear identification task.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from validate_ultrastick_physics import published_matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.linalg import expm

    from tensoraerospace.agent.mpc.mpc import MPCAgent, MPCConstraints, MPCWeights

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dt, steps, horizon = 0.02, 100, 10
    scales = np.array([1.0, 1.0, np.deg2rad(5), np.deg2rad(20), 1.0])
    elevator_limit = np.deg2rad(5.0)
    a, b = published_matrices()
    block = np.zeros((6, 6))
    block[:5, :5] = a
    block[:5, 5] = b
    zoh = expm(block * dt)
    ad = zoh[:5, :5] * scales[None, :] / scales[:, None]
    bd = zoh[:5, 5] * elevator_limit / scales
    q = np.array([0.001, 0.001, 1.0, 0.05, 0.001])
    rate = np.deg2rad(300) * dt / elevator_limit

    class Spaces:
        observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    agent = MPCAgent(
        Spaces(),
        model=torch.nn.Linear(6, 5),
        normalize=False,
        dynamics_lr=0.01,
        grad_clip_norm=None,
        seed=args.seed,
        horizon=horizon,
        iters=8,
        mpc_lr=0.05,
        dtype=torch.float64,
        weights=MPCWeights(Q_diag=q, R_diag=[0.02], terminal_weight=1),
        constraints=MPCConstraints(
            u_min=[-1], u_max=[1], du_min=[-rate], du_max=[rate]
        ),
    )
    samples = rng.uniform(-1, 1, (4096, 6))
    next_states = samples[:, :5] @ ad.T + samples[:, 5:] * bd
    for xu, ns in zip(samples, next_states):
        agent.memory.push(xu[:5].copy(), xu[5:].copy(), 0.0, ns.copy(), False)
    test = np.random.default_rng(20260917).uniform(-1, 1, (1024, 6))
    truth = test[:, :5] @ ad.T + test[:, 5:] * bd

    def prediction_error():
        with torch.no_grad():
            predicted = agent.mpc.dynamics(
                torch.tensor(test[:, :5]), torch.tensor(test[:, 5:])
            ).numpy()
        return float(np.max(abs(predicted - truth)))

    curves = []
    if args.checkpoint:
        agent.model.load_state_dict(
            torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        )
    else:
        for epoch_steps in [200, 200, 400]:
            metrics = agent.train_dynamics(
                epochs=1, steps_per_epoch=epoch_steps, batch_size=128
            )
            assert np.isfinite(metrics["loss"])
            assert all(torch.isfinite(p).all() for p in agent.model.parameters())
            curves.append(
                dict(
                    updates=epoch_steps + sum(c["updates_added"] for c in curves),
                    updates_added=epoch_steps,
                    loss=metrics["loss"],
                    max_holdout_prediction_error=prediction_error(),
                )
            )
        torch.save(agent.model.state_dict(), args.output.with_suffix(".pt"))
    agent.model.eval()
    cases = [[-3, -2, 0], [-3, 2, 0], [3, -2, 0], [3, 2, 0], [0, 0, -5], [0, 0, 5]]

    def evaluate(kind):
        errors, traces, failures, gaps = [], [], 0, []
        peak_theta, peak_q, max_slew = 0.0, 0.0, 0.0
        for target_deg, theta0, q0 in cases:
            x = np.array([0, 0, np.deg2rad(theta0), np.deg2rad(q0), 0]) / scales
            previous = 0.0
            agent.mpc.reset()
            err, trace = [], []
            for k in range(steps):
                times = (k + np.arange(1, horizon + 1)) * dt
                ref = np.zeros((horizon, 5))
                ref[:, 2] = np.where(times >= 0.4, target_deg / 5, 0.0)
                if kind == "mpc":
                    result = agent.mpc.solve(x0=x, x_ref=ref, u_prev=[previous])
                    command = float(result.u0[0])
                    independent_cost = float(
                        np.sum((result.x_seq[1:] - ref) ** 2 * q)
                        + 0.02 * np.sum(result.u_seq**2)
                        + np.sum((result.x_seq[-1] - ref[-1]) ** 2 * q)
                    )
                    gaps.append(abs(independent_cost - result.final_cost))
                elif kind == "pd":
                    command = float(
                        (0.7 * (x[2] * 5 - ref[0, 2] * 5) + 0.15 * x[3] * 20) / 5
                    )
                else:
                    command = 0.0
                applied = float(
                    np.clip(np.clip(command, -1, 1), previous - rate, previous + rate)
                )
                max_slew = max(max_slew, abs(applied - previous) * 5 / dt)
                x = ad @ x + bd * applied
                previous = applied
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("Nonfinite closed-loop state")
                theta, pitch_rate = float(x[2] * 5), float(x[3] * 20)
                peak_theta, peak_q = max(peak_theta, abs(theta)), max(
                    peak_q, abs(pitch_rate)
                )
                err.append(theta - ref[0, 2] * 5)
                trace.append([(k + 1) * dt, theta, pitch_rate, applied * 5])
                if abs(theta) > 10 or abs(pitch_rate) > 60:
                    failures += 1
                    break
            errors.append(float(np.sqrt(np.mean(np.square(err)))))
            traces.append(trace)
        return dict(
            mean_rmse_deg=float(np.mean(errors)),
            episode_rmse_deg=errors,
            failures=failures,
            episodes=len(cases),
            max_theta_deg=peak_theta,
            max_q_deg_s=peak_q,
            max_slew_deg_s=max_slew,
            max_reported_cost_error=max(gaps, default=0),
            traces=traces,
        )

    started = time.monotonic()
    result = dict(
        repo=str(args.repo.resolve()),
        seed=args.seed,
        checkpoint=(
            str(args.checkpoint)
            if args.checkpoint
            else str(args.output.with_suffix(".pt"))
        ),
        config=dict(
            dt_s=dt,
            duration_s=dt * steps,
            horizon=horizon,
            iters=8,
            lr=0.05,
            cases=cases,
            scales=scales.tolist(),
            action_limit_deg=5,
            Q=q.tolist(),
            R=0.02,
        ),
        learning_curve=curves,
        max_holdout_prediction_error=prediction_error(),
        zero=evaluate("zero"),
        pd=evaluate("pd"),
        evaluated=evaluate("mpc"),
    )
    result["elapsed_evaluation_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result["evaluated"].items() if k != "traces"}))


if __name__ == "__main__":
    main()
