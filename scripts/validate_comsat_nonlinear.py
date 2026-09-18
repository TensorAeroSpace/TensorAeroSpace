"""Check ComSat local linearization and frozen policies against orbital ODEs.

Uses the independent normalized nonlinear equations of Choudhary (2015),
without atmospheric drag, perturbing bodies, actuator calibration or SI claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--policy-results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.agent.sac.model import GaussianPolicy

    torch.set_num_threads(1)
    source = json.loads(args.policy_results.read_text())
    cfg = source["config"]
    dt, limit = cfg["dt_tau"], cfg["input_limit_normalized"]
    scales, q = np.array(cfg["scales"]), np.array(cfg["q"])
    gain = np.array(cfg["lqr_gain"])
    r0 = 6.6108
    w0 = np.sqrt(1 / r0**3)  # Exact circular equilibrium, before paper rounding.
    equilibrium = np.array([r0, 0, w0])
    jacobian = np.array([[0, 1, 0], [3 / r0**3, 0, 2 * r0 * w0], [0, -2 * w0 / r0, 0]])
    paper = np.array([[0, 1, 0], [0.01036, 0, 0.7757], [0, -0.01775, 0]])

    def nonlinear(_t, state, control):
        rho, velocity, omega = state
        return [
            velocity,
            rho * omega**2 - 1 / rho**2,
            -2 * velocity * omega / rho + control / rho,
        ]

    def advance(state, control):
        solution = solve_ivp(
            lambda t, x: nonlinear(t, x, control),
            (0, dt),
            state,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise FloatingPointError("Nonlinear orbit integration failed")
        return solution.y[:, -1]

    result = {
        "seed": source["seed"],
        "checkpoint": str(args.checkpoint),
        "units": "normalized perturbations; time tau",
        "equilibrium": equilibrium.tolist(),
        "exact_jacobian": jacobian.tolist(),
        "linearization": [],
    }
    grid = np.linspace(0, 10, 101)
    for scale in [1, 0.1, 0.01]:
        for sign in [-1, 1]:
            delta = sign * scale * np.array([0.02, -0.004, 0.002])
            solution = solve_ivp(
                lambda t, x: nonlinear(t, x, 0),
                (0, 10),
                equilibrium + delta,
                t_eval=grid,
                method="DOP853",
                rtol=1e-12,
                atol=1e-14,
            )
            assert solution.success
            actual = solution.y.T - equilibrium
            exact = np.array([expm(jacobian * t) @ delta for t in grid])
            published = np.array([expm(paper * t) @ delta for t in grid])
            angular_momentum = solution.y[0] ** 2 * solution.y[2]
            result["linearization"].append(
                {
                    "scale": scale,
                    "sign": sign,
                    "exact_jacobian_error": float(abs(actual - exact).max()),
                    "published_matrix_error": float(abs(actual - published).max()),
                    "nonlinear_angular_momentum_drift": float(
                        abs(angular_momentum - angular_momentum[0]).max()
                    ),
                }
            )

    policy = GaussianPolicy(4, 1, 64, gym.spaces.Box(-1, 1, (1,), np.float32))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(saved["policy"])
    policy.eval()

    def learned(obs):
        with torch.no_grad():
            mean, _ = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return float(torch.tanh(mean).item())

    for name, controller in [
        ("zero", lambda obs: 0.0),
        ("lqr", lambda obs: float((-(gain @ (obs[:3] * scales)) / limit).item())),
        ("policy", learned),
    ]:
        costs, errors, traces = [], [], []
        failed = 0
        maximum = np.zeros(3)
        for case in cfg["cases"]:
            state = equilibrium + np.array(case)
            previous = 0.0
            trace = []
            for _ in range(round(cfg["horizon_tau"] / dt)):
                obs = np.append((state - equilibrium) / scales, previous / limit)
                command = float(np.clip(controller(obs), -1, 1)) * limit
                applied = float(
                    np.clip(command, previous - 60 * dt, previous + 60 * dt)
                )
                state = advance(state, applied)
                delta = state - equilibrium
                maximum = np.maximum(maximum, abs(delta))
                trace.append(delta.tolist())
                previous = applied
                if np.any(abs(delta) > [0.2, 0.05, 0.02]):
                    failed += 1
                    break
            states = np.array(trace)
            traces.append(trace)
            costs.append(float(np.mean(np.einsum("ni,ij,nj->n", states, q, states))))
            errors.append(np.sqrt(np.mean(states**2, axis=0)).tolist())
        result[name] = {
            "mean_state_cost": float(np.mean(costs)),
            "mean_rmse": np.mean(errors, axis=0).tolist(),
            "episode_state_cost": costs,
            "failed_episodes": failed,
            "episodes": len(cfg["cases"]),
            "max_abs_state": maximum.tolist(),
            "traces": traces,
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: {n: v for n, v in result[k].items() if n != "traces"}
                for k in ["zero", "lqr", "policy"]
            }
        )
    )


if __name__ == "__main__":
    main()
