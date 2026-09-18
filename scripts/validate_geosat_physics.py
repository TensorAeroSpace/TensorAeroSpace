"""Validate GeoSat against the published reduced model and orbital equations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.aerospacemodel.geosat import GeoSat

    a = np.array([[0, 1, 0], [0.01036, 0, 0.7753], [0, -0.01774, 0]])
    b = np.array([[0], [0], [0.1512]])
    augmented = np.zeros((4, 4))
    augmented[:3, :3], augmented[:3, 3:] = a, b
    sample = GeoSat(np.zeros(3), 1)
    result = {
        "repo": str(args.repo.resolve()),
        "units": "normalized perturbations and time tau",
        "A": sample.A.tolist(),
        "B": sample.B.tolist(),
        "oscillation_frequency_per_tau": float(
            max(abs(np.linalg.eigvals(sample.A).imag))
        ),
        "held_input": [],
        "unforced": [],
        "nonlinear": [],
    }
    for dt in [0.1, 0.05, 0.01]:
        steps = round(10 / dt)
        model = GeoSat([0.02, -0.004, 0.002], steps, dt=dt)
        expected = np.array([0.02, -0.004, 0.002])
        previous = 0.0
        error = slew = 0.0
        actual_states, oracle_states, controls = [], [], []
        for index in range(steps):
            command = (
                1 if index < round(2 / dt) else (-1 if index < round(4 / dt) else 0)
            )
            applied = np.clip(
                np.clip(
                    command,
                    previous - np.deg2rad(60) * dt,
                    previous + np.deg2rad(60) * dt,
                ),
                -np.deg2rad(25),
                np.deg2rad(25),
            )
            expected = (expm(augmented * dt) @ np.append(expected, applied))[:3]
            actual = model.run_step([command]).reshape(-1)
            actual_input = model.store_input[0, index]
            previous_actual = model.store_input[0, index - 1] if index else 0
            slew = max(slew, abs(actual_input - previous_actual) / dt)
            error = max(error, float(abs(actual - expected).max()))
            actual_states.append(actual.tolist())
            oracle_states.append(expected.tolist())
            controls.append(float(actual_input))
            previous = applied
        result["held_input"].append(
            {
                "dt": dt,
                "max_state_error": error,
                "max_input_rate": float(slew),
                "actual": actual_states,
                "reference": oracle_states,
                "controls": controls,
            }
        )
    for sign in [-1, 1]:
        initial = sign * np.array([0.02, -0.004, 0.002])
        dt = 0.1
        model = GeoSat(initial, 1000, dt=dt)
        reference = solve_ivp(
            lambda t, x: a @ x,
            (0, 100),
            initial,
            t_eval=np.arange(1, 1001) * dt,
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        assert reference.success
        actual = np.array([model.run_step([0]).reshape(-1) for _ in range(1000)])
        invariant = actual[:, 2] + 0.01774 * actual[:, 0]
        result["unforced"].append(
            {
                "sign": sign,
                "dop853_error": float(abs(actual - reference.y.T).max()),
                "linear_momentum_drift": float(
                    abs(invariant - (initial[2] + 0.01774 * initial[0])).max()
                ),
            }
        )
    radius = 6.6108
    equilibrium = np.array([radius, 0, np.sqrt(1 / radius**3)])
    exact_a = np.array(
        [[0, 1, 0], [3 / radius**3, 0, 2 / np.sqrt(radius)], [0, -2 / radius**2.5, 0]]
    )
    result["exact_circular_jacobian"] = exact_a.tolist()
    result["paper_jacobian_max_rounding_difference"] = float(abs(a - exact_a).max())
    for scale in [1, 0.1, 0.01]:
        for sign in [-1, 1]:
            initial = sign * scale * np.array([0.02, -0.004, 0.002])

            def rhs(t, x):
                rho, v, w = x
                return [v, rho * w * w - 1 / rho**2, -2 * v * w / rho]

            times = np.arange(1, 101) * 0.1
            solution = solve_ivp(
                rhs,
                (0, 10),
                equilibrium + initial,
                t_eval=times,
                method="DOP853",
                rtol=1e-12,
                atol=1e-14,
            )
            assert solution.success
            physical = solution.y.T - equilibrium
            model = GeoSat(initial, 100, dt=0.1)
            actual = np.array([model.run_step([0]).reshape(-1) for _ in range(100)])
            exact = np.array([expm(exact_a * t) @ initial for t in times])
            momentum = solution.y[0] ** 2 * solution.y[2]
            initial_momentum = (equilibrium[0] + initial[0]) ** 2 * (
                equilibrium[2] + initial[2]
            )
            result["nonlinear"].append(
                {
                    "scale": scale,
                    "sign": sign,
                    "model_error": float(abs(actual - physical).max()),
                    "exact_jacobian_error": float(abs(exact - physical).max()),
                    "nonlinear_momentum_drift": float(
                        abs(momentum - initial_momentum).max()
                    ),
                }
            )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "held_input"}))


if __name__ == "__main__":
    main()
