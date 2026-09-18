"""Check the implemented linear B747 equations against independent integration.

Coefficients below freeze the pre-fix implementation, not independently verified
flight data. This checks numerical consistency, kinematics and actuator limits.
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
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.aerospacemodel import LongitudinalB747

    a = np.array(
        [
            [-0.0069, 0.0139, 0, -9.81],
            [-0.0905, -0.3149, 235.8928, 0],
            [0.0004, -0.0034, -0.4282, 0],
            [0, 0, 1, 0],
        ]
    )
    b = np.array([-0.0001, -5.5079, -1.1569, 0])
    initial = np.array([0.2, -0.1, 0.005, 0.01])
    result = dict(
        repo=str(args.repo.resolve()),
        eigenvalues=[[p.real, p.imag] for p in np.linalg.eigvals(a)],
        open_loop=[],
    )
    for dt in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        n = round(4 / dt)
        plant = LongitudinalB747(initial.copy(), n, dt=dt)
        state = initial.copy()
        elevator = 0.0
        errors, trace = [], []
        for i in range(n):
            t = i * dt
            command = np.deg2rad(2 if t < 0.2 else -2 if t < 0.4 else 0)
            elevator = np.clip(
                command,
                elevator - np.deg2rad(60) * dt,
                elevator + np.deg2rad(60) * dt,
            )
            elevator = np.clip(elevator, -np.deg2rad(25), np.deg2rad(25))
            continuous = solve_ivp(
                lambda _, x: a @ x + b * elevator,
                (0, dt),
                state,
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
            )
            assert continuous.success
            state = continuous.y[:, -1]
            actual = plant.run_step([command]).ravel()
            errors.append(abs(actual - state))
            trace.append(
                [
                    (i + 1) * dt,
                    float(np.rad2deg(actual[3])),
                    float(np.rad2deg(actual[2])),
                    float(np.rad2deg(plant.store_input[0, i])),
                ]
            )
        controls = np.r_[0.0, plant.store_input[0]]
        zero = LongitudinalB747(initial.copy(), n, dt=dt)
        for _ in range(n):
            zero.run_step([0])
        result["open_loop"].append(
            dict(
                dt=dt,
                max_state_error=float(np.max(errors)),
                component_max_errors=np.max(errors, axis=0).tolist(),
                zero_input_error=float(
                    np.max(abs(zero.xt.ravel() - expm(a * 4) @ initial))
                ),
                max_rate_deg_s=float(np.rad2deg(np.max(abs(np.diff(controls))) / dt)),
                matrix_error=float(
                    max(np.max(abs(plant.A - a)), np.max(abs(plant.B.ravel() - b)))
                ),
                trace=trace[:: max(1, n // 100)],
            )
        )

    controllability = np.column_stack(
        [np.linalg.matrix_power(a, i) @ b for i in range(4)]
    )
    result["controllability_rank"] = int(np.linalg.matrix_rank(controllability))
    result["coefficient_provenance"] = (
        "Frozen B747 implementation; external identification not verified"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "open_loop"}, indent=2))


if __name__ == "__main__":
    main()
