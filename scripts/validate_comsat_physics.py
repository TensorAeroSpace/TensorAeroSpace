"""Validate normalized ComSat dynamics against equations and a held-input oracle."""

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

    from tensoraerospace.aerospacemodel.comsat import ComSat
    from tensoraerospace.envs.comsat import ImprovedComSatEnv

    a = np.array([[0.0, 1.0, 0.0], [0.01036, 0.0, 0.7757], [0.0, -0.01775, 0.0]])
    b = np.array([0.0, 0.0, 0.1513])
    r0, w0 = 6.6108, 0.0587
    jacobian = np.array(
        [
            [0.0, 1.0, 0.0],
            [w0 * w0 + 2 / r0**3, 0.0, 2 * r0 * w0],
            [0.0, -2 * w0 / r0, 0.0],
        ]
    )
    result = {
        "repo": str(args.repo.resolve()),
        "units": "normalized perturbations; time tau",
        "paper_matrix": a.tolist(),
        "nonlinear_jacobian_at_published_operating_point": jacobian.tolist(),
        "rounding_max_difference": float(abs(a - jacobian).max()),
        "held_input": [],
        "zero_input": [],
        "origin_invariance": [],
    }
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = a
    augmented[:3, 3] = b
    for dt in [0.1, 0.05, 0.01]:
        steps = round(10 / dt)
        model = ComSat([0.02, -0.004, 0.002], steps, dt=dt)
        expected = np.array([0.02, -0.004, 0.002])
        maximum_error = 0.0
        maximum_slew = 0.0
        previous = 0.0
        states, reference, commands = [], [], []
        for index in range(steps):
            requested = (
                25.0
                if index < round(2 / dt)
                else (-25.0 if index < round(4 / dt) else 0.001)
            )
            applied = float(
                np.clip(
                    np.clip(requested, previous - 60 * dt, previous + 60 * dt), -25, 25
                )
            )
            expected = (expm(augmented * dt) @ np.append(expected, applied))[:3]
            actual = model.run_step([requested]).reshape(-1)
            actual_control = model.store_input[0, index]
            last_actual = model.store_input[0, index - 1] if index else 0.0
            maximum_slew = max(maximum_slew, abs(actual_control - last_actual) / dt)
            maximum_error = max(maximum_error, float(abs(actual - expected).max()))
            states.append(actual.tolist())
            reference.append(expected.tolist())
            commands.append(float(actual_control))
            previous = applied
        result["held_input"].append(
            {
                "dt": dt,
                "max_state_error": maximum_error,
                "max_slew_per_tau": maximum_slew,
                "states": states,
                "reference": reference,
                "applied_inputs": commands,
            }
        )
    for initial in [[0.02, -0.004, 0.002], [-0.02, 0.004, -0.002]]:
        model = ComSat(initial, 1000, dt=0.1)
        initial = np.array(initial)
        h0 = initial[2] + 0.01775 * initial[0]
        invariant_error = 0.0
        ode_error = 0.0
        continuous = solve_ivp(
            lambda t, x: a @ x,
            (0, 100),
            initial,
            t_eval=np.arange(1, 1001) * 0.1,
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        assert continuous.success
        for expected in continuous.y.T:
            state = model.run_step([0.0]).reshape(-1)
            invariant_error = max(
                invariant_error, abs(state[2] + 0.01775 * state[0] - h0)
            )
            ode_error = max(ode_error, float(abs(state - expected).max()))
        result["zero_input"].append(
            {
                "initial": initial.tolist(),
                "angular_momentum_linear_invariant_error": float(invariant_error),
                "dop853_max_error": ode_error,
            }
        )
    for nominal in [0.0, 6.6108, 6371.0]:
        env = ImprovedComSatEnv(
            [nominal, 0, 0], np.zeros((1, 101)), 101, nominal_rho=nominal, dt=0.1
        )
        env.reset()
        maximum = 0.0
        failed = False
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step([0.0])
            maximum = max(
                maximum, float(abs(env.state - np.array([nominal, 0, 0])).max())
            )
            if terminated:
                failed = True
                break
            if truncated:
                break
        result["origin_invariance"].append(
            {
                "nominal_rho": nominal,
                "max_unforced_deviation": maximum,
                "terminated": failed,
                "steps": env.current_step,
            }
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "held_input"}))


if __name__ == "__main__":
    main()
