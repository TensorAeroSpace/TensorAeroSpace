"""Independent published-model, actuator and environment-clock checks for Ultrastick."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def published_matrices():
    """Ahmed et al., DOI 10.4172/2168-9695.1000126, p. 6; altitude positive."""
    import numpy as np

    source = np.array(
        [
            [-0.5944, 0.8008, -9.791, -0.8747, 5.077e-5],
            [-0.744, -7.56, -0.5294, 15.72, -0.000939],
            [0, 0, 0, 1, 0],
            [1.041, -7.406, 0, -15.81, -7.284e-18],
            [-0.05399, 0.9985, -17, 0, 0],
        ]
    )
    transform = np.diag([1, 1, 1, 1, -1])
    return transform @ source @ transform, transform @ np.array(
        [0.4669, -2.703, 0, -133.7, 0]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.aerospacemodel.ultrastick import Ultrastick
    from tensoraerospace.envs.ultrastick import ImprovedUltrastickEnv

    a, b = published_matrices()
    initial = np.array([0.2, -0.1, 0.02, -0.03, 0.1])
    result = {"repo": str(args.repo.resolve()), "runs": []}
    for dt in [0.002, 0.01, 0.02, 0.05]:
        steps = round(2 / dt)
        model = Ultrastick(initial, steps, dt=dt)
        reference = initial.copy()
        applied_reference = 0.0
        errors, controls = [], []
        for k in range(steps):
            command = np.deg2rad(30 if k * dt < 0.15 else -20 if k * dt < 0.3 else 0)
            applied_reference = float(
                np.clip(
                    command,
                    applied_reference - np.deg2rad(300) * dt,
                    applied_reference + np.deg2rad(300) * dt,
                )
            )
            sol = solve_ivp(
                lambda t, x: a @ x + b * applied_reference,
                (0, dt),
                reference,
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
            )
            assert sol.success
            reference = sol.y[:, -1]
            model.run_step(np.array([command, 0.5]))
            errors.append(float(np.max(abs(model.xt.reshape(-1) - reference))))
            controls.append(float(model.store_input[0, k]))
        env = ImprovedUltrastickEnv(
            initial,
            np.zeros((1, steps + 1)),
            steps + 1,
            dt=dt,
            use_initial_action_on_first_step=False,
        )
        env.reset()
        for _ in range(steps):
            env.step([0, -1])
        expected = expm(a * (steps * dt)) @ initial
        result["runs"].append(
            {
                "dt_s": dt,
                "duration_s": steps * dt,
                "independent_ode_max_error": max(errors),
                "max_elevator_rate_deg_s": float(
                    np.max(abs(np.diff([0, *controls]))) / dt * 180 / np.pi
                ),
                "env_zero_input_error": float(
                    np.max(abs(env.model.xt.reshape(-1) - expected))
                ),
                "actual_model_duration_s": steps * env.model.dt,
            }
        )
    poles = np.linalg.eigvals(a)
    result["poles_s_inverse"] = [[float(v.real), float(v.imag)] for v in poles]
    result["all_modes_stable"] = bool(np.all(poles.real < 0))
    result["pitch_kinematics"] = a[2].tolist()
    result["limits"] = (
        "Linear perturbation model, 17 m/s trim; throttle inactive. "
        "Actuator clipping is discrete sample-and-hold, not a continuous servo. "
        "No flight-test or nonlinear-aircraft validation."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
