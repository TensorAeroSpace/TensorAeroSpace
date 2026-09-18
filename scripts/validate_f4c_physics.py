"""Check F4C against continuous integration and test PID saturation recovery.

The coefficients are the documented Python four-state model, not a flight-data
validation. The repository's separate MATLAB model has different coefficients.
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
    from types import SimpleNamespace

    import numpy as np
    from gymnasium import spaces
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.aerospacemodel.f4c import LongitudinalF4C
    from tensoraerospace.agent.pid import PID

    a = np.array(
        [
            [0.0007618, 0.0047612, 0, -9.81],
            [-0.066657, -0.28567, 180, 0],
            [0.0015124, -0.010083, -0.16384, 0],
            [0, 0, 1, 0],
        ]
    )
    b = np.array([0.0026532701, -6.8562, -5.4446, 0])
    initial = np.array([0.2, -0.1, 0.005, 0.01])
    result = dict(
        repo=str(args.repo.resolve()),
        eigenvalues=[[p.real, p.imag] for p in np.linalg.eigvals(a)],
        open_loop=[],
    )
    for dt in [0.005, 0.01, 0.02, 0.05]:
        n = round(4 / dt)
        plant = LongitudinalF4C(initial.copy(), n, dt=dt)
        state = initial.copy()
        elevator = 0.0
        errors, trace = [], []
        for i in range(n):
            t = i * dt
            command = np.deg2rad(2 if t < 0.2 else -2 if t < 0.4 else 0)
            elevator = np.clip(
                command, elevator - np.deg2rad(60) * dt, elevator + np.deg2rad(60) * dt
            )
            elevator = np.clip(elevator, -np.deg2rad(20), np.deg2rad(20))
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
        zero = LongitudinalF4C(initial.copy(), n, dt=dt)
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

    # Independent scalar unwinding oracle: same action direction for both signs of Ki.
    result["integral_recovery"] = []
    for ki in [1.0, -1.0]:
        controller = PID(
            SimpleNamespace(action_space=spaces.Box(-2.0, 2.0, (1,), np.float32)),
            kp=0,
            ki=ki,
            kd=0,
            dt=1,
        )
        controller.integral = 5 / ki
        outputs = [controller.select_action(-1 / ki, 0) for _ in range(5)]
        result["integral_recovery"].append(
            dict(ki=ki, outputs=outputs, integral=controller.integral)
        )

    dt = 0.02
    plant = LongitudinalF4C(np.zeros(4), 500, dt=dt)
    bounds = SimpleNamespace(
        action_space=spaces.Box(
            -float(np.deg2rad(2)), float(np.deg2rad(2)), (1,), np.float64
        )
    )
    pid = PID(bounds, kp=-0.7, ki=-0.4, kd=-0.25, dt=dt)
    pid.integral = -0.3
    errors, trace = [], []
    for i in range(500):
        target = np.deg2rad(2)
        measurement = plant.xt.ravel()[3]
        action = pid.select_action(target, measurement)
        x = plant.run_step([action]).ravel()
        errors.append(np.rad2deg(x[3] - target))
        trace.append(
            [
                (i + 1) * dt,
                float(np.rad2deg(x[3])),
                float(np.rad2deg(x[2])),
                float(np.rad2deg(action)),
                pid.integral,
            ]
        )
    result["pid_f4c_recovery"] = dict(
        rmse_deg=float(np.sqrt(np.mean(np.square(errors)))),
        final_error_deg=errors[-1],
        max_pitch_deg=float(np.max(np.abs(np.array(trace)[:, 1]))),
        max_pitch_rate_deg_s=float(np.max(np.abs(np.array(trace)[:, 2]))),
        trace=trace[::5],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "open_loop"}, indent=2))


if __name__ == "__main__":
    main()
