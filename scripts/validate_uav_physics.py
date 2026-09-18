"""Validate UAV reduction against Rauf et al. and independent held-input integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def source_matrices():
    """Rauf et al. (2011), DOI 10.1109/ICCRD.2011.5763860, pp. 91-92."""
    import numpy as np

    a = np.array(
        [
            [-0.1982, 0.593, 1.245, -9.779, -0.0001, 0.0101],
            [-0.7239, -3.9848, 18.7028, -0.6286, 0.0009, 0],
            [0.3537, -5.5023, -5.4722, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0.0641, -0.9979, 0, 19.9997, 0, 0],
            [27.838, 1.7894, 0, 0, -0.0086, -2.1436],
        ]
    )
    b = np.array([0.2281, -4.6830, -36.1341, 0, 0, 0])
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    from tensoraerospace.aerospacemodel.uav import LongitudinalUAV

    full_a, full_b = source_matrices()
    a, b = full_a[:4, :4], full_b[:4]
    initial = np.array([0.1, -0.05, 0.01, 0.02])
    result = {"repo": str(args.repo.resolve()), "runs": []}
    for dt in [0.005, 0.01, 0.02, 0.05]:
        count = round(4 / dt)
        plant = LongitudinalUAV(initial, count, dt=dt)
        reference = initial.copy()
        previous = 0.0
        errors, controls, peaks = [], [], np.zeros(4)
        for k in range(count):
            command = np.deg2rad(2 if k * dt < 0.2 else -2 if k * dt < 0.4 else 0)
            previous = float(
                np.clip(
                    command,
                    previous - np.deg2rad(60) * dt,
                    previous + np.deg2rad(60) * dt,
                )
            )
            sol = solve_ivp(
                lambda t, x: a @ x + b * previous,
                (0, dt),
                reference,
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
            )
            assert sol.success
            reference = sol.y[:, -1]
            x = plant.run_step(np.array([command])).reshape(-1)
            errors.append(float(np.max(abs(x - reference))))
            peaks = np.maximum(peaks, abs(x))
            controls.append(float(plant.store_input[0, k]))
        zero = LongitudinalUAV(initial, count, dt=dt)
        for _ in range(count):
            zero.run_step(np.array([0.0]))
        result["runs"].append(
            dict(
                dt_s=dt,
                max_ode_error=max(errors),
                max_abs_state=peaks.tolist(),
                max_rate_deg_s=float(
                    np.max(abs(np.diff([0, *controls]))) * 180 / np.pi / dt
                ),
                zero_input_error=float(
                    np.max(abs(zero.xt.reshape(-1) - expm(a * 4) @ initial))
                ),
                last_state=plant.xt.reshape(-1).tolist(),
            )
        )
    plant = LongitudinalUAV(np.zeros(4), 2)
    result["A_source_error"] = float(np.max(abs(plant.A - a)))
    result["B_source_error"] = float(np.max(abs(plant.B[:, 0] - b)))
    result["pitch_kinematics"] = plant.A[3].tolist()
    result["pitch_rate_damping_s_inverse"] = float(plant.A[2, 2])
    result["gravity_column_norm_m_s2"] = float(np.linalg.norm(plant.A[:2, 3]))
    result["reduced_poles_s_inverse"] = [
        [float(v.real), float(v.imag)] for v in np.linalg.eigvals(a)
    ]
    result["full_poles_s_inverse"] = [
        [float(v.real), float(v.imag)] for v in np.linalg.eigvals(full_a)
    ]
    result["reduction_vs_full_zero_input_4s"] = (
        expm(a * 4) @ initial - (expm(full_a * 4) @ np.r_[initial, 0, 0])[:4]
    ).tolist()
    result["limits"] = (
        "Four-state linear reduction; height/engine coupling omitted. Python outputs physical states (C=I); MATLAB first rows output airspeed/angle of attack. No nonlinear or flight-test validation."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
