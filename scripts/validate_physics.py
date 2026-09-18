"""Report numerical physical invariants and trim drift for a repository revision.

This checks internal physical consistency, not agreement with flight-test data.
Run separately for each --repo and retain the JSON outputs for comparison.
"""

from __future__ import annotations

import argparse
import importlib
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

    from tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics import (
        quadrotor_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.quadrotor.nonlinear.model import (
        NonlinearQuadrotor,
    )
    from tensoraerospace.aerospacemodel.quadrotor.nonlinear.params import (
        QuadrotorParameters,
    )

    p = QuadrotorParameters()
    rng = np.random.default_rng(20260916)
    errors = []
    for _ in range(100):
        x = rng.normal(0, 0.2, 12)
        u = np.array([p.m * p.g, 0.02, -0.03, 0.01])
        dx = quadrotor_ode_6dof(x, u, 0.0, p)
        inertia = np.array([p.Jx, p.Jy, p.Jz])
        energy_rate = (
            p.m * x[3:6] @ dx[3:6] + (inertia * x[9:12]) @ dx[9:12] - p.m * p.g * dx[2]
        )
        external_power = (
            -u[0] * x[5]
            + u[1:] @ x[9:12]
            - np.array([p.kdx, p.kdy, p.kdz]) @ (x[3:6] ** 2)
        )
        errors.append(abs(float(energy_rate - external_power)))
    hover = quadrotor_ode_6dof(
        np.zeros(12), np.array([p.m * p.g, 0.0, 0.0, 0.0]), 0.0, p
    )
    initial = np.array([0.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.1, -0.1, 0.0, 0.02, 0.0, 0.01])
    command = np.array([p.m * p.g, 0.002, -0.003, 0.001])
    ref = solve_ivp(
        lambda t, x: quadrotor_ode_6dof(x, command, t, p),
        (0.0, 2.0),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    step_errors = []
    for dt in [0.08, 0.04, 0.02]:
        model = NonlinearQuadrotor(initial, dt=dt, integrator="rk4")
        for _ in range(round(2 / dt)):
            model.run_step(command)
        step_errors.append(float(np.max(np.abs(model.current_state - ref.y[:, -1]))))
    result = {
        "repo": str(args.repo.resolve()),
        "quadrotor": {
            "energy_balance_max_error_W": max(errors),
            "hover_derivative_max": float(np.max(np.abs(hover))),
            "rk4_dt": [0.08, 0.04, 0.02],
            "rk4_max_error": step_errors,
        },
        "aircraft": [],
    }

    cases = [
        ("b747", "NonlinearB747", 30000.0, 750.0),
        ("b737", "NonlinearB737", 25000.0, 738.0),
        ("skywalker_x8", "NonlinearSkywalkerX8", 178.0, 18.0),
        ("aai_shadow", "NonlinearAAIShadow", 1000.0, 36.0),
    ]
    for name, class_name, altitude, speed in cases:
        prefix = f"tensoraerospace.aerospacemodel.{name}.nonlinear"
        trimmer = importlib.import_module(prefix + ".trim").trim
        model_class = getattr(importlib.import_module(prefix + ".model"), class_name)
        for factor in [0.95, 1.0, 1.05]:
            trim = trimmer(altitude, speed * factor)
            x0 = trim.to_state()
            model = model_class(x0, dt=0.01, integrator="rk4")
            command = np.array(
                [trim.elevator_rad, getattr(trim, "aileron_rad", 0.0), trim.throttle]
                if name == "skywalker_x8"
                else [trim.elevator_rad, 0.0, 0.0, trim.throttle]
            )
            for _ in range(500):
                model.run_step(command)
            final = model.current_state
            result["aircraft"].append(
                {
                    "model": name,
                    "speed": speed * factor,
                    "altitude": altitude,
                    "units": "ft, ft/s" if name.startswith("b7") else "m, m/s",
                    "trim_converged": bool(trim.converged),
                    "trim_residual": trim.residual,
                    "throttle": trim.throttle,
                    "max_velocity_drift": float(np.max(np.abs(final[:3] - x0[:3]))),
                    "max_attitude_drift_rad": float(
                        np.max(np.abs(final[6:9] - x0[6:9]))
                    ),
                    "altitude_drift": float(abs(final[11] - x0[11])),
                }
            )

    # Check X8 transient integration against an independent adaptive solver.
    from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import (
        NonlinearSkywalkerX8,
        default_parameters,
    )
    from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import trim as x8_trim
    from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import (
        x8_ode_6dof,
    )

    x8 = x8_trim(178.0, 18.0)
    x8_initial = x8.to_state()
    x8_command = np.array(
        [x8.elevator_rad, getattr(x8, "aileron_rad", 0.0), x8.throttle]
    )
    x8_command += np.array([np.deg2rad(1.0), np.deg2rad(0.5), 0.01])
    x8_params = default_parameters()
    x8_ref = solve_ivp(
        lambda t, x: x8_ode_6dof(x, x8_command, t, x8_params),
        (0.0, 2.0),
        x8_initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    x8_errors = []
    for dt in [0.04, 0.02, 0.01]:
        model = NonlinearSkywalkerX8(x8_initial, dt=dt)
        for _ in range(round(2.0 / dt)):
            model.run_step(x8_command)
        x8_errors.append(float(np.max(np.abs(model.current_state - x8_ref.y[:, -1]))))
    result["skywalker_x8_integration"] = {
        "reference_solver_success": bool(x8_ref.success),
        "duration_s": 2.0,
        "rk4_dt_s": [0.04, 0.02, 0.01],
        "rk4_max_state_error": x8_errors,
    }

    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
        NonlinearAAIShadow,
    )
    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
        default_parameters as shadow_parameters,
    )
    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
        shadow_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import trim as shadow_trim

    shadow = shadow_trim(1000.0, 36.0)
    shadow_initial = shadow.to_state()
    shadow_command = np.array(
        [
            shadow.elevator_rad + np.deg2rad(1.0),
            np.deg2rad(0.5),
            np.deg2rad(-0.25),
            shadow.throttle - 0.01,
        ]
    )
    shadow_params = shadow_parameters()
    shadow_ref = solve_ivp(
        lambda t, x: shadow_ode_6dof(x, shadow_command, t, shadow_params),
        (0.0, 2.0),
        shadow_initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    shadow_errors = []
    for dt in [0.04, 0.02, 0.01]:
        model = NonlinearAAIShadow(shadow_initial, dt=dt)
        for _ in range(round(2.0 / dt)):
            model.run_step(shadow_command)
        shadow_errors.append(
            float(np.max(np.abs(model.current_state - shadow_ref.y[:, -1])))
        )
    result["aai_shadow_integration"] = {
        "reference_solver_success": bool(shadow_ref.success),
        "duration_s": 2.0,
        "rk4_dt_s": [0.04, 0.02, 0.01],
        "rk4_max_state_error": shadow_errors,
    }

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim

    result["f16"] = []
    for speed, height in [(100.0, 1000.0), (120.0, 3000.0), (160.0, 5000.0)]:
        trim = find_trim(V_target=speed, h_target=height)
        model = AngularF16(trim.x0, dt=0.01, integrator="rk4", track_altitude=True)
        model.param.T_thrust = trim.T_thrust
        for _ in range(500):
            model.run_step(np.array([trim.stab_rad, 0.0, 0.0]))
        final = model.current_state
        result["f16"].append(
            {
                "speed_m_s": speed,
                "altitude_m": height,
                "converged": bool(trim.converged),
                "thrust_N": trim.T_thrust,
                "residuals": list(trim.residuals),
                "speed_drift_m_s": float(final[15] - speed),
                "altitude_drift_m": float(final[14] - height),
                "pitch_drift_deg": float(np.rad2deg(final[7] - trim.alpha_rad)),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
