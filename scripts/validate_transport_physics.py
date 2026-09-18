"""Check transport thrust continuity, Newton-Euler balance and integration.

Run each repository revision separately; no flight-data calibration is claimed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def audit_transport(name):
    import numpy as np
    from scipy.integrate import solve_ivp

    prefix = f"tensoraerospace.aerospacemodel.{name}.nonlinear"
    package = importlib.import_module(prefix)
    params = package.default_parameters()
    dynamics = getattr(
        importlib.import_module(prefix + ".dynamics"), name + "_ode_6dof"
    )
    aero_module = importlib.import_module(prefix + ".aero")
    aero_fn = getattr(aero_module, name + "_aero")
    engine_module = importlib.import_module(prefix + ".engine")
    thrust_fn = getattr(
        engine_module,
        (
            "jt9d_thrust_with_asymmetry"
            if name == "b747"
            else "b737_thrust_with_asymmetry"
        ),
    )
    atmosphere = importlib.import_module(prefix + ".params")
    engine = getattr(engine_module, "JT9DEngine" if name == "b747" else "B737Engine")()
    model_cls = getattr(package, "Nonlinear" + name.upper())
    inertia = np.array(
        [[params.Ix, 0, -params.Ixz], [0, params.Iy, 0], [-params.Ixz, 0, params.Iz]]
    )
    rng = np.random.default_rng(20260917)
    energy_errors, moment_errors = [], []
    from types import SimpleNamespace

    for i in range(100):
        params.damage_state = (
            SimpleNamespace(engines_mu={1: 0.0}, flap_jam_config=None)
            if i % 2
            else None
        )
        speed, alpha, beta = (
            rng.uniform(500, 750),
            rng.uniform(0.01, 0.1),
            rng.uniform(-0.05, 0.05),
        )
        x = np.zeros(12)
        x[:3] = speed * np.array(
            [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)]
        )
        x[3:9] = rng.uniform(-0.1, 0.1, 6)
        x[11] = -rng.uniform(0, 40000)
        u = np.r_[rng.uniform(-0.03, 0.03, 3), rng.uniform(0.3, 1.0)]
        aero = aero_fn(
            aero_module.AeroState(alpha, beta, speed, *x[3:6], -x[11], *u[:3]), params
        )
        thrust, yaw = thrust_fn(
            u[3], speed / atmosphere.isa_speed_of_sound_ft_s(-x[11]), -x[11], params
        )
        moments = np.array([aero.l, aero.m, aero.n + yaw])
        dx = dynamics(x, u, 0, params)
        moment_errors.append(
            float(
                np.max(
                    np.abs(
                        inertia @ dx[3:6] + np.cross(x[3:6], inertia @ x[3:6]) - moments
                    )
                )
            )
        )
        energy_rate = (
            params.mass_slug * x[:3] @ dx[:3]
            + x[3:6] @ inertia @ dx[3:6]
            - params.weight_lb * dx[11]
        )
        power = (
            thrust * x[0]
            - aero.D * speed * np.cos(beta)
            + aero.Y * x[1]
            + moments @ x[3:6]
        )
        energy_errors.append(abs(float(energy_rate - power)))
    params.damage_state = None
    hs = [36089 - 0.001, 36089.0, 36089 + 0.001]
    thrusts = [engine.installed_thrust(0.8, h, 1.0) for h in hs]
    speed = 700.0 if name == "b747" else 738.0
    result = package.trim(20000.0, speed)
    if not result.converged:
        raise RuntimeError("Reference low-altitude trim did not converge")
    # Start just below the layer boundary and cross it while climbing.
    x0 = result.to_state()
    x0[11] = -36088.0
    x0[7] += 0.01
    command = np.array([result.elevator_rad, 0, 0, result.throttle])
    ref = solve_ivp(
        lambda t, x: dynamics(x, command, t, params),
        (0.0, 2.0),
        x0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        max_step=0.01,
    )
    errors = []
    for dt in [0.04, 0.02, 0.01]:
        model = model_cls(x0, dt=dt)
        for _ in range(round(2 / dt)):
            model.run_step(command)
        errors.append(float(np.max(np.abs(model.current_state - ref.y[:, -1]))))
    grid = []
    for altitude in [20000.0, 30000.0, 36089.0, 37000.0]:
        for velocity in [speed * 0.95, speed, speed * 1.05]:
            tr = package.trim(altitude, velocity)
            grid.append(
                {
                    "altitude_ft": altitude,
                    "speed_ft_s": velocity,
                    "converged": bool(tr.converged),
                    "residual": tr.residual,
                    "throttle": tr.throttle,
                    "elevator_rad": tr.elevator_rad,
                }
            )
    return {
        "energy_balance_max_error_ft_lbf_s": max(energy_errors),
        "moment_balance_max_error_lbf_ft": max(moment_errors),
        "random_states": 100,
        "engine_out_states": 50,
        "boundary_altitudes_ft": hs,
        "boundary_thrust_lbf": thrusts,
        "boundary_relative_jump": abs(thrusts[-1] / thrusts[0] - 1),
        "integration": {
            "reference_success": bool(ref.success),
            "final_altitude_ft": float(-ref.y[11, -1]),
            "max_altitude_ft": float(np.max(-ref.y[11])),
            "duration_s": 2.0,
            "rk4_dt": [0.04, 0.02, 0.01],
            "max_component_error": errors,
        },
        "trim_grid": grid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    result = {name: audit_transport(name) for name in ["b747", "b737"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
