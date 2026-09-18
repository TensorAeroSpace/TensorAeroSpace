"""Check the F-16 environment against model units, trim and adaptive integration.

The adaptive solver shares the ODE; this validates integration and the env/model
boundary, not aerodynamic calibration against flight measurements.
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

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    result = {"repo": str(args.repo.resolve()), "cases": []}
    env = NonlinearAngularF16(
        np.zeros(14), 100, track_altitude=True, thrust_mode="control"
    )
    obs, _ = env.reset()
    result["reset_shape"] = list(obs.shape)
    result["space_shape"] = list(env.observation_space.shape)
    result["thrust_bounds_N"] = [
        float(env.action_space.low[-1]),
        float(env.action_space.high[-1]),
    ]
    for speed, altitude in [(120, 1000), (120, 3000), (150, 2500), (180, 6000)]:
        trim = find_trim(V_target=speed, h_target=altitude)
        if not trim.converged:
            raise AssertionError("Trim must converge for the validation envelope")
        command = np.array([trim.stab_rad, 0, 0, trim.T_thrust])
        model = AngularF16(
            trim.x0,
            dt=0.01,
            integrator="rk4",
            track_altitude=True,
            thrust_mode="control",
        )
        for _ in range(500):
            model.run_step(command)
        drift = np.abs(model.current_state - trim.x0)
        case = {
            "speed_m_s": speed,
            "altitude_m": altitude,
            "trim_thrust_N": trim.T_thrust,
            "trim_residuals": list(trim.residuals),
            "model_trim_drift_5s": drift.tolist(),
            "model_final_state": model.current_state.tolist(),
        }
        try:
            env = NonlinearAngularF16(
                trim.x0, 500, dt=0.01, track_altitude=True, thrust_mode="control"
            )
        except ValueError as exc:
            case["env_construction_error"] = str(exc)
        else:
            env.reset()
            command_deg = command.copy()
            command_deg[:3] = np.rad2deg(command[:3])
            for _ in range(500):
                env.step(command_deg)
            case["env_model_max_difference"] = float(
                np.max(np.abs(env.model.current_state - model.current_state))
            )
            case["env_trim_drift_5s"] = np.abs(
                env.model.current_state - trim.x0
            ).tolist()

            # Excite all three actuators and thrust. Stay within actuator rate limits.
            transient = command + np.array([0.01, 0.005, -0.003, 2000])
            params = default_parameters()
            params.T_active = float(transient[-1])
            reference = solve_ivp(
                lambda t, x: f16_ode_6dof(x, transient[:3], t, params),
                (0, 0.5),
                trim.x0,
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
            )
            if not reference.success:
                raise AssertionError(reference.message)
            errors = []
            for dt in (0.01, 0.005, 0.0025):
                env = NonlinearAngularF16(
                    trim.x0,
                    round(0.5 / dt),
                    dt=dt,
                    track_altitude=True,
                    thrust_mode="control",
                )
                env.reset()
                control = transient.copy()
                control[:3] = np.rad2deg(control[:3])
                for _ in range(round(0.5 / dt)):
                    env.step(control)
                error = np.abs(env.model.current_state - reference.y[:, -1])
                errors.append(
                    {
                        "dt_s": dt,
                        "max_state_error": float(error.max()),
                        "altitude_error_m": float(error[14]),
                        "speed_error_m_s": float(error[15]),
                    }
                )
            case["transient_rk4_vs_dop853"] = errors
        result["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
