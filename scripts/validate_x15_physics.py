"""Check X-15 burnout against a rocket impulse and piecewise DOP853 solution."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp

    from tensoraerospace.aerospacemodel.x15.nonlinear import (
        NonlinearX15,
        default_parameters,
    )
    from tensoraerospace.aerospacemodel.x15.nonlinear.engine import xlr99_thrust
    from tensoraerospace.aerospacemodel.x15.nonlinear.initial import set_initial_state

    dynamics = importlib.import_module(
        "tensoraerospace.aerospacemodel.x15.nonlinear.dynamics"
    )
    params = default_parameters()
    _, flow = xlr99_thrust(1.0, 1000.0, params)
    original_aero = dynamics.x15_aero
    dynamics.x15_aero = lambda *a: SimpleNamespace(
        L=0.0, D=0.0, Y=0.0, l=0.0, m=0.0, n=0.0
    )
    impulse = []
    try:
        for integrator in ("euler", "rk4"):
            for burn_time in (0.1, 0.5, 1.0):
                fuel = flow * burn_time
                state = set_initial_state(
                    altitude_ft=70000.0,
                    V_ft_s=1000.0,
                    alpha_deg=0.0,
                    propellant_lb=fuel,
                )
                model = NonlinearX15(state, dt=1.0, integrator=integrator)
                model.run_step([0.0, 0.0, 0.0, 1.0])
                exact = (
                    params.engine_isp_s
                    * params.g_ft_s2
                    * np.log1p(fuel / params.empty_weight_lb)
                )
                actual = model.current_state[0] - state[0]
                impulse.append(
                    dict(
                        integrator=integrator,
                        burn_time_s=burn_time,
                        expected_delta_u_ft_s=float(exact),
                        actual_delta_u_ft_s=float(actual),
                        abs_error_ft_s=float(abs(actual - exact)),
                        fuel_lb=model.propellant_lb,
                    )
                )
    finally:
        dynamics.x15_aero = original_aero

    def reference(state, control, duration, burnout):
        def powered(t, x):
            x = x.copy()
            x[12] = max(x[12], np.finfo(float).tiny)
            return dynamics.x15_ode_6dof(x, control, t, params)

        solution = solve_ivp(
            powered,
            (0.0, min(duration, burnout)),
            state,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            max_step=0.002,
        )
        assert solution.success
        end = solution.y[:, -1]
        if burnout < duration:
            end[12] = 0.0
            coast = control.copy()
            coast[3] = 0.0
            solution = solve_ivp(
                lambda t, x: dynamics.x15_ode_6dof(x, coast, t, params),
                (burnout, duration),
                end,
                method="DOP853",
                rtol=1e-12,
                atol=1e-12,
                max_step=0.002,
            )
            assert solution.success
            end = solution.y[:, -1]
        return end

    scenarios = []
    for burnout in (0.037, 0.37, 10.0):
        state = set_initial_state(
            altitude_ft=70000.0,
            V_ft_s=2000.0,
            alpha_deg=4.0,
            propellant_lb=flow * burnout,
        )
        control = np.array([np.deg2rad(-1.0), 0.0, 0.0, 1.0])
        expected = reference(state, control, 1.0, burnout)
        for dt in (0.2, 0.1, 0.05, 0.02):
            model = NonlinearX15(state, dt=dt)
            for _ in range(round(1.0 / dt)):
                model.run_step(control)
            error = model.current_state - expected
            scenarios.append(
                dict(
                    burn_time_s=burnout,
                    dt_s=dt,
                    max_velocity_error_ft_s=float(np.max(np.abs(error[:3]))),
                    max_angular_rate_error_rad_s=float(np.max(np.abs(error[3:6]))),
                    max_attitude_error_rad=float(np.max(np.abs(error[6:9]))),
                    max_position_error_ft=float(np.max(np.abs(error[9:12]))),
                    propellant_error_lb=float(error[12]),
                    final_state=model.current_state.tolist(),
                    reference_state=expected.tolist(),
                )
            )
    result = {
        "repo": str(args.repo.resolve()),
        "rocket_equation": impulse,
        "dop853_comparison": scenarios,
        "limitations": [
            "Same aerodynamic RHS in reference; validates integration, not coefficient calibration.",
            "Constant commanded throttle within each integration step; no flight-data validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in impulse:
        print(row)
    for row in scenarios:
        print({k: v for k, v in row.items() if not k.endswith("state")})


if __name__ == "__main__":
    main()
