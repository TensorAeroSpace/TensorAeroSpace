"""Compare F-16 failure timing with piecewise DOP853 and an analytic actuator.

The actuator oracle is a matrix exponential of the second-order servo equation.
Small commands keep it below the model's rate limiter. The full-state reference
shares the aerodynamic ODE but integrates separately at every physical event.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
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

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.controls import (
        ANGULAR_LEGACY_INDEX,
        apply_control_failures,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.propulsion import (
        effective_thrust,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import EngineState
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
        f16_ode_long,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16

    duration = 0.2
    command_rad = 0.01
    cases = {
        "healthy": [],
        "servo_loss_midstep": [
            DamageEvent(
                0.037,
                "control_failure",
                {"surface": "stab_left", "mode": "efficiency_loss", "efficiency": 0.4},
            )
        ],
        "servo_loss_at_reset": [
            DamageEvent(
                0.0, "control_failure", {"surface": "stab_left", "mode": "lost"}
            )
        ],
        "servo_recovery": [
            DamageEvent(
                0.113,
                "control_failure",
                {"surface": "stab_left", "mode": "efficiency_loss", "efficiency": 0.8},
            ),
            DamageEvent(
                0.037,
                "control_failure",
                {"surface": "stab_left", "mode": "efficiency_loss", "efficiency": 0.4},
            ),
        ],
        "engine_loss": [
            DamageEvent(
                0.037, "engine_failure", {"thrust_factor": 0.0, "hard_failure": True}
            )
        ],
    }

    def make_env(kind, events, dt):
        common = dict(
            number_time_steps=round(duration / dt) + 1,
            dt=dt,
            integrator="rk4",
            damage_profile=DamageProfile(events),
        )
        if kind == "angular":
            return NonlinearAngularF16(
                np.zeros(14), track_altitude=True, thrust_mode="control", **common
            )
        return NonlinearLongitudinalF16(
            np.zeros(4),
            np.zeros((1, round(duration / dt) + 2)),
            state_space=["alpha", "wz", "stab", "dstab"],
            **common,
        )

    result = {
        "repo": str(args.repo.resolve()),
        "duration_s": duration,
        "command_rad": command_rad,
        "comparisons": [],
    }
    for kind in ["angular", "longitudinal"]:
        for name, events in cases.items():
            if kind == "longitudinal" and name == "engine_loss":
                continue
            ref = make_env(kind, [], 0.01)
            ref.reset()
            model = ref.model
            params = model.param
            params.damage_state = ref.damage_manager.state
            params.damage_geometry = model.damage_geometry
            if kind == "angular":
                params.T_active = 20000.0
                ode, base_command, mapping = (
                    f16_ode_6dof,
                    np.array([command_rad, 0.0, 0.0]),
                    ANGULAR_LEGACY_INDEX,
                )
                state_indices = [8, 9]
            else:
                ode, base_command, mapping = (
                    f16_ode_long,
                    np.array([command_rad]),
                    {"stab_left": 0, "stab_right": 0},
                )
                state_indices = [2, 3]
            initial = model.current_state.copy()
            state = initial.copy()
            actuator = np.zeros(2)
            matrix = np.array(
                [[0.0, 1.0], [-1 / params.Tstab**2, -2 * params.Xistab / params.Tstab]]
            )
            ordered = sorted(events, key=lambda event: event.trigger_time)
            boundaries = sorted(
                {0.0, duration, *(event.trigger_time for event in ordered)}
            )
            cursor = 0.0
            for boundary in boundaries:
                if boundary > cursor:
                    # Apply engine damage independently of the tested ODE hook.
                    # Keep aero damage, but give the ODE an already scaled force
                    # and an otherwise healthy engine to avoid double scaling.
                    params.damage_state = copy(ref.damage_manager.state)
                    params.damage_state.engine = EngineState()
                    if kind == "angular":
                        params.T_active = effective_thrust(
                            20000.0, ref.damage_manager.state
                        )
                    applied = apply_control_failures(
                        base_command, ref.damage_manager.state, mapping
                    )
                    solution = solve_ivp(
                        lambda t, x: ode(x, applied, t, params),
                        (cursor, boundary),
                        state,
                        method="DOP853",
                        rtol=1e-12,
                        atol=1e-13,
                    )
                    if not solution.success:
                        raise AssertionError(solution.message)
                    state = solution.y[:, -1]
                    equilibrium = np.array([applied[0], 0.0])
                    actuator = equilibrium + expm(matrix * (boundary - cursor)) @ (
                        actuator - equilibrium
                    )
                for event in ordered:
                    if event.trigger_time == boundary:
                        ref.damage_manager.inject_event(event)
                ref.damage_manager.update(
                    boundary, float(np.nextafter(boundary, -np.inf))
                )
                cursor = boundary
            np.testing.assert_allclose(
                state[state_indices], actuator, rtol=0, atol=1e-10
            )
            runs = []
            for dt in [0.02, 0.01, 0.005, 0.0025]:
                env = make_env(kind, events, dt)
                env.reset()
                command = np.rad2deg(base_command)
                if kind == "angular":
                    command = np.append(command, 20000.0)
                for _ in range(round(duration / dt)):
                    env.step(command)
                final = env.model.current_state
                error = np.abs(final - state)
                runs.append(
                    {
                        "dt_s": dt,
                        "max_state_error": float(error.max()),
                        "actuator_angle_error_rad": float(
                            abs(final[state_indices[0]] - actuator[0])
                        ),
                        "actuator_rate_error_rad_s": float(
                            abs(final[state_indices[1]] - actuator[1])
                        ),
                        "state": final.tolist(),
                        "history_rows": len(env.model.x_history),
                        "event_times": [
                            event["time"] for event in env.damage_events_log
                        ],
                    }
                )
            result["comparisons"].append(
                {
                    "model": kind,
                    "case": name,
                    "reference_state": state.tolist(),
                    "analytic_actuator": actuator.tolist(),
                    "runs": runs,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in c.items() if k in ["model", "case"]}
                | {"errors": [r["max_state_error"] for r in c["runs"]]}
                for c in result["comparisons"]
            ]
        )
    )


if __name__ == "__main__":
    main()
