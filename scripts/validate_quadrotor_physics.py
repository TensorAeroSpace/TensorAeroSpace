"""Independent checks of scheduled rotor failures and continuous loss of thrust."""

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

    from tensoraerospace.aerospacemodel.quadrotor.allocation import default_allocator
    from tensoraerospace.aerospacemodel.quadrotor.damage import (
        DamageProfile,
        MotorEfficiencyDecay,
        RotorDamageEvent,
        RotorLossEvent,
    )
    from tensoraerospace.aerospacemodel.quadrotor.nonlinear.dynamics import (
        quadrotor_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.quadrotor.nonlinear.params import (
        default_parameters,
    )
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    params = default_parameters()
    params.kdx = params.kdy = params.kdz = 0.0
    allocator = default_allocator()
    omega = allocator.unmix(np.array([params.m * params.g, 0.0, 0.0, 0.0]))

    def events(name):
        if name == "loss":
            return [RotorLossEvent(0.037, i) for i in range(4)]
        if name == "decay":
            return [
                MotorEfficiencyDecay(0.037, i, tau=0.08, mu_floor=0.2) for i in range(4)
            ]
        if name == "asymmetric":
            return [
                RotorDamageEvent(0.073, 0, mu=0.6),
                MotorEfficiencyDecay(0.11, 2, tau=0.2, mu_floor=0.3),
            ]
        return []

    def mu(name, t):
        result = np.ones(4)
        if name == "loss" and t >= 0.037:
            result[:] = 0.0
        elif name == "decay" and t >= 0.037:
            result[:] = 0.2 + 0.8 * np.exp(-(t - 0.037) / 0.08)
        elif name == "asymmetric":
            if t >= 0.073:
                result[0] = 0.6
            if t >= 0.11:
                result[2] = 0.3 + 0.7 * np.exp(-(t - 0.11) / 0.2)
        return result

    result = []
    duration = 0.2
    for name in ["healthy", "loss", "decay", "asymmetric"]:
        bounds = sorted({0.0, duration, *[e.trigger_time for e in events(name)]})
        reference = np.zeros(12)
        for start, end in zip(bounds[:-1], bounds[1:]):

            def rhs(t, x):
                command = allocator.mix(
                    mu(name, min(t, np.nextafter(end, -np.inf))) * omega
                )
                return quadrotor_ode_6dof(x, command, t, params)

            solution = solve_ivp(
                rhs,
                (start, end),
                reference,
                method="DOP853",
                rtol=1e-12,
                atol=1e-12,
                max_step=0.001,
            )
            assert solution.success
            reference = solution.y[:, -1]
        for dt in [0.05, 0.02, 0.01, 0.005]:
            env = NonlinearQuadrotorEnv(
                np.zeros(12),
                round(duration / dt),
                dt=dt,
                action_space="rotor",
                damage_profile=DamageProfile(events(name)),
            )
            env.reset()
            env.model.set_param(params)
            for _ in range(round(duration / dt)):
                env.step(omega)
            state = env.model.current_state
            error = state - reference
            result.append(
                dict(
                    scenario=name,
                    dt_s=dt,
                    max_position_error_m=float(max(abs(error[:3]))),
                    max_velocity_error_m_s=float(max(abs(error[3:6]))),
                    max_attitude_error_rad=float(max(abs(error[6:9]))),
                    max_rate_error_rad_s=float(max(abs(error[9:12]))),
                    final_state=state.tolist(),
                    reference_state=reference.tolist(),
                    effectiveness=env.damage_manager.state.mu.tolist(),
                    expected_effectiveness=mu(name, duration).tolist(),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"repo": str(args.repo.resolve()), "comparisons": result},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    for row in result:
        print({k: v for k, v in row.items() if isinstance(v, (str, float))})


if __name__ == "__main__":
    main()
