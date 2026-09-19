"""Paired online adaptive control of the native nonlinear F-16 with its servo.

Uses trapezoidal mean surface position over each transition as actuator feedback.
This approximates the effective input; it is not a zero-order-held surface model.
The fault attenuates the commanded delta from trim, not the aerodynamic tables.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=["iadp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--train-duration", type=float, default=20.0)
    parser.add_argument("--eval-duration", type=float, default=30.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.linalg import solve_discrete_are
    from scipy.optimize import root

    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
        f16_ode_long,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
        default_parameters,
    )
    from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
    from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16

    dt = args.dt
    params = default_parameters()

    def trim_residual(z):
        return f16_ode_long(np.array([z[0], 0, z[1], 0]), np.array([z[1]]), 0, params)[
            :2
        ]

    trim = root(trim_residual, np.deg2rad([2, -2]))
    assert trim.success and np.max(abs(trim.fun)) < 1e-10
    alpha_trim, stab_trim = trim.x
    x_trim = np.array([alpha_trim, 0, stab_trim, 0])
    epsilon = 1e-5

    def derivative(index):
        plus, minus = x_trim.copy(), x_trim.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        return (
            f16_ode_long(plus, [stab_trim], 0, params)[1]
            - f16_ode_long(minus, [stab_trim], 0, params)[1]
        ) / (2 * epsilon)

    g = derivative(2) * np.pi / 180
    factor = float(np.random.default_rng(args.seed).uniform(0.8, 1.2))
    F = np.diag([1 + dt * derivative(1), 1.0])
    G = np.array([[dt * g * factor], [0.0]])
    R = np.array([[(abs(g) / 20) ** 2]])
    P = solve_discrete_are(
        np.sqrt(0.99) * F,
        np.sqrt(0.99) * G,
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        R,
    )
    agent = IADPAgent(
        1,
        1,
        IADPConfig(
            dt=dt,
            F_init=F,
            G_init=G,
            P_init=P,
            Q=np.eye(1),
            R=R,
            gamma=0.99,
            gamma_rls=0.9995,
            phi_init=1e3,
            policy_eval_window=300,
            policy_eval_every=20,
            policy_eval_warmup_updates=40,
            u_magnitude_limit=10,
            u_rate_limit=60,
            seed=args.seed,
        ),
    )
    feedback = "applied_action" in inspect.signature(agent.learn).parameters

    def rollout(controller, fault, adapting, duration, phase):
        controller.reset()
        n = round(duration / dt)
        time = np.arange(n + 1) * dt
        reference = np.deg2rad(
            0.5 * np.sin(2 * np.pi * 0.12 * time)
            + 0.15 * np.sin(2 * np.pi * 0.31 * time + phase)
        )[None, :]
        env = NonlinearLongitudinalF16(
            initial_state=x_trim.copy(),
            reference_signal=np.full((1, n + 1), alpha_trim),
            number_time_steps=n + 1,
            state_space=["alpha", "wz", "stab", "dstab"],
            control_space=["stab"],
            tracking_states=["alpha"],
            use_reward=False,
            dt=dt,
            integrator="rk4",
            control_bias=float(np.rad2deg(stab_trim)),
        )
        observation, _ = env.reset()
        x = x_trim.copy()
        covariance_name = "Phi" if args.agent == "iadp" else "P"
        theta_fixed = controller.rls.theta.copy()
        covariance_fixed = getattr(controller.rls, covariance_name).copy()
        value_fixed = controller.P.copy() if args.agent == "iadp" else None
        errors, trace = [], []
        covmin, covmax, sign_errors = np.inf, 0.0, 0
        max_state = abs(x)
        numerical_failure = None
        failed = False
        max_observation_error = 0.0
        failure_reason = None
        for k in range(n):
            try:
                u = controller.predict(
                    np.asarray(observation[1:2], dtype=float), reference, k
                )
                gain = 0.5 if fault and k * dt >= 15 else 1.0
                previous_surface = x[2]
                observation, _, terminated, truncated, _ = env.step(gain * u)
                x = env.model.current_state.copy()
                max_observation_error = max(
                    max_observation_error, float(np.max(np.abs(observation - x)))
                )
                np.testing.assert_allclose(observation, x, rtol=1e-6, atol=1e-8)
                mean_surface = np.rad2deg(0.5 * (previous_surface + x[2]) - stab_trim)
                kwargs = (
                    {"applied_action": np.array([mean_surface])} if feedback else {}
                )
                controller.learn(
                    np.asarray(observation[1:2], dtype=float), reference, k, **kwargs
                )
                if not adapting:
                    controller.rls.theta[:] = theta_fixed
                    setattr(controller.rls, covariance_name, covariance_fixed.copy())
                    if value_fixed is not None:
                        controller.P[:] = value_fixed
                cov = getattr(controller.rls, covariance_name)
                matrices = [x, u, cov, controller.rls.theta]
                if value_fixed is not None:
                    matrices.append(controller.P)
                if not all(np.isfinite(v).all() for v in matrices):
                    raise FloatingPointError("nonfinite state or adaptive matrices")
                eig = np.linalg.eigvalsh(cov)
                covmin, covmax = min(covmin, float(eig.min())), max(
                    covmax, float(eig.max())
                )
                G = float(
                    controller.G[0, 0]
                    if args.agent == "iadp"
                    else controller.rls.G[0, 0]
                )
                sign_errors += int(G * g <= 0)
                errors.append(float(np.rad2deg(x[1] - reference[0, k + 1])))
                max_state = np.maximum(max_state, abs(x))
                if k % max(1, n // 300) == 0:
                    trace.append(
                        [
                            float(time[k + 1]),
                            float(np.rad2deg(reference[0, k + 1])),
                            float(np.rad2deg(x[1])),
                            float(np.rad2deg(x[0])),
                            float(u[0]),
                            float(mean_surface),
                            G,
                            gain,
                        ]
                    )
                if abs(x[1]) > np.deg2rad(10) or abs(x[0] - alpha_trim) > np.deg2rad(
                    10
                ):
                    failed = True
                    failure_reason = (
                        "pitch_rate_limit"
                        if abs(x[1]) > np.deg2rad(10)
                        else "angle_of_attack_limit"
                    )
                    break
                if terminated or truncated:
                    if k + 1 < n:
                        failed = True
                        failure_reason = "early_environment_end"
                    break
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                numerical_failure = str(exc)
                failure_reason = "numerical_failure"
                failed = True
                break
        return dict(
            fault=fault,
            adaptation=adapting,
            duration_s=duration,
            completed_s=len(errors) * dt,
            rmse_deg_s=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            failed=failed,
            max_observation_error=max_observation_error,
            failure_reason=failure_reason,
            numerical_failure=numerical_failure,
            max_abs_state=max_state.tolist(),
            positive_G_steps=sign_errors,
            covariance_min_eigenvalue=covmin if np.isfinite(covmin) else None,
            covariance_max_eigenvalue=covmax,
            trace=trace,
            trace_columns=[
                "time_s",
                "reference_deg_s",
                "q_deg_s",
                "alpha_deg",
                "command_deg",
                "mean_surface_delta_deg",
                "identified_G",
                "command_gain",
            ],
        )

    trained = rollout(agent, False, True, args.train_duration, 0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = agent.save(args.output.with_suffix(""))
    evaluations = [
        rollout(copy.deepcopy(agent), fault, adapting, args.eval_duration, 0.7)
        for fault in [False, True]
        for adapting in [False, True]
    ]
    result = dict(
        repo=str(args.repo.resolve()),
        agent=args.agent,
        plant="f16",
        seed=args.seed,
        dt=dt,
        trim_state=x_trim.tolist(),
        trim_residual=trim.fun.tolist(),
        physical_G=g,
        warm_start_factor=factor,
        feedback_supported=feedback,
        observation_source="env",
        config=agent.get_param_env(),
        checkpoint=checkpoint,
        training=trained,
        evaluations=evaluations,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in e.items() if k not in ["trace", "trace_columns"]}
                for e in [trained, *evaluations]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
