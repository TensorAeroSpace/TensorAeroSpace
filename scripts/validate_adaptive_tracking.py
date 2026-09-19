"""iADP validation on native B747 and LAPAN environments.

Separate online training from held-out frozen/adapting evaluations. Actual
actuator feedback is supplied to the controller. AA-INDI validation, requiring
independent navigation, is in validate_paper_adaptive.py.
Dynamics and actuator limits are untouched except for an explicitly simulated
50% reduction of B in the effectiveness-loss scenario.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path


def observation_in_si(observation, plant, full_state):
    """Adapt public env units to the controller's radian state convention."""
    import numpy as np

    state = np.asarray(observation, dtype=float).copy()
    if plant == "b747":
        # B747 exposes angular observations in degrees; LAPAN exposes radians.
        indices = [2, 3] if full_state else [0, 1]
        state[indices] = np.deg2rad(state[indices])
    return state


def tracking_reference(time, phase, full_state, mode="sampled"):
    """Return desired output or full state of the two-sine reference generator.

    The oscillator state is [r1, dr1, r1+r2, dr2]; only its third component
    enters the tracking cost. This supplies additional reference information.
    """
    import numpy as np

    w1, w2 = 2 * np.pi * np.array([0.12, 0.31])
    r1 = np.deg2rad(0.5) * np.sin(w1 * time)
    reference = np.deg2rad(0.5 * np.sin(w1 * time) + 0.15 * np.sin(w2 * time + phase))[
        None, :
    ]
    if mode == "constant":
        reference[:] = np.deg2rad(0.05)
    if not full_state:
        return reference
    result = np.zeros((4, len(time)))
    result[2] = reference[0]
    if mode == "oscillator":
        result[0] = r1
        result[1] = np.deg2rad(0.5) * w1 * np.cos(w1 * time)
        result[3] = np.deg2rad(0.15) * w2 * np.cos(w2 * time + phase)
    return result


def reference_transition(dt):
    """Exact autonomous transition of the augmented two-sine generator."""
    import numpy as np
    from scipy.linalg import expm

    w1, w2 = 2 * np.pi * np.array([0.12, 0.31])
    continuous = np.array(
        [[0, 1, 0, 0], [-(w1**2), 0, 0, 0], [0, 1, 0, 1], [w2**2, 0, -(w2**2), 0]]
    )
    return expm(dt * continuous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=["iadp"], required=True)
    parser.add_argument("--plant", choices=["b747", "lapan"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--train-duration", type=float, default=30.0)
    parser.add_argument("--eval-duration", type=float, default=60.0)
    parser.add_argument(
        "--full-state",
        action="store_true",
        help="Give iADP all four native states; track q with the same cost.",
    )
    parser.add_argument(
        "--freeze-critic",
        action="store_true",
        help="Keep the initial iADP value matrix for diagnosis.",
    )
    parser.add_argument(
        "--freeze-model",
        action="store_true",
        help="Freeze identifier parameters during the diagnostic run.",
    )
    parser.add_argument("--iadp-gamma", type=float, default=0.99)
    parser.add_argument(
        "--reference-mode",
        choices=["sampled", "oscillator", "constant"],
        default="sampled",
    )
    parser.add_argument("--iadp-window", type=int, default=300)
    parser.add_argument("--iadp-warmup", type=int, default=40)
    parser.add_argument(
        "--train-excitation-deg",
        type=float,
        default=0.0,
        help="Additional sinusoidal command during training only, before env actuator limits.",
    )
    parser.add_argument(
        "--critic-min-samples",
        type=int,
        default=0,
        help="Wait for this many new critic samples after each reset.",
    )
    args = parser.parse_args()
    if args.reference_mode == "oscillator" and not args.full_state:
        parser.error("oscillator reference requires --full-state")
    if (args.full_state or args.freeze_critic) and args.agent != "iadp":
        parser.error("--full-state and --freeze-critic apply to iADP")
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.linalg import solve_discrete_are
    from scipy.signal import cont2discrete

    from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
    from tensoraerospace.envs.b747 import LinearLongitudinalB747
    from tensoraerospace.envs.lapan import LinearLongitudinalLAPAN

    Env = LinearLongitudinalB747 if args.plant == "b747" else LinearLongitudinalLAPAN
    probe = Env(np.zeros(4), np.zeros((1, 5)), 5, dt=args.dt)
    a = float(probe.model.filt_A[2, 2])
    b = float(probe.model.filt_B[2, 0] * np.pi / 180)
    g = b / args.dt
    rng = np.random.default_rng(args.seed)
    warm_start_factor = float(rng.uniform(0.8, 1.2))
    state_count = 4 if args.full_state else 1
    q_index = 2 if args.full_state else 0
    Q = np.zeros((state_count, state_count))
    Q[q_index, q_index] = 1.0
    if args.full_state:
        F = np.zeros((8, 8))
        F[:4, :4] = probe.model.filt_A
        F[4:, 4:] = (
            reference_transition(args.dt)
            if args.reference_mode == "oscillator"
            else np.eye(4)
        )
        G = np.vstack(
            [probe.model.filt_B * np.pi / 180 * warm_start_factor, np.zeros((4, 1))]
        )
    else:
        F = np.diag([a, 1.0])
        G = np.array([[b * warm_start_factor], [0.0]])
    gamma = args.iadp_gamma
    R = np.array([[(abs(g) / 20) ** 2]])
    P = solve_discrete_are(
        np.sqrt(gamma) * F,
        np.sqrt(gamma) * G,
        np.block([[Q, -Q], [-Q, Q]]),
        R,
    )
    agent = IADPAgent(
        state_count,
        1,
        IADPConfig(
            dt=args.dt,
            Q=Q,
            R=R,
            gamma=gamma,
            gamma_rls=0.9995,
            phi_init=1e3,
            policy_eval_window=args.iadp_window,
            policy_eval_every=20,
            policy_training_start_step=sys.maxsize if args.freeze_critic else 0,
            policy_eval_warmup_updates=args.iadp_warmup,
            F_init=F,
            G_init=G,
            P_init=P,
            u_magnitude_limit=10,
            u_rate_limit=60,
            seed=args.seed,
            **(
                {"policy_eval_min_samples": args.critic_min_samples}
                if args.critic_min_samples
                else {}
            ),
        ),
    )
    feedback_supported = "applied_action" in inspect.signature(agent.learn).parameters

    def rollout(controller, scenario, adaptation, duration, phase, *, training=False):
        controller.reset()
        steps = round(duration / args.dt)
        time = np.arange(steps + 1) * args.dt
        reference = tracking_reference(
            time, phase, args.full_state, args.reference_mode
        )
        initial = np.zeros(4)
        initial[2] = np.deg2rad(0.02 * np.cos(phase))
        env = Env(
            initial,
            np.zeros((1, steps + 1)),
            steps + 1,
            dt=args.dt,
            tracking_states=["q"],
            output_space=(
                ["u", "w", "q", "theta"] if args.full_state else ["q", "theta"]
            ),
        )
        observation, _ = env.reset()
        observation = observation_in_si(observation, args.plant, args.full_state)
        nominal_B = env.model.B.copy()
        noise_rng = np.random.default_rng(20260917 + args.seed)
        sensor_noise = (
            noise_rng.normal(0, np.deg2rad(0.005), steps + 1)
            if scenario == "noise"
            else np.zeros(steps + 1)
        )
        x = initial.copy()
        errors, trace, residuals = [], [], []
        physical_costs, input_squares = [], []
        peaks = np.abs(x)
        applied_prev = 0.0
        rate_peak, command_rate_peak = 0.0, 0.0
        command_prev = 0.0
        failures = 0
        numerical_failure = None
        sign_errors, min_covariance, max_covariance = 0, np.inf, 0.0
        G_trace, P_trace = [], []
        critic_diagnostics = []
        max_observation_error = 0.0
        failure_reason = None
        covariance_name = "Phi" if args.agent == "iadp" else "P"
        theta_fixed = controller.rls.theta.copy()
        covariance_fixed = getattr(controller.rls, covariance_name).copy()
        value_fixed = controller.P.copy() if args.agent == "iadp" else None
        original_update = controller.rls.update
        if args.freeze_model:

            def update_fixed_model(*update_args, **update_kwargs):
                residual = original_update(*update_args, **update_kwargs)
                controller.rls.theta[:] = theta_fixed
                setattr(controller.rls, covariance_name, covariance_fixed.copy())
                return residual

            controller.rls.update = update_fixed_model
        previous_gain = 1.0
        for k in range(steps):
            t = k * args.dt
            gain = 0.5 if scenario == "loss" and t >= 15 else 1.0
            if gain != previous_gain:
                env.model.B = gain * nominal_B
                env.model.filt_B = cont2discrete(
                    (env.model.A, env.model.B, env.model.C, env.model.D), args.dt
                )[1]
                previous_gain = gain
            noise = sensor_noise[k]
            measurement = (
                np.asarray(observation, dtype=float).copy()
                if args.full_state
                else np.asarray(observation[:1], dtype=float).copy()
            )
            measurement[q_index] += noise
            try:
                physical_error_t = float(x[2] - reference[q_index, k])
                command = controller.predict(measurement, reference, k)
                # This is an additive command bias, not an aerodynamic gust model.
                bias = (
                    np.deg2rad(0.1) / abs(g) if scenario == "bias" and t >= 20 else 0.0
                )
                excitation = (
                    args.train_excitation_deg * np.sin(2 * np.pi * 0.7 * t)
                    if training
                    else 0.0
                )
                observation, _, terminated, truncated, info = env.step(
                    command + bias + excitation
                )
                observation = observation_in_si(
                    observation, args.plant, args.full_state
                )
                x = np.asarray(env.model.xt).reshape(-1).copy()
                actual = np.rad2deg(env.model.store_input[:, env.model.time_step - 1])
                if "applied_action" in info:
                    np.testing.assert_allclose(
                        info["applied_action"], actual, rtol=1e-6, atol=1e-7
                    )
                expected_observation = x if args.full_state else x[[2, 3]]
                max_observation_error = max(
                    max_observation_error,
                    float(np.max(np.abs(observation - expected_observation))),
                )
                np.testing.assert_allclose(
                    observation, expected_observation, rtol=1e-6, atol=1e-8
                )
                next_noise = sensor_noise[k + 1]
                kwargs = {"applied_action": actual} if feedback_supported else {}
                next_measurement = (
                    np.asarray(observation, dtype=float).copy()
                    if args.full_state
                    else np.asarray(observation[:1], dtype=float).copy()
                )
                next_measurement[q_index] += next_noise
                metrics = controller.learn(next_measurement, reference, k, **kwargs)
                if not adaptation:
                    controller.rls.theta[:] = theta_fixed
                    setattr(controller.rls, covariance_name, covariance_fixed.copy())
                    if value_fixed is not None:
                        controller.P[:] = value_fixed
                cov = getattr(controller.rls, covariance_name)
                eig = np.linalg.eigvalsh(cov)
                matrices = [controller.rls.theta, cov]
                if args.agent == "iadp":
                    matrices.append(controller.P)
                if not all(
                    np.isfinite(v).all() for v in [x, command, *matrices]
                ) or not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("nonfinite adaptive state or trajectory")
                min_covariance = min(min_covariance, float(eig.min()))
                max_covariance = max(max_covariance, float(eig.max()))
                G_now = float(
                    controller.G[q_index, 0]
                    if args.agent == "iadp"
                    else controller.rls.G[0, 0]
                )
                sign_errors += int(G_now >= 0)
                residuals.append(metrics["rls_pred_error_norm"])
                G_trace.append(G_now)
                P_trace.append(
                    float(np.linalg.norm(controller.P)) if args.agent == "iadp" else 0.0
                )
                applied = float(actual[0])
                input_squares.append(applied**2)
                physical_costs.append(
                    physical_error_t**2 + (abs(g) / 20) ** 2 * applied**2
                )
                rate_peak = max(rate_peak, abs(applied - applied_prev) / args.dt)
                command_rate_peak = max(
                    command_rate_peak, abs(float(command[0]) - command_prev) / args.dt
                )
                applied_prev, command_prev = applied, float(command[0])
                errors.append(float(np.rad2deg(x[2] - reference[q_index, k + 1])))
                peaks = np.maximum(peaks, np.abs(x))
                if k % max(1, steps // 300) == 0:
                    trace.append(
                        [
                            float(time[k + 1]),
                            float(np.rad2deg(reference[q_index, k + 1])),
                            float(np.rad2deg(x[2])),
                            float(np.rad2deg(x[3])),
                            float(command[0]),
                            applied,
                            G_now,
                            gain,
                        ]
                    )
                if (
                    args.agent == "iadp"
                    and controller._window
                    and k % max(1, round(1 / args.dt)) == 0
                ):
                    features = np.array(
                        [np.outer(s["X"], s["X"]).ravel() for s in controller._window]
                    )
                    singular = np.linalg.svd(features, compute_uv=False)
                    rank = int(np.linalg.matrix_rank(features))
                    p_eig = np.linalg.eigvalsh(controller.P)
                    critic_diagnostics.append(
                        dict(
                            time_s=float(time[k + 1]),
                            rank=rank,
                            max_singular=float(singular[0]),
                            min_nonzero_singular=(
                                float(singular[rank - 1]) if rank else 0.0
                            ),
                            P_min_eigenvalue=float(p_eig.min()),
                            P_max_eigenvalue=float(p_eig.max()),
                        )
                    )
                if abs(x[2]) > np.deg2rad(10) or abs(x[3]) > np.deg2rad(15):
                    failure_reason = (
                        "pitch_rate_limit"
                        if abs(x[2]) > np.deg2rad(10)
                        else "pitch_angle_limit"
                    )
                    failures = 1
                    break
                if terminated or truncated:
                    if k + 1 < steps:
                        failures = 1
                        failure_reason = "early_environment_end"
                    break
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                numerical_failure = str(exc)
                failure_reason = "numerical_failure"
                failures = 1
                break
        if args.freeze_model:
            del controller.rls.update
        finite_cov = np.isfinite(min_covariance)
        return dict(
            scenario=scenario,
            adaptation=adaptation,
            duration_s=duration,
            completed_s=len(errors) * args.dt,
            rmse_deg_s=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            rmse_before_fault_deg_s=(
                float(np.sqrt(np.mean(np.square(errors[: round(15 / args.dt)]))))
                if errors
                else None
            ),
            rmse_last_15s_deg_s=(
                float(np.sqrt(np.mean(np.square(errors[-round(15 / args.dt) :]))))
                if len(errors) >= round(15 / args.dt)
                else None
            ),
            mean_physical_cost=(
                float(np.mean(physical_costs)) if physical_costs else None
            ),
            rms_applied_deg=(
                float(np.sqrt(np.mean(input_squares))) if input_squares else None
            ),
            discounted_physical_cost=(
                float(
                    np.dot(
                        np.power(args.iadp_gamma, np.arange(len(physical_costs))),
                        physical_costs,
                    )
                )
                if physical_costs
                else None
            ),
            failed=bool(failures),
            failure_reason=failure_reason,
            max_observation_error=max_observation_error,
            critic_diagnostics=critic_diagnostics,
            numerical_failure=numerical_failure,
            max_abs_state=peaks.tolist(),
            max_applied_rate_deg_s=rate_peak,
            max_command_rate_deg_s=command_rate_peak,
            positive_G_steps=sign_errors,
            covariance_min_eigenvalue=min_covariance if finite_cov else None,
            covariance_max_eigenvalue=max_covariance,
            residual_rms=(
                float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None
            ),
            final_G=G_trace[-1] if G_trace else None,
            final_P_norm=P_trace[-1] if P_trace else None,
            trace_columns=[
                "time_s",
                "reference_deg_s",
                "q_deg_s",
                "theta_deg",
                "command_deg",
                "applied_deg",
                "identified_G",
                "effectiveness",
            ],
            trace=trace,
        )

    trained = rollout(
        agent, "nominal", True, args.train_duration, phase=0.2, training=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = agent.save(args.output.with_suffix(""))
    results = []
    for scenario in ["nominal", "loss", "bias", "noise"]:
        for adaptation in [False, True]:
            record = rollout(
                copy.deepcopy(agent),
                scenario,
                adaptation,
                args.eval_duration,
                phase=0.7,
            )
            results.append(record)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ["trace", "trace_columns"]
                    }
                ),
                flush=True,
            )
    result = dict(
        agent=args.agent,
        plant=args.plant,
        seed=args.seed,
        repo=str(args.repo.resolve()),
        dt=args.dt,
        warm_start_factor=warm_start_factor,
        initial_a=a,
        initial_b=b,
        feedback_supported=feedback_supported,
        full_state=args.full_state,
        freeze_critic=args.freeze_critic,
        freeze_model=args.freeze_model,
        reference_mode=args.reference_mode,
        train_excitation_deg=args.train_excitation_deg,
        critic_min_samples=args.critic_min_samples,
        observation_source="env",
        noise_alignment="same timestamp in learn and next predict",
        checkpoint=checkpoint,
        config=agent.get_param_env(),
        training=trained,
        evaluations=results,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
