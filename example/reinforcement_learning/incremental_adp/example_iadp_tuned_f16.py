"""Healthy-only iADP tuning and an explicitly integral-augmented F-16 example.

The library's iADP equations are unchanged. The integral variant supplies
x = [q, z], reference = [q_ref, 0], z_next = z + dt * (q_ref - q), and penalizes
both rate error and accumulated error. No external PID correction is added.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from tensoraerospace.agent.iadp import IADPAgent

from . import example_iadp_small_fault_f16 as baseline
from . import example_iadp_vs_pid_f16 as comparison


@dataclass(frozen=True)
class Tuning:
    r_scale: float = 0.5
    integral_weight: float = 30.0
    gamma: float = 0.99
    blend: float = 0.0001
    forgetting: float = 0.9995
    covariance: float = 20000.0
    window: int = 300
    every: int = 10
    min_samples: int = 300


INTEGRAL_TUNING = Tuning()
RATE_TUNING = Tuning(
    r_scale=0.10042337710735787,
    integral_weight=0.0,
    blend=0.0032139005151721003,
    covariance=20246.43637460468,
)


def make_agent(cfg, tuning):
    """Healthy initialization, with the known integral update in the model."""
    initial_state, nominal, trim = baseline.trim_and_agent(cfg)
    settings = copy.deepcopy(nominal.cfg)
    settings.R = nominal.R * tuning.r_scale
    settings.gamma = tuning.gamma
    settings.gamma_rls = tuning.forgetting
    settings.phi_init = tuning.covariance
    settings.policy_eval_window = tuning.window
    settings.policy_eval_min_samples = tuning.min_samples
    settings.policy_eval_warmup_updates = tuning.min_samples
    settings.policy_eval_every = tuning.every
    settings.policy_eval_blend = tuning.blend
    integral = tuning.integral_weight > 0
    n_state = 2 if integral else 1
    Q = np.diag([1.0, tuning.integral_weight]) if integral else np.eye(1)
    if integral:
        # Augmented order: [q, z, q_ref, 0]. Forward Euler matches the causal
        # z update used below. Integral dynamics are not aircraft dynamics.
        F, G = np.eye(4), np.zeros((4, 1))
        F[0, 0] = nominal.F[0, 0]
        F[1, 0], F[1, 2] = -cfg.dt, cfg.dt
        G[0, 0] = nominal.G[0, 0]
        settings.F_init, settings.G_init = F, G
    settings.Q = Q
    Q_aug = np.block([[Q, -Q], [-Q, Q]])
    settings.P_init = solve_discrete_are(
        np.sqrt(settings.gamma) * settings.F_init,
        np.sqrt(settings.gamma) * settings.G_init,
        Q_aug,
        settings.R,
    )
    return initial_state, IADPAgent(n_state, 1, settings), trim


def integrate_error(integral, measured_q, reference_q, dt):
    """Causal left-endpoint integration: uses no next/future reference sample."""
    return integral + dt * (reference_q - measured_q)


LEARNING_COLUMNS = (
    "integral_error_rad",
    "identified_F_qq",
    "identified_G_q",
    "critic_norm",
    "model_change_since_event",
    "critic_change_since_event",
    "covariance_min_eigenvalue",
)


def rollout(
    cfg,
    tuning,
    *,
    fault,
    training=False,
    frozen_after_event=False,
    telemetry=None,
    damage_profile=None,
    environment_factory=None,
):
    """Run the controller; optionally collect state/learning rows in a caller list.

    Telemetry contains TRACE_COLUMNS followed by LEARNING_COLUMNS. It is passive
    and keeps completed steps available to the caller if a runtime guard fails.
    """
    initial_state, agent, _ = make_agent(cfg, tuning)
    reference = comparison.reference_signal(cfg, training=training)
    use_integral = tuning.integral_weight > 0
    augmented_reference = (
        np.vstack([reference, np.zeros_like(reference)]) if use_integral else reference
    )
    factory = environment_factory or baseline.make_environment
    env = factory(
        cfg, initial_state, reference, fault=fault, damage_profile=damage_profile
    )
    observation, _ = env.reset()
    state, integral = initial_state.copy(), 0.0
    rows = []
    covariance_min = float("inf")
    event_model, event_critic = None, None
    try:
        for k in range(cfg.steps):
            if k == cfg.fault_step:
                event_model, event_critic = agent.rls.theta.copy(), agent.P.copy()
            measured_q = float(observation[1])
            features = (
                np.array([measured_q, integral])
                if use_integral
                else np.array([measured_q])
            )
            command = agent.predict(features, augmented_reference, k)
            surface_mean = 0.0
            for _ in range(cfg.integration_substeps):
                previous_surface = state[2]
                observation, _, terminated, truncated, _ = env.step(command)
                state = env.model.current_state
                surface_mean += 0.5 * (previous_surface + state[2])
            surface_mean /= cfg.integration_substeps
            applied = np.array([np.rad2deg(surface_mean - initial_state[2])])
            if use_integral:
                integral = integrate_error(
                    integral, measured_q, reference[0, k], cfg.dt
                )
            features_next = (
                np.array([float(observation[1]), integral])
                if use_integral
                else observation[1:2].astype(float)
            )
            frozen = frozen_after_event and k >= cfg.fault_step
            if frozen:
                theta, covariance, critic = (
                    agent.rls.theta.copy(),
                    agent.rls.Phi.copy(),
                    agent.P.copy(),
                )
            agent.learn(features_next, augmented_reference, k, applied_action=applied)
            if frozen:
                agent.rls.theta[:], agent.rls.Phi[:], agent.P[:] = (
                    theta,
                    covariance,
                    critic,
                )
            if not all(
                np.isfinite(x).all()
                for x in (
                    state,
                    command,
                    agent.rls.theta,
                    agent.rls.Phi,
                    agent.P,
                    features_next,
                )
            ):
                raise FloatingPointError(f"Nonfinite controller/plant state at {k}")
            eigenvalue = float(np.linalg.eigvalsh(agent.rls.Phi).min())
            covariance_min = min(covariance_min, eigenvalue)
            if eigenvalue <= 0 or agent.G[0, 0] >= 0:
                raise RuntimeError(f"Invalid identifier at {k}")
            if abs(state[1]) > np.deg2rad(10) or abs(
                state[0] - initial_state[0]
            ) > np.deg2rad(10):
                raise RuntimeError(f"Left the demonstration flight envelope at {k}")
            if (
                abs(state[2]) > env.model.param.maxabsstab + 1e-10
                or abs(state[3]) > env.model.param.maxabsdstab + 1e-10
            ):
                raise RuntimeError(f"Servo limits exceeded at {k}")
            if (terminated or truncated) and k + 1 < cfg.steps:
                raise RuntimeError(f"Environment ended early at {k}")
            rows.append(
                [
                    (k + 1) * cfg.dt,
                    np.rad2deg(reference[0, k + 1]),
                    np.rad2deg(state[1]),
                    np.rad2deg(state[0]),
                    np.rad2deg(state[2]),
                    np.rad2deg(state[3]),
                    float(command[0]),
                    float(applied[0]),
                ]
            )
            if telemetry is not None:
                telemetry.append(
                    rows[-1]
                    + [
                        float(integral),
                        float(agent.F[0, 0]),
                        float(agent.G[0, 0]),
                        float(np.linalg.norm(agent.P)),
                        (
                            0.0
                            if event_model is None
                            else float(np.linalg.norm(agent.rls.theta - event_model))
                        ),
                        (
                            0.0
                            if event_critic is None
                            else float(np.linalg.norm(agent.P - event_critic))
                        ),
                        eigenvalue,
                    ]
                )
        return np.asarray(rows), {
            "events": copy.deepcopy(env.damage_events_log),
            "minimum_covariance_eigenvalue": covariance_min,
            "final_integral_error_rad": float(integral),
            "model_change_after_event": float(
                np.linalg.norm(agent.rls.theta - event_model)
            ),
            "critic_change_after_event": float(np.linalg.norm(agent.P - event_critic)),
            "initial_configuration": agent.get_param_env(),
        }
    finally:
        env.close()


def recovery_time(trace, cfg, band=0.05, minimum_remaining_s=5.0):
    """First post-event entry into a band maintained to the end of the episode.

    At least five seconds must remain to avoid a spurious last-sample recovery.
    None means this criterion was not met, not an infinite/estimated time.
    """
    errors = np.abs(trace[:, 2] - trace[:, 1])
    future_max = np.maximum.accumulate(errors[::-1])[::-1]
    eligible = np.flatnonzero(
        (trace[:, 0] > cfg.fault_time)
        & (trace[:, 0] <= cfg.duration - minimum_remaining_s)
        & (future_max <= band)
    )
    return float(trace[eligible[0], 0] - cfg.fault_time) if eligible.size else None


def metrics(trace, cfg):
    result = baseline.tracking_metrics(trace, cfg)
    result["recovery_time_s"] = {
        str(band): recovery_time(trace, cfg, band) for band in (0.1, 0.05, 0.02, 0.01)
    }
    return result


def nominal_search():
    """Reproduce the final 27-candidate search on a healthy 60 s manoeuvre."""
    cfg = baseline.Experiment(duration=60.0, fault_time=20.0, loss=0.0)
    _, nominal, _ = baseline.trim_and_agent(cfg)
    # Selection uses the SAME external error/control cost for all candidates,
    # independent of their internal integral weights and R hyperparameters.
    physical_R = float(nominal.R[0, 0])
    history = []
    for r_scale in (0.5, 1.0, 2.0):
        for weight in (10.0, 30.0, 60.0):
            for blend in (0.0001, 0.0003, 0.001):
                tuning = Tuning(r_scale=r_scale, integral_weight=weight, blend=blend)
                try:
                    trace, diagnostics = rollout(
                        cfg, tuning, fault=False, training=True
                    )
                    if diagnostics["events"]:
                        raise AssertionError("Fault data leaked into nominal search")
                    error = np.deg2rad(trace[:, 2] - trace[:, 1])
                    cost = float(np.mean(error**2 + physical_R * trace[:, 7] ** 2))
                    record = {
                        "tuning": asdict(tuning),
                        "cost": cost,
                        "failed": False,
                        "rmse_deg_s": float(np.sqrt(np.mean(np.rad2deg(error) ** 2))),
                    }
                except (
                    ValueError,
                    RuntimeError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as exc:
                    record = {
                        "tuning": asdict(tuning),
                        "cost": None,
                        "failed": True,
                        "reason": str(exc),
                    }
                history.append(record)
                print(
                    f"Nominal candidate {len(history)}/27: {record['cost']}", flush=True
                )
    valid = [record for record in history if not record["failed"]]
    if not valid:
        raise RuntimeError("No valid tuning found")
    winner = min(valid, key=lambda record: record["cost"])
    return Tuning(**winner["tuning"]), {
        "fault_used": False,
        "duration_s": cfg.duration,
        "reference_frequencies_hz": [0.10, 0.23],
        "external_R": physical_R,
        "history": history,
        "selected": winner,
    }


def run_comparison(cfg, tuning=INTEGRAL_TUNING, *, fault=True):
    traces, results = {}, {}
    initial_state, nominal, trim = baseline.trim_and_agent(cfg)
    reference = comparison.reference_signal(cfg)
    trace, diagnostics = baseline.rollout(
        cfg, initial_state, nominal, reference, fault=fault, adaptive=True
    )
    traces["iadp_original"] = trace[:, :8]
    results["iadp_original"] = {**metrics(trace, cfg), **diagnostics}
    trace, diagnostics = comparison.rollout_pid(
        cfg, initial_state, reference, comparison.DEFAULT_PID_GAINS, fault=fault
    )
    traces["pid"] = trace
    results["pid"] = {**metrics(trace, cfg), **diagnostics}
    for name, selected, frozen in (
        ("iadp_rate_tuned", RATE_TUNING, False),
        ("iadp_integral", tuning, False),
        ("iadp_integral_frozen", tuning, True),
    ):
        trace, diagnostics = rollout(
            cfg, selected, fault=fault, frozen_after_event=frozen
        )
        traces[name] = trace
        results[name] = {**metrics(trace, cfg), **diagnostics}
    np.testing.assert_allclose(
        traces["iadp_integral"][: cfg.fault_step],
        traces["iadp_integral_frozen"][: cfg.fault_step],
        rtol=0,
        atol=1e-10,
    )
    return traces, {
        "experiment": asdict(cfg),
        "fault_active": fault,
        "trim": trim,
        "tuning": asdict(tuning),
        "rate_tuning": asdict(RATE_TUNING),
        "pid_gains": list(comparison.DEFAULT_PID_GAINS),
        "metrics": results,
        "integral_definition": "z[k+1] = z[k] + dt * (q_ref[k] - q_measured[k]); x=[q,z], ref=[q_ref,0]",
        "recovery_definition": "error stays inside stated deg/s band to end, with at least 5 s remaining",
        "trace_columns": list(comparison.TRACE_COLUMNS),
    }


def save_results(traces, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, trace in traces.items():
        np.savetxt(
            output / f"{name}.csv",
            trace,
            delimiter=",",
            header=",".join(comparison.TRACE_COLUMNS),
            comments="",
        )
    figure = comparison.plot_comparison(
        {"iadp_fault": traces["iadp_integral"], "pid_fault": traces["pid"]},
        {
            "experiment": report["experiment"],
            "metrics": {
                "iadp_fault": report["metrics"]["iadp_integral"],
                "pid_fault": report["metrics"]["pid"],
            },
        },
    )
    for ax in figure.axes:
        for line in ax.lines:
            if line.get_label() == "iADP, learning online":
                line.set_label("iADP + integral state")
    figure.axes[0].legend(fontsize=9)
    m = report["metrics"]
    cfg = report["experiment"]
    figure.suptitle(
        f"F-16: iADP + integral state vs PID | {100*cfg['loss']:.0f}% fault at {cfg['fault_time']:g} s\n"
        f"Post-event RMSE: iADP-I {m['iadp_integral']['after_fault']['rmse_deg_s']:.4f}; "
        f"PID {m['pid']['after_fault']['rmse_deg_s']:.4f} deg/s",
        fontsize=14,
    )
    figure.savefig(output / "comparison.png", dpi=160)
    figure.savefig(output / "comparison.svg")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/f16-iadp-tuned"))
    parser.add_argument("--loss", type=float, default=0.15)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--search", action="store_true", help="repeat the 27 healthy-only tuning trials"
    )
    args = parser.parse_args()
    try:
        cfg = baseline.Experiment(
            duration=args.duration,
            loss=args.loss,
            phase=args.phase,
            integration_substeps=args.substeps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tuning, search_report = INTEGRAL_TUNING, None
    if args.search:
        tuning, search_report = nominal_search()
    traces, report = run_comparison(cfg, tuning)
    report["nominal_search"] = search_report
    figure = save_results(traces, report, args.output)
    plt.close(figure)
    for name, m in report["metrics"].items():
        print(
            f"{name:22s} post-event RMSE {m['after_fault']['rmse_deg_s']:.6f}; "
            f"late {m['late']['rmse_deg_s']:.6f} deg/s; recovery ±0.05: {m['recovery_time_s']['0.05']}"
        )
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
