"""iADP versus a PID tuned ONLY on the healthy nonlinear F-16.

The fault is identical to example_iadp_small_fault_f16: a 15% attenuation of the
collective stabilator's total servo command at 20 s. PID gains stay fixed, while
its integral and measurement feedback continue to evolve. No fault information
is provided to either controller.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gymnasium.spaces import Box
from scipy.optimize import minimize

from tensoraerospace.agent.pid import PID

from . import example_iadp_small_fault_f16 as baseline

# Healthy-only bounded fit; reproduce with --retune-pid (see validation report).
DEFAULT_PID_GAINS = (-183.55882312255446, -499.9999999999999, -0.0013077268644681695)
TRACE_COLUMNS = baseline.COLUMNS[:8]
PID_SEARCH_BOUNDS = ((0.5, 300.0), (0.5, 500.0), (0.001, 50.0))


def reference_signal(cfg, *, training=False):
    t = np.arange(cfg.steps + 1) * cfg.dt
    if training:
        # Separate, healthy tuning manoeuvre; evaluation frequencies are held out.
        return np.deg2rad(
            0.5 * np.sin(2 * np.pi * 0.10 * t) + 0.15 * np.sin(2 * np.pi * 0.23 * t)
        )[None, :]
    return np.deg2rad(
        0.5 * np.sin(2 * np.pi * 0.12 * t)
        + 0.15 * np.sin(2 * np.pi * 0.31 * t + cfg.phase)
    )[None, :]


def rollout_pid(
    cfg,
    initial_state,
    reference,
    gains,
    *,
    fault,
    damage_profile=None,
    telemetry=None,
    environment_factory=None,
):
    """Use the library PID and the same magnitude/increment limits as iADP."""
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (3,) or not np.isfinite(gains).all():
        raise ValueError("PID gains must be three finite numbers")
    factory = environment_factory or baseline.make_environment
    env = factory(
        cfg, initial_state, reference, fault=fault, damage_profile=damage_profile
    )
    observation, _ = env.reset()
    limits = SimpleNamespace(action_space=Box(-10.0, 10.0, (1,), np.float64))
    pid = PID(limits, *gains, dt=cfg.dt)
    state, applied_delta = initial_state.copy(), 0.0
    rows, saturation_steps = [], 0
    try:
        for k in range(cfg.steps):
            # Match IADPAgent's increment limit relative to previous measured
            # input. Exposing this as the PID's saturation also enables its
            # conditional-integration anti-windup; no external hidden clipping.
            limits.action_space.low[:] = max(-10.0, applied_delta - 60 * cfg.dt)
            limits.action_space.high[:] = min(10.0, applied_delta + 60 * cfg.dt)
            command = pid.select_action(float(reference[0, k]), float(observation[1]))
            saturation_steps += int(abs(command) >= 10 - 1e-9)
            surface_mean = 0.0
            for _ in range(cfg.integration_substeps):
                old_surface = state[2]
                observation, _, terminated, truncated, _ = env.step(np.array([command]))
                state = env.model.current_state
                surface_mean += 0.5 * (old_surface + state[2])
            surface_mean /= cfg.integration_substeps
            applied_delta = float(np.rad2deg(surface_mean - initial_state[2]))
            if not np.isfinite(state).all() or not np.isfinite(pid.integral):
                raise FloatingPointError(f"Nonfinite PID/plant state at step {k}")
            if abs(state[1]) > np.deg2rad(10) or abs(
                state[0] - initial_state[0]
            ) > np.deg2rad(10):
                raise RuntimeError(
                    f"PID left the demonstration flight envelope at step {k}"
                )
            if (
                abs(state[2]) > env.model.param.maxabsstab + 1e-10
                or abs(state[3]) > env.model.param.maxabsdstab + 1e-10
            ):
                raise RuntimeError(f"Servo limits exceeded at step {k}")
            if (terminated or truncated) and k + 1 < cfg.steps:
                raise RuntimeError(f"Environment ended early at step {k}")
            rows.append(
                [
                    (k + 1) * cfg.dt,
                    np.rad2deg(reference[0, k + 1]),
                    np.rad2deg(state[1]),
                    np.rad2deg(state[0]),
                    np.rad2deg(state[2]),
                    np.rad2deg(state[3]),
                    command,
                    applied_delta,
                ]
            )
            if telemetry is not None:
                telemetry.append(rows[-1].copy())
        final_gains = [pid.kp, pid.ki, pid.kd]
        np.testing.assert_array_equal(final_gains, gains)
        return np.asarray(rows), {
            "events": env.damage_events_log,
            "gains_before": gains.tolist(),
            "gains_after": final_gains,
            "final_integral": pid.integral,
            "command_saturation_steps": saturation_steps,
        }
    finally:
        env.close()


def tune_pid(*, max_evaluations=100, gain_magnitude_bounds=None):
    """Fit gains on a fresh, fault-free 20 s manoeuvre; never evaluate a fault."""
    if max_evaluations < 4:
        raise ValueError("PID tuning needs at least four objective evaluations")
    bounds = (
        PID_SEARCH_BOUNDS if gain_magnitude_bounds is None else gain_magnitude_bounds
    )
    bounds_array = np.asarray(bounds, dtype=float)
    if (
        bounds_array.shape != (3, 2)
        or not np.isfinite(bounds_array).all()
        or np.any(bounds_array <= 0)
        or np.any(bounds_array[:, 0] >= bounds_array[:, 1])
    ):
        raise ValueError("PID gain bounds must be three ordered positive intervals")
    cfg = baseline.Experiment(duration=20.0, fault_time=10.0, loss=0.0)
    initial_state, agent, _ = baseline.trim_and_agent(cfg)
    reference = reference_signal(cfg, training=True)
    weight = float(agent.R[0, 0])
    history = []

    def objective(log_magnitudes):
        gains = -np.exp(log_magnitudes)  # negative pitch control effectiveness
        try:
            trace, diagnostics = rollout_pid(
                cfg, initial_state, reference, gains, fault=False
            )
            error_rad = np.deg2rad(trace[:, 2] - trace[:, 1])
            cost = float(np.mean(error_rad**2 + weight * trace[:, 7] ** 2))
            rmse = float(np.sqrt(np.mean((trace[:, 2] - trace[:, 1]) ** 2)))
            if diagnostics["events"]:
                raise AssertionError("A fault was used during PID tuning")
        except (FloatingPointError, RuntimeError):
            cost, rmse = 1.0, None
        history.append(
            {"gains": gains.tolist(), "healthy_cost": cost, "healthy_rmse_deg_s": rmse}
        )
        return cost

    result = minimize(
        objective,
        np.log([30.0, 30.0, 1.0]),
        method="Nelder-Mead",
        bounds=np.log(bounds_array),
        options={
            "maxfev": max_evaluations,
            "xatol": 0.05,
            "fatol": 1e-9,
            "initial_simplex": np.log(
                [
                    [30.0, 30.0, 1.0],
                    [60.0, 30.0, 1.0],
                    [30.0, 60.0, 1.0],
                    [30.0, 30.0, 3.0],
                ]
            ),
        },
    )
    if not any(item["healthy_rmse_deg_s"] is not None for item in history):
        raise RuntimeError("No valid PID found during healthy-only tuning")
    gains = tuple(float(value) for value in -np.exp(result.x))
    return gains, {
        "fault_used": False,
        "training_duration_s": cfg.duration,
        "training_frequencies_hz": [0.10, 0.23],
        "evaluation_frequencies_hz": [0.12, 0.31],
        "objective": "mean(error_rad_s**2 + R * mean_surface_delta_deg**2)",
        "R": weight,
        "gain_magnitude_bounds": bounds_array.tolist(),
        "optimizer": "bounded Nelder-Mead in log gain magnitudes",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": int(result.nfev),
        "best_healthy_cost": float(result.fun),
        "gains": gains,
        "history": history,
    }


def run_comparison(cfg, gains=DEFAULT_PID_GAINS):
    initial_state, agent, trim = baseline.trim_and_agent(cfg)
    reference = reference_signal(cfg)
    traces, metrics = {}, {}
    for fault in (False, True):
        condition = "fault" if fault else "healthy"
        trace, diagnostics = baseline.rollout(
            cfg, initial_state, agent, reference, fault=fault, adaptive=True
        )
        traces[f"iadp_{condition}"] = trace[:, :8]
        metrics[f"iadp_{condition}"] = {
            **baseline.tracking_metrics(trace, cfg),
            **diagnostics,
        }
        trace, diagnostics = rollout_pid(
            cfg, initial_state, reference, gains, fault=fault
        )
        traces[f"pid_{condition}"] = trace
        metrics[f"pid_{condition}"] = {
            **baseline.tracking_metrics(trace, cfg),
            **diagnostics,
        }
    for algorithm in ("iadp", "pid"):
        np.testing.assert_allclose(
            traces[f"{algorithm}_fault"][: cfg.fault_step],
            traces[f"{algorithm}_healthy"][: cfg.fault_step],
            rtol=0,
            atol=1e-10,
        )
    return traces, {
        "experiment": asdict(cfg),
        "trim": trim,
        "iadp_config": agent.get_param_env(),
        "pid_gains": list(gains),
        "pid_gain_units": {"kp": "deg/(rad/s)", "ki": "deg/rad", "kd": "deg/(rad/s^2)"},
        "pid_fixed_gains": True,
        "metrics": metrics,
        "trace_columns": list(TRACE_COLUMNS),
        "controller_limits": {
            "magnitude_deg": 10.0,
            "increment_rate_deg_s": 60.0,
            "increment_baseline": "previous mean measured surface minus trim",
        },
        "fault_semantics": "(1-loss) * total servo command, including trim",
        "prefault_match_with_healthy_for_each_algorithm": True,
    }


def plot_comparison(traces, report):
    import matplotlib.pyplot as plt

    cfg = report["experiment"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    t = traces["iadp_fault"][:, 0]
    reference = traces["iadp_fault"][:, 1]
    for ax in axes[0]:
        ax.plot(t, reference, "k:", lw=1.4, label="Command")
    for algorithm, color, label in [
        ("iadp", "#0077bb", "iADP, learning online"),
        ("pid", "#cc3311", "PID, gains fixed"),
    ]:
        trace = traces[f"{algorithm}_fault"]
        values = [
            trace[:, 2],
            trace[:, 2],
            trace[:, 2] - trace[:, 1],
            trace[:, 4],
            trace[:, 6],
            trace[:, 3],
        ]
        for ax, data in zip(axes.flat, values):
            ax.plot(t, data, color=color, lw=1.3, label=label)
    descriptions = [
        ("Pitch-rate tracking", "q [deg/s]"),
        ("Fault transient", "q [deg/s]"),
        ("Tracking error", "q - command [deg/s]"),
        ("Actual stabilator position", "Surface [deg]"),
        ("Controller output (before trim and fault)", "Command delta [deg]"),
        ("Angle of attack", "Alpha [deg]"),
    ]
    for ax, (title, ylabel) in zip(axes.flat, descriptions):
        ax.axvline(cfg["fault_time"], color="#555555", ls=":", lw=1)
        ax.set(title=title, ylabel=ylabel, xlabel="Time [s]", xlim=(0, cfg["duration"]))
        ax.grid(alpha=0.2)
    axes[0, 1].set_xlim(
        max(0, cfg["fault_time"] - 2), min(cfg["duration"], cfg["fault_time"] + 6)
    )
    axes[0, 0].legend(fontsize=9)
    iadp = report["metrics"]["iadp_fault"]["after_fault"]["rmse_deg_s"]
    pid = report["metrics"]["pid_fault"]["after_fault"]["rmse_deg_s"]
    fig.suptitle(
        f"F-16: iADP vs healthy-tuned PID | {100*cfg['loss']:.0f}% command-gain loss at {cfg['fault_time']:g} s\n"
        f"Post-fault RMSE: iADP {iadp:.4f} deg/s; PID {pid:.4f} deg/s",
        fontsize=14,
    )
    return fig


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
            header=",".join(TRACE_COLUMNS),
            comments="",
        )
    fig = plot_comparison(traces, report)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/f16-iadp-vs-pid"))
    parser.add_argument("--loss", type=float, default=0.15)
    parser.add_argument("--fault-time", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--retune-pid", action="store_true", help="repeat healthy-only nominal tuning"
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        cfg = baseline.Experiment(
            duration=args.duration,
            fault_time=args.fault_time,
            loss=args.loss,
            phase=args.phase,
            integration_substeps=args.substeps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains, tuning = DEFAULT_PID_GAINS, None
    if args.retune_pid:
        print("Tuning PID on the healthy aircraft only...", flush=True)
        gains, tuning = tune_pid()
    traces, report = run_comparison(cfg, gains)
    report["pid_tuning"] = (
        tuning
        if tuning is not None
        else {"source": "checked-in healthy-only fitted gains", "fault_used": False}
    )
    fig = save_results(traces, report, args.output)
    print("PID gains:", gains)
    for name, metrics in report["metrics"].items():
        print(
            f"{name:15s} RMSE after event: {metrics['after_fault']['rmse_deg_s']:.6f} deg/s; "
            f"late: {metrics['late']['rmse_deg_s']:.6f} deg/s"
        )
    print(f"Saved to {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
