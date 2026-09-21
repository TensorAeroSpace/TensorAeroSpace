"""Evaluate the fixed F-16 controllers for 500 s, including learning telemetry.

No parameter search is performed. Each scenario starts from the healthy trim;
PID gains stay fixed and the frozen iADP keeps its integral feedback active.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import example_iadp_tuned_f16 as tuned

PROFILES = ("iadp_integral", "pid", "iadp_integral_frozen")
LABELS = ("iADP + integral, online", "PID", "iADP + integral, frozen at 20 s")
COLORS = ("tab:blue", "tab:orange", "tab:green")


def window_metrics(trace, start, stop):
    """Use the same (start, stop] convention as the short-horizon examples."""
    selected = trace[(trace[:, 0] > start) & (trace[:, 0] <= stop)]
    if not len(selected):
        return None
    error = selected[:, 2] - selected[:, 1]
    peak = int(np.argmax(np.abs(error)))
    return {
        "start_s": float(start),
        "stop_s": float(stop),
        "samples": len(selected),
        "rmse_deg_s": float(np.sqrt(np.mean(error**2))),
        "peak_error_deg_s": float(abs(error[peak])),
        "peak_time_s": float(selected[peak, 0]),
    }


def rolling_rmse(trace, window_steps):
    """Trailing full windows, without smoothing over future samples."""
    squared = (trace[:, 2] - trace[:, 1]) ** 2
    total = np.concatenate(([0.0], np.cumsum(squared)))
    return trace[window_steps - 1 :, 0], np.sqrt(
        np.maximum((total[window_steps:] - total[:-window_steps]) / window_steps, 0)
    )


def run_experiment(cfg, output, conditions=("fault", "healthy")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": asdict(cfg),
        "tuning": asdict(tuned.INTEGRAL_TUNING),
        "pid_gains": list(tuned.comparison.DEFAULT_PID_GAINS),
        "retuned": False,
        "trace_columns": list(tuned.comparison.TRACE_COLUMNS),
        "telemetry_columns": list(tuned.comparison.TRACE_COLUMNS)
        + list(tuned.LEARNING_COLUMNS),
        "runs": {},
    }
    traces, learning = {}, {}
    initial, _, report["trim"] = tuned.baseline.trim_and_agent(cfg)
    reference = tuned.comparison.reference_signal(cfg)
    for condition in conditions:
        for profile in PROFILES:
            key = f"{condition}_{profile}"
            print(f"Running {key}, duration={cfg.duration:g} s", flush=True)
            telemetry = []
            try:
                if profile == "pid":
                    trace, diagnostics = tuned.comparison.rollout_pid(
                        cfg,
                        initial,
                        reference,
                        tuned.comparison.DEFAULT_PID_GAINS,
                        fault=condition == "fault",
                    )
                else:
                    trace, diagnostics = tuned.rollout(
                        cfg,
                        tuned.INTEGRAL_TUNING,
                        fault=condition == "fault",
                        frozen_after_event=profile.endswith("frozen"),
                        telemetry=telemetry,
                    )
                boundaries = [cfg.fault_time]
                boundaries += [
                    boundary
                    for boundary in np.arange(100.0, cfg.duration, 100.0)
                    if boundary > cfg.fault_time
                ]
                boundaries.append(cfg.duration)
                result = {
                    "status": "complete",
                    "metrics": tuned.metrics(trace, cfg),
                    "blocks": [
                        window_metrics(trace, start, stop)
                        for start, stop in zip(boundaries[:-1], boundaries[1:])
                    ],
                    "last_100_s": window_metrics(
                        trace, max(cfg.fault_time, cfg.duration - 100), cfg.duration
                    ),
                    "diagnostics": diagnostics,
                }
                if len(trace) != cfg.steps or trace[-1, 0] != cfg.duration:
                    raise AssertionError("Incomplete trajectory")
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                # Never silently present a shorter trajectory as a complete run.
                trace = np.asarray(telemetry, dtype=float).reshape(
                    -1, len(report["telemetry_columns"])
                )[:, :8]
                result = {
                    "status": "failed",
                    "reason": str(exc),
                    "last_completed_time_s": float(trace[-1, 0]) if len(trace) else 0,
                }
            traces[key] = trace
            report["runs"][key] = result
            np.savetxt(
                output / f"{key}.csv",
                trace,
                delimiter=",",
                header=",".join(report["trace_columns"]),
                comments="",
            )
            if telemetry:
                learning[key] = np.asarray(telemetry)
                np.savetxt(
                    output / f"{key}_learning.csv",
                    learning[key],
                    delimiter=",",
                    header=",".join(report["telemetry_columns"]),
                    comments="",
                )
            # Save each completed or failed arm independently.
            (output / "metrics.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(key, result.get("last_100_s", result), flush=True)
    validate_prefault(traces, report, cfg, conditions)
    return traces, learning, report


def validate_prefault(traces, report, cfg, conditions):
    """Allow only roundoff from integration split at the event boundary."""
    if any(run["status"] != "complete" for run in report["runs"].values()):
        return
    for condition in conditions:
        np.testing.assert_allclose(
            traces[f"{condition}_iadp_integral"][: cfg.fault_step],
            traces[f"{condition}_iadp_integral_frozen"][: cfg.fault_step],
            rtol=0,
            atol=1e-10,
        )
    if "fault" in conditions and "healthy" in conditions:
        for profile in PROFILES:
            np.testing.assert_allclose(
                traces[f"fault_{profile}"][: cfg.fault_step],
                traces[f"healthy_{profile}"][: cfg.fault_step],
                rtol=0,
                atol=1e-10,
            )


def plot_results(traces, learning, cfg):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for column, condition in enumerate(("fault", "healthy")):
        if not any(key.startswith(f"{condition}_") for key in traces):
            for ax in axes[:, column]:
                ax.set_visible(False)
            continue
        for name, label, color in zip(PROFILES, LABELS, COLORS):
            key = f"{condition}_{name}"
            trace = traces.get(key)
            if trace is None or not len(trace):
                continue
            time = trace[:, 0]
            axes[0, column].plot(
                time, trace[:, 2] - trace[:, 1], color=color, lw=0.7, label=label
            )
            window = max(1, round(20 / cfg.dt))
            if len(trace) >= window:
                t, error = rolling_rmse(trace, window)
                axes[1, column].semilogy(t, error, color=color, label=label)
            selected = time > max(0, cfg.duration - 20)
            axes[2, column].plot(
                time[selected], trace[selected, 2], color=color, lw=1.2, label=label
            )
            if name == "pid":
                axes[2, column].plot(
                    time[selected], trace[selected, 1], "k--", lw=0.9, label="Reference"
                )
            if key in learning:
                data = learning[key]
                axes[3, column].plot(data[:, 0], data[:, 13], color=color, label=label)
        axes[0, column].set_title(
            f"{100 * cfg.loss:g}% command-gain loss at {cfg.fault_time:g} s"
            if condition == "fault"
            else "Healthy aircraft, same controllers"
        )
        for row, label in enumerate(
            (
                "Tracking error [deg/s]",
                "Trailing 20 s RMSE [deg/s]",
                "Pitch rate [deg/s]",
                "Critic change norm since 20 s",
            )
        ):
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlabel("Time [s]")
            axes[row, column].grid(alpha=0.25)
            if row != 2:
                axes[row, column].axvline(cfg.fault_time, color="0.5", ls=":")
        axes[0, column].legend(fontsize=8)
        axes[2, column].legend(fontsize=8)
    figure.suptitle(
        f"F-16: {cfg.duration:g} s validation, fixed tuning, control period 20 ms"
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/f16-iadp-500s"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("fault", "healthy"),
        default=["fault", "healthy"],
    )
    args = parser.parse_args()
    cfg = tuned.baseline.Experiment(
        duration=args.duration, integration_substeps=args.substeps
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces, learning, report = run_experiment(cfg, args.output, args.conditions)
    figure = plot_results(traces, learning, cfg)
    for suffix in ("png", "svg"):
        figure.savefig(args.output / f"comparison.{suffix}", dpi=150)
    plt.close(figure)
    if any(run["status"] != "complete" for run in report["runs"].values()):
        raise SystemExit("At least one run failed; partial diagnostics were saved")


if __name__ == "__main__":
    main()
