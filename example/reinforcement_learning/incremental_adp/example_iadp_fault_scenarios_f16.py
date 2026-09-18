"""Compare continuously learning iADP with fixed-gain PID under unknown faults.

The schedule is drawn and saved before evaluation. Only the environment receives
it; neither controller is reset, frozen, retuned or notified at a damage event.
Wing loss uses the existing heuristic sectional model, not validated damage data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import example_iadp_long_horizon_f16 as long_run
from . import example_iadp_tuned_f16 as tuned

DamageEvent = tuned.baseline.DamageEvent
DamageProfile = tuned.baseline.DamageProfile


def scenarios(duration=500.0, dt=0.02, seed=20260918):
    """Declare all schedules without looking at controller outcomes."""
    if duration <= 0 or dt <= 0 or not np.isfinite([duration, dt]).all():
        raise ValueError("duration and dt must be positive and finite")
    rng = np.random.default_rng(seed)

    def align(time):
        return round(time / dt) * dt

    def gain(time, efficiency, label):
        return DamageEvent(
            align(time),
            "control_failure",
            {
                "surface": "stab_left",
                "mode": "healthy" if efficiency == 1 else "efficiency_loss",
                "efficiency": efficiency,
            },
            label=label,
        )

    onset = align(duration * rng.uniform(0.45, 0.65))
    late = DamageProfile([gain(onset, 0.85, "late_command_gain_loss_15pct")])
    onset = align(duration * rng.uniform(0.12, 0.25))
    ramp = DamageProfile(
        [
            gain(
                onset + fraction * 0.24 * duration,
                1 - fraction * 0.20,
                "progressive_command_gain_loss",
            )
            for fraction in np.linspace(0, 1, 61)[1:]
        ]
    )
    onset = align(duration * rng.uniform(0.10, 0.14))
    intermittent = DamageProfile(
        [
            gain(onset + offset * duration, efficiency, "intermittent_command_gain")
            for offset, efficiency in (
                (0, 0.90),
                (0.10, 1),
                (0.26, 0.85),
                (0.40, 1),
                (0.60, 0.82),
            )
        ]
    )
    onset = align(duration * rng.uniform(0.25, 0.50))
    tips = DamageProfile(
        [
            DamageEvent(
                onset,
                "section_loss",
                {"section": section, "loss_fraction": 0.30},
                label=f"{section}_loss_30pct",
            )
            for section in ("left_tip", "right_tip")
        ]
    )
    return {
        "late_gain_loss": late,
        "progressive_gain_loss": ramp,
        "intermittent_gain_loss": intermittent,
        "symmetric_wing_tip_loss": tips,
    }


def run_experiment(duration, seed, output, *, substeps=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profiles = scenarios(duration=duration, seed=seed)
    report = {
        "duration_s": duration,
        "seed": seed,
        "dt_s": 0.02,
        "integration_substeps": substeps,
        "controller_knows_fault_schedule": False,
        "continuous_learning": True,
        "retuning": False,
        "tuning": asdict(tuned.INTEGRAL_TUNING),
        "pid_gains": list(tuned.comparison.DEFAULT_PID_GAINS),
        "scenarios": {name: profile.to_dict() for name, profile in profiles.items()},
        "trace_columns": list(tuned.comparison.TRACE_COLUMNS),
        "learning_columns": list(tuned.comparison.TRACE_COLUMNS)
        + list(tuned.LEARNING_COLUMNS),
        "results": {},
    }
    # Persist the full scenario list before observing any result.
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    traces = {}
    healthy = {}
    # An independent fault-free control run exposes deterioration unrelated to damage.
    for name, profile in {"healthy": DamageProfile([]), **profiles}.items():
        first = min(
            (event.trigger_time for event in profile.events), default=duration / 2
        )
        cfg = tuned.baseline.Experiment(
            duration=duration, fault_time=first, integration_substeps=substeps
        )
        for algorithm in ("iadp", "pid"):
            key = f"{name}_{algorithm}"
            print(f"Running {key}", flush=True)
            telemetry = []
            try:
                if algorithm == "iadp":
                    trace, diagnostics = tuned.rollout(
                        cfg,
                        tuned.INTEGRAL_TUNING,
                        fault=True,
                        damage_profile=profile,
                        telemetry=telemetry,
                        frozen_after_event=False,
                    )
                else:
                    initial, _, _ = tuned.baseline.trim_and_agent(cfg)
                    trace, diagnostics = tuned.comparison.rollout_pid(
                        cfg,
                        initial,
                        tuned.comparison.reference_signal(cfg),
                        tuned.comparison.DEFAULT_PID_GAINS,
                        fault=True,
                        damage_profile=profile,
                        telemetry=telemetry,
                    )
                if name == "healthy":
                    healthy[algorithm] = trace
                else:
                    np.testing.assert_allclose(
                        trace[: cfg.fault_step],
                        healthy[algorithm][: cfg.fault_step],
                        atol=1e-10,
                        rtol=0,
                    )
                events = sorted({event.trigger_time for event in profile.events})
                result = {
                    "status": "complete",
                    "has_fault": bool(profile.events),
                    "whole_episode": long_run.window_metrics(trace, 0, duration),
                    "before_first_event": long_run.window_metrics(trace, 0, first),
                    "after_first_event": long_run.window_metrics(
                        trace, first, duration
                    ),
                    "last_100_s": long_run.window_metrics(
                        trace, max(0, duration - 100), duration
                    ),
                    "event_transients": [
                        long_run.window_metrics(trace, event, min(event + 10, duration))
                        for event in events
                    ],
                    "metrics": tuned.metrics(trace, cfg),
                    "diagnostics": diagnostics,
                }
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                columns = 15 if algorithm == "iadp" else 8
                trace = np.asarray(telemetry).reshape(-1, columns)[:, :8]
                result = {
                    "status": "failed",
                    "reason": str(exc),
                    "last_completed_time_s": float(trace[-1, 0]) if len(trace) else 0,
                }
            traces[key] = trace
            report["results"][key] = result
            np.savetxt(
                output / f"{key}.csv",
                trace,
                delimiter=",",
                header=",".join(report["trace_columns"]),
                comments="",
            )
            if telemetry and algorithm == "iadp":
                np.savetxt(
                    output / f"{key}_learning.csv",
                    telemetry,
                    delimiter=",",
                    header=",".join(report["learning_columns"]),
                    comments="",
                )
            (output / "metrics.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(key, result.get("after_first_event", result), flush=True)
    return traces, report


def plot_results(traces, report):
    import matplotlib.pyplot as plt

    names = ["healthy", *report["scenarios"]]
    figure, axes = plt.subplots(
        len(names), 2, figsize=(15, 15), constrained_layout=True
    )
    for row, name in enumerate(names):
        for algorithm, color in (("iadp", "tab:blue"), ("pid", "tab:orange")):
            trace = traces[f"{name}_{algorithm}"]
            if not len(trace):
                continue
            axes[row, 0].plot(
                trace[:, 0],
                trace[:, 2] - trace[:, 1],
                lw=0.7,
                color=color,
                label=algorithm.upper(),
            )
            window = round(20 / report["dt_s"])
            if len(trace) >= window:
                time, error = long_run.rolling_rmse(trace, window)
                axes[row, 1].semilogy(time, error, color=color, label=algorithm.upper())
        for ax in axes[row]:
            ax.grid(alpha=0.25)
            ax.set_xlabel("Time [s]")
            ax.set_xlim(0, report["duration_s"])
            ax.legend(fontsize=8)
            times = {
                e["trigger_time"]
                for e in report["scenarios"].get(name, {}).get("events", [])
            }
            for time in times:
                ax.axvline(time, color="gray", alpha=0.15, lw=0.6)
        axes[row, 0].set_title(name.replace("_", " "))
        axes[row, 0].set_ylabel("Tracking error [deg/s]")
        axes[row, 1].set_ylabel("Trailing 20 s RMSE [deg/s]")
    figure.suptitle(
        "F-16: continuously learning iADP + integral vs fixed-gain PID\nFault schedule is available only to the environment; no retuning"
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/f16-iadp-fault-scenarios")
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces, report = run_experiment(
        args.duration, args.seed, args.output, substeps=args.substeps
    )
    figure = plot_results(traces, report)
    for suffix in ("png", "svg"):
        figure.savefig(args.output / f"comparison.{suffix}", dpi=150)
    plt.close(figure)
    if any(result["status"] != "complete" for result in report["results"].values()):
        raise SystemExit("One or more runs failed; partial diagnostics were saved")


if __name__ == "__main__":
    main()
