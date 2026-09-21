"""Experimental loss of stabilator aerodynamic effectiveness, with measured servo.

For C in {Cy, Cm}: C_fault(delta) = C(0) + eta * [C(delta) - C(0)].
This is a parametric control-effectiveness test, not a calibrated model of a
particular damaged aircraft. Mass, geometry and native servo dynamics are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear._actuators import project_actuators
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import DamageProfile
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import LongitudinalF16
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    default_parameters,
)
from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16


@dataclass(frozen=True)
class AeroFault:
    time_s: float = 137.0
    effectiveness: float = 0.7

    def __post_init__(self):
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("Fault time must be finite and nonnegative")
        if not np.isfinite(self.effectiveness) or not 0 <= self.effectiveness <= 1:
            raise ValueError("Aerodynamic effectiveness must be in [0, 1]")


def effectiveness_rhs(state, control, time, params, *, effectiveness):
    """Scale only surface-dependent force/moment; preserve native servo equations.

    Translational and rotational accelerations are affine in Cy and Cm. Mixing
    their native right-hand sides at the same alpha and q therefore implements
    the stated coefficient formula, including the force moment about the CG.
    """
    native = f16_ode_long(state, control, time, params)
    if effectiveness == 1:
        return native
    zero_surface = np.array(state, dtype=float, copy=True)
    zero_surface[2] = 0.0
    neutral = f16_ode_long(zero_surface, control, time, params)
    native[:2] = neutral[:2] + effectiveness * (native[:2] - neutral[:2])
    return native


class EffectivenessModel(LongitudinalF16):
    """Use native integration/event splitting with an experiment-specific RHS."""

    def __init__(self, *args, aero_fault, on_fault, **kwargs):
        super().__init__(*args, **kwargs)
        self.aero_fault = aero_fault
        self.on_fault = on_fault
        self.effectiveness = 1.0
        self.fault_applied = False

    def _activate_fault(self):
        self.effectiveness = self.aero_fault.effectiveness
        self.fault_applied = True
        self.on_fault(self.aero_fault)

    def run_step(self, u, *, events=()):
        start = self.t0 + self.dt * (self.time_step - 1)
        end = self.t0 + self.dt * self.time_step
        scheduled = list(events)
        if (
            not self.fault_applied
            and start - 1e-12 <= self.aero_fault.time_s <= end + 1e-12
        ):
            scheduled.append(
                (
                    float(np.clip(self.aero_fault.time_s - start, 0, self.dt)),
                    self._activate_fault,
                )
            )
        return super().run_step(u, events=scheduled)

    def _advance(self, state, command, time, dt):
        control = self._prepare_control(command)
        rhs = partial(effectiveness_rhs, effectiveness=self.effectiveness)
        next_state = self._step_fn(rhs, state, control, time, dt, self.param)
        return project_actuators(next_state, self.param, ((2, "stab"),))


class EffectivenessEnv(NonlinearLongitudinalF16):
    """The schedule stays inside the environment and outside controller inputs."""

    def __init__(self, *args, aero_fault, **kwargs):
        self.aero_fault = aero_fault
        super().__init__(*args, **kwargs)

    def _record_fault(self, fault):
        if fault.effectiveness == 1.0:
            return
        self.damage_events_log.append(
            {
                "time": fault.time_s,
                "label": "stabilator_aerodynamic_effectiveness_loss",
                "event_type": "experimental_aerodynamic_effectiveness",
                "payload": {"effectiveness": fault.effectiveness},
            }
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        native = self.model
        self.model = EffectivenessModel(
            native.current_state,
            selected_state_output=self.state_space,
            dt=self.dt,
            integrator=self.integrator,
            aero_fault=self.aero_fault,
            on_fault=self._record_fault,
        )
        # Retain the SAME nominal parameters as the other experiments, including
        # the harmless geometry recalculation made by an empty DamageProfile.
        self.model.param = native.param
        self.model.damage_state = native.damage_state
        self.model.damage_geometry = native.damage_geometry
        if self.aero_fault.time_s == 0:
            self.model._activate_fault()
        return observation, info


def make_environment(
    cfg, initial_state, reference, *, fault, damage_profile=None, aero_fault=AeroFault()
):
    if damage_profile is not None and damage_profile.events:
        raise ValueError("Do not combine native and experimental faults in this test")
    scenario = aero_fault if fault else AeroFault(aero_fault.time_s, 1.0)
    return EffectivenessEnv(
        initial_state=initial_state.copy(),
        reference_signal=reference,
        number_time_steps=cfg.steps * cfg.integration_substeps + 1,
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["wz"],
        use_reward=False,
        dt=cfg.dt / cfg.integration_substeps,
        integrator="rk4",
        control_bias=float(np.rad2deg(initial_state[2])),
        airspeed=default_parameters().V,
        damage_profile=DamageProfile([]),
        aero_fault=scenario,
    )


# PID tuned on a separate healthy 20 s trajectory with expanded gain bounds.
NOMINAL_PID_GAINS = (-135.60695234860972, -982.6934506639378, -0.0020707024599338265)
PID_GAIN_BOUNDS = ((0.5, 1000.0), (0.5, 5000.0), (0.001, 100.0))


def selected_tuning():
    """Starting settings for the unregularized critic; rerun healthy tuning."""
    from .example_iadp_tuned_f16 import Tuning

    return Tuning(forgetting=0.9999)


def search_healthy():
    """Compare forgetting settings without evaluating any damaged plant."""
    from dataclasses import asdict, replace

    from . import example_iadp_tuned_f16 as tuned

    cfg = tuned.baseline.Experiment(duration=500.0, loss=0.0)
    _, nominal, _ = tuned.baseline.trim_and_agent(cfg)
    weight = float(nominal.R[0, 0])
    history = []
    for forgetting in (0.9995, 0.9999):
        tuning = replace(tuned.INTEGRAL_TUNING, forgetting=forgetting)
        trace, diagnostics = tuned.rollout(cfg, tuning, fault=False, training=True)
        if diagnostics["events"]:
            raise AssertionError("Fault data leaked into healthy tuning")
        error = np.deg2rad(trace[:, 2] - trace[:, 1])
        history.append(
            {
                "tuning": asdict(tuning),
                "cost": float(np.mean(error**2 + weight * trace[:, 7] ** 2)),
                "rmse_deg_s": float(np.sqrt(np.mean(np.rad2deg(error) ** 2))),
            }
        )
        print(f"Healthy candidate {len(history)}/2: {history[-1]['cost']}", flush=True)
    best = min(history, key=lambda row: row["cost"])
    return tuned.Tuning(**best["tuning"]), history


def run_comparison(cfg, aero_fault, *, tuning=None, pid_gains=NOMINAL_PID_GAINS):
    """Compare continuously learning iADP and fixed-gain PID, with healthy controls."""
    from dataclasses import asdict

    from . import example_iadp_tuned_f16 as tuned
    from .example_iadp_long_horizon_f16 import window_metrics

    tuning = selected_tuning() if tuning is None else tuning
    if not 0 < aero_fault.time_s < cfg.duration:
        raise ValueError("Fault must occur inside the episode")
    if aero_fault.time_s != cfg.fault_time:
        raise ValueError("Metrics and environment must use the same event time")
    factory = partial(make_environment, aero_fault=aero_fault)
    initial, _, trim = tuned.baseline.trim_and_agent(cfg)
    traces, learning, results = {}, {}, {}
    for condition, fault in (("healthy", False), ("fault", True)):
        for algorithm in ("iadp", "pid"):
            key = f"{condition}_{algorithm}"
            print(f"Running {key}, {cfg.duration:g} s", flush=True)
            rows = []
            if algorithm == "iadp":
                trace, diagnostics = tuned.rollout(
                    cfg,
                    tuning,
                    fault=fault,
                    environment_factory=factory,
                    telemetry=rows,
                    frozen_after_event=False,
                )
                learning[key] = np.asarray(rows)
            else:
                trace, diagnostics = tuned.comparison.rollout_pid(
                    cfg,
                    initial,
                    tuned.comparison.reference_signal(cfg),
                    pid_gains,
                    fault=fault,
                    environment_factory=factory,
                )
            traces[key] = trace
            results[key] = {
                **tuned.metrics(trace, cfg),
                **diagnostics,
                "whole_episode": window_metrics(trace, 0, cfg.duration),
                "last_100_s": window_metrics(
                    trace, max(cfg.fault_time, cfg.duration - 100), cfg.duration
                ),
                "mean_surface_delta_rms_deg": float(np.sqrt(np.mean(trace[:, 7] ** 2))),
            }
    for algorithm in ("iadp", "pid"):
        np.testing.assert_allclose(
            traces[f"healthy_{algorithm}"][: cfg.fault_step],
            traces[f"fault_{algorithm}"][: cfg.fault_step],
            rtol=0,
            atol=1e-10,
        )
    return (
        traces,
        learning,
        {
            "experiment": asdict(cfg),
            "aero_fault": asdict(aero_fault),
            "native_command_gain_loss": 0.0,
            "tuning": asdict(tuning),
            "pid_gains": list(pid_gains),
            "trim": trim,
            "metrics": results,
            "controller_knows_failure_schedule": False,
            "continuous_learning": True,
            "trace_columns": list(tuned.comparison.TRACE_COLUMNS),
            "learning_columns": list(tuned.comparison.TRACE_COLUMNS)
            + list(tuned.LEARNING_COLUMNS),
            "fault_semantics": "C_fault(delta)=C(0)+eta*(C(delta)-C(0)); C is Cy or Cm; native servo and geometry unchanged",
            "model_limit": "parametric effectiveness test, not a validated damaged-aircraft model",
        },
    )


def plot_results(traces, learning, report):
    import matplotlib.pyplot as plt

    cfg = report["experiment"]
    figure, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for algorithm, color, label in (
        ("iadp", "tab:blue", "iADP + integral, continuous"),
        ("pid", "tab:orange", "PID, expanded healthy tuning"),
    ):
        x = traces[f"fault_{algorithm}"]
        time, error = x[:, 0], x[:, 2] - x[:, 1]
        for ax in axes[0]:
            ax.plot(time, x[:, 2], color=color, lw=1, label=label)
        axes[1, 0].plot(time, error, color=color, lw=0.8, label=label)
        tail = time >= max(0, cfg["duration"] - 100)
        axes[1, 1].plot(time[tail], error[tail], color=color, lw=0.8, label=label)
        axes[3, 0].plot(time, x[:, 4], color=color, lw=0.8, label=label)
        healthy = traces[f"healthy_{algorithm}"]
        axes[3, 1].plot(
            healthy[:, 0],
            healthy[:, 2] - healthy[:, 1],
            color=color,
            lw=0.8,
            label=label,
        )
    x = traces["fault_iadp"]
    for ax in axes[0]:
        ax.plot(x[:, 0], x[:, 1], "k--", lw=0.8, label="Reference")
        ax.set_ylabel("Pitch rate [deg/s]")
    axes[0, 1].set_xlim(
        cfg["fault_time"] - 1, min(cfg["duration"], cfg["fault_time"] + 3)
    )
    axes[1, 1].set_xlim(max(0, cfg["duration"] - 100), cfg["duration"])
    for ax in axes[1]:
        ax.set_ylabel("Tracking error [deg/s]")
    for condition, color in (("fault", "tab:blue"), ("healthy", "tab:green")):
        x = learning[f"{condition}_iadp"]
        axes[2, 0].plot(x[:, 0], x[:, 10], color=color, label=condition)
        axes[2, 1].plot(x[:, 0], x[:, 11], color=color, label=condition)
    axes[2, 0].set_ylabel("Identified G_q")
    axes[2, 1].set_ylabel("Critic norm (updates remain active)")
    axes[3, 0].set_ylabel("Actual stabilator [deg]")
    axes[3, 1].set_ylabel("Healthy tracking error [deg/s]")
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.25)
        ax.axvline(cfg["fault_time"], ls=":", color="gray", lw=0.8)
        ax.legend(fontsize=8)
    loss = 100 * (1 - report["aero_fault"]["effectiveness"])
    figure.suptitle(
        f"F-16: {loss:g}% aerodynamic effectiveness loss, {cfg['duration']:g} s\nNo controller notification, freezing or reset at the event"
    )
    return figure


def save_results(traces, learning, report, output):
    import json
    from pathlib import Path

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, trace in traces.items():
        np.savetxt(
            output / f"{name}.csv",
            trace,
            delimiter=",",
            header=",".join(report["trace_columns"]),
            comments="",
        )
    for name, trace in learning.items():
        np.savetxt(
            output / f"{name}_learning.csv",
            trace,
            delimiter=",",
            header=",".join(report["learning_columns"]),
            comments="",
        )
    figure = plot_results(traces, learning, report)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"comparison.{suffix}", dpi=150)
    return figure


def main():
    import argparse
    from pathlib import Path

    from . import example_iadp_tuned_f16 as tuned

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--fault-time", type=float, default=137.0)
    parser.add_argument("--effectiveness", type=float, default=0.7)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--search",
        action="store_true",
        help="repeat the six healthy-only iADP tuning trials",
    )
    parser.add_argument(
        "--retune-pid",
        action="store_true",
        help="repeat expanded healthy-only PID tuning",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/f16-iadp-effectiveness")
    )
    args = parser.parse_args()
    cfg = tuned.baseline.Experiment(
        duration=args.duration,
        fault_time=args.fault_time,
        loss=0.0,
        phase=args.phase,
        integration_substeps=args.substeps,
    )
    fault = AeroFault(args.fault_time, args.effectiveness)
    tuning, history = selected_tuning(), None
    if args.search:
        tuning, history = search_healthy()
    gains, pid_history = NOMINAL_PID_GAINS, None
    if args.retune_pid:
        gains, pid_history = tuned.comparison.tune_pid(
            max_evaluations=160, gain_magnitude_bounds=PID_GAIN_BOUNDS
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces, learning, report = run_comparison(
        cfg, fault, tuning=tuning, pid_gains=gains
    )
    report["iadp_healthy_search"] = history
    report["pid_healthy_search"] = pid_history
    figure = save_results(traces, learning, report, args.output)
    plt.close(figure)
    for name, metrics in report["metrics"].items():
        print(
            name,
            "post-event RMSE",
            metrics["after_fault"]["rmse_deg_s"],
            "last 100 s",
            metrics["last_100_s"]["rmse_deg_s"],
        )


if __name__ == "__main__":
    main()
