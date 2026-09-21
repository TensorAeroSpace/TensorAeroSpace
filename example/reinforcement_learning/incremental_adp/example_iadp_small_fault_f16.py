"""Paired iADP/F-16 demonstration with a small stabilator command-gain fault.

Run from the repository root:
    python -m example.reinforcement_learning.incremental_adp.example_iadp_small_fault_f16

The native DamageProfile scales the TOTAL servo command, including trim. It does
not change aerodynamic tables. Both controllers learn until the event; one then
keeps its learned model and critic fixed, while measurement feedback stays active.
See the accompanying documentation for the experiment's physical limitations.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import root

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    default_parameters,
)
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16

SCENARIOS = {
    "healthy_adaptive": (False, True),
    "healthy_frozen": (False, False),
    "fault_adaptive": (True, True),
    "fault_frozen": (True, False),
}
COLUMNS = (
    "time_s",
    "reference_deg_s",
    "q_deg_s",
    "alpha_deg",
    "surface_deg",
    "surface_rate_deg_s",
    "command_delta_deg",
    "mean_surface_delta_deg",
    "identified_F",
    "identified_G",
    "critic_norm",
    "model_change_since_event",
    "critic_change_since_event",
    "covariance_min_eigenvalue",
)


@dataclass(frozen=True)
class Experiment:
    """Seconds, degrees at the environment boundary, radians inside the plant."""

    dt: float = 0.02
    duration: float = 60.0
    fault_time: float = 20.0
    loss: float = 0.15
    phase: float = 0.0
    integration_substeps: int = 1

    def __post_init__(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("All experiment parameters must be finite")
        if self.dt != 0.02:
            raise ValueError(
                "This tuning uses dt=0.02 s; refine integration_substeps instead"
            )
        if type(self.integration_substeps) is not int or self.integration_substeps < 1:
            raise ValueError("integration_substeps must be a positive integer")
        if not 0 < self.fault_time < self.duration:
            raise ValueError("Require 0 < fault_time < duration")
        if not 0 <= self.loss <= 0.2:
            raise ValueError("This small-fault demo supports loss in [0, 0.2]")
        for value in (self.fault_time, self.duration):
            if not np.isclose(
                value / self.dt, round(value / self.dt), atol=1e-8, rtol=0
            ):
                raise ValueError("fault_time and duration must align with dt")

    @property
    def steps(self):
        return round(self.duration / self.dt)

    @property
    def fault_step(self):
        return round(self.fault_time / self.dt)


def trim_and_agent(cfg):
    """Initialize from a healthy local linearization; no faulty model is used."""
    params = default_parameters()
    solution = root(
        lambda z: f16_ode_long([z[0], 0.0, z[1], 0.0], [z[1]], 0.0, params)[:2],
        np.deg2rad([2.0, -2.0]),
    )
    if not solution.success or np.max(np.abs(solution.fun)) > 1e-10:
        raise RuntimeError(f"F-16 trim failed: {solution.message}")
    alpha, surface = solution.x
    initial_state = np.array([alpha, 0.0, surface, 0.0])

    def derivative(index):
        plus, minus = initial_state.copy(), initial_state.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        return (
            f16_ode_long(plus, [surface], 0.0, params)[1]
            - f16_ode_long(minus, [surface], 0.0, params)[1]
        ) / 2e-5

    gain = derivative(2) * np.pi / 180  # rad/s² per degree of actual surface
    F = np.diag([1 + cfg.dt * derivative(1), 1.0])
    G = np.array([[cfg.dt * gain], [0.0]])
    R = np.array([[(abs(gain) / 20) ** 2]])
    gamma = 0.99
    P = solve_discrete_are(
        np.sqrt(gamma) * F,
        np.sqrt(gamma) * G,
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        R,
    )
    agent = IADPAgent(
        1,
        1,
        IADPConfig(
            dt=cfg.dt,
            F_init=F,
            G_init=G,
            P_init=P,
            Q=np.eye(1),
            R=R,
            gamma=gamma,
            gamma_rls=0.9995,
            phi_init=1e3,
            policy_eval_window=300,
            policy_eval_every=20,
            policy_eval_warmup_updates=40,
            u_magnitude_limit=10,
            u_rate_limit=60,
        ),
    )
    return (
        initial_state,
        agent,
        {
            "state_rad": initial_state.tolist(),
            "residual": solution.fun.tolist(),
            "healthy_surface_gain_rad_s2_per_deg": float(gain),
            "airspeed_m_s": params.V,
            "altitude_m": params.Oy,
        },
    )


def make_environment(cfg, initial_state, reference, *, fault, damage_profile=None):
    # In this longitudinal model both stab_left and stab_right map to the ONE
    # collective channel. One event attenuates it once; two would square the gain.
    events = (
        [
            DamageEvent(
                trigger_time=cfg.fault_time,
                event_type="control_failure",
                payload={
                    "surface": "stab_left",
                    "mode": "efficiency_loss",
                    "efficiency": 1 - cfg.loss,
                },
                label="collective_stabilator_command_gain_loss",
            )
        ]
        if fault
        else []
    )
    return NonlinearLongitudinalF16(
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
        damage_profile=(
            copy.deepcopy(damage_profile)
            if fault and damage_profile is not None
            else DamageProfile(events=events)
        ),
    )


def learn_transition(agent, observation, reference, k, applied, *, freeze):
    """Keep rolling measurement/input history alive even in the frozen arm."""
    if freeze:
        theta, covariance, critic = (
            agent.rls.theta.copy(),
            agent.rls.Phi.copy(),
            agent.P.copy(),
        )
    agent.learn(observation[1:2].astype(float), reference, k, applied_action=applied)
    if freeze:
        agent.rls.theta[:] = theta
        agent.rls.Phi[:] = covariance
        agent.P[:] = critic


def rollout(cfg, initial_state, initial_agent, reference, *, fault, adaptive):
    agent = copy.deepcopy(initial_agent)
    env = make_environment(cfg, initial_state, reference, fault=fault)
    observation, _ = env.reset()
    state = initial_state.copy()
    rows = []
    theta_event, critic_event = None, None
    command_saturation_steps = 0
    try:
        for k in range(cfg.steps):
            if k == cfg.fault_step:
                theta_event, critic_event = agent.rls.theta.copy(), agent.P.copy()
            command = agent.predict(observation[1:2].astype(float), reference, k)
            # Hold the command for the full controller period, even when the
            # plant integrator is refined. Approximate mean actual position
            # using a composite trapezoid, then remove the same trim bias.
            surface_mean = 0.0
            for _ in range(cfg.integration_substeps):
                previous_surface = state[2]
                observation, _, terminated, truncated, _ = env.step(command)
                state = env.model.current_state
                surface_mean += 0.5 * (previous_surface + state[2])
            surface_mean /= cfg.integration_substeps
            applied = np.array([np.rad2deg(surface_mean - initial_state[2])])
            learn_transition(
                agent,
                observation,
                reference,
                k,
                applied,
                freeze=not adaptive and k >= cfg.fault_step,
            )
            matrices = (state, command, agent.rls.theta, agent.rls.Phi, agent.P)
            if not all(np.isfinite(matrix).all() for matrix in matrices):
                raise FloatingPointError(f"Nonfinite state/parameters at step {k}")
            covariance_min = float(np.linalg.eigvalsh(agent.rls.Phi).min())
            if covariance_min <= 0 or agent.G[0, 0] >= 0:
                raise RuntimeError(f"Invalid identifier covariance/gain at step {k}")
            if abs(state[1]) > np.deg2rad(10) or abs(
                state[0] - initial_state[0]
            ) > np.deg2rad(10):
                raise RuntimeError(
                    f"Left the demonstration flight envelope at step {k}"
                )
            if (
                abs(state[2]) > env.model.param.maxabsstab + 1e-10
                or abs(state[3]) > env.model.param.maxabsdstab + 1e-10
            ):
                raise RuntimeError(f"Servo limits exceeded at step {k}")
            if (terminated or truncated) and k + 1 < cfg.steps:
                raise RuntimeError(f"Environment ended early at step {k}")
            command_saturation_steps += int(abs(command[0]) >= 10 - 1e-9)
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
                    float(agent.F[0, 0]),
                    float(agent.G[0, 0]),
                    float(np.linalg.norm(agent.P)),
                    (
                        0.0
                        if theta_event is None
                        else float(np.linalg.norm(agent.rls.theta - theta_event))
                    ),
                    (
                        0.0
                        if critic_event is None
                        else float(np.linalg.norm(agent.P - critic_event))
                    ),
                    covariance_min,
                ]
            )
        return np.asarray(rows), {
            "events": copy.deepcopy(env.damage_events_log),
            "command_saturation_steps": command_saturation_steps,
        }
    finally:
        env.close()


def tracking_metrics(trace, cfg):
    times = trace[:, 0]
    error = trace[:, 2] - trace[:, 1]
    remaining = cfg.duration - cfg.fault_time
    windows = {
        "before_fault": (0.0, cfg.fault_time),
        "after_fault": (cfg.fault_time, cfg.duration),
        "initial_transient": (cfg.fault_time, cfg.fault_time + min(5.0, remaining / 2)),
        "late": (cfg.fault_time + remaining / 2, cfg.duration),
    }
    result = {}
    for name, (start, stop) in windows.items():
        selected = error[(times > start) & (times <= stop)]
        result[name] = {
            "start_s": start,
            "stop_s": stop,
            "rmse_deg_s": float(np.sqrt(np.mean(selected**2))),
            "peak_error_deg_s": float(np.max(np.abs(selected))),
        }
    result["max_abs_q_deg_s"] = float(np.max(np.abs(trace[:, 2])))
    result["max_abs_alpha_deg"] = float(np.max(np.abs(trace[:, 3])))
    result["max_abs_surface_deg"] = float(np.max(np.abs(trace[:, 4])))
    result["max_abs_surface_rate_deg_s"] = float(np.max(np.abs(trace[:, 5])))
    return result


def run_experiment(cfg):
    initial_state, agent, trim = trim_and_agent(cfg)
    times = np.arange(cfg.steps + 1) * cfg.dt
    reference = np.deg2rad(
        0.5 * np.sin(2 * np.pi * 0.12 * times)
        + 0.15 * np.sin(2 * np.pi * 0.31 * times + cfg.phase)
    )[None, :]
    traces, metrics = {}, {}
    for name, (fault, adaptive) in SCENARIOS.items():
        trace, diagnostics = rollout(
            cfg, initial_state, agent, reference, fault=fault, adaptive=adaptive
        )
        traces[name] = trace
        metrics[name] = {**tracking_metrics(trace, cfg), **diagnostics}
    # All arms use exactly the same measurements and parameter updates until
    # the event. Fail rather than silently compare different initial histories.
    common = traces["healthy_adaptive"][: cfg.fault_step]
    prefault_max_difference = max(
        float(np.max(np.abs(trace[: cfg.fault_step] - common)))
        for trace in traces.values()
    )
    for trace in traces.values():
        np.testing.assert_allclose(trace[: cfg.fault_step], common, rtol=0, atol=1e-10)
    improvement = {}
    for window in ("after_fault", "late"):
        adaptive = metrics["fault_adaptive"][window]["rmse_deg_s"]
        frozen = metrics["fault_frozen"][window]["rmse_deg_s"]
        improvement[window] = 100 * (1 - adaptive / frozen) if frozen > 0 else None
    report = {
        "experiment": asdict(cfg),
        "trim": trim,
        "controller": agent.get_param_env(),
        "metrics": metrics,
        "fault_rmse_reduction_percent": improvement,
        "prefault_histories_match": True,
        "prefault_max_absolute_difference": prefault_max_difference,
        "fault_semantics": "servo_target = (1-loss) * clip(trim + command_delta); aerodynamic tables unchanged",
        "frozen_semantics": "freeze RLS theta/covariance and critic P at event; retain measurement and actuator feedback",
        "trace_columns": list(COLUMNS),
    }
    return traces, report


def plot_results(traces, report):
    """A standalone scientific figure; no GUI or file writes on import."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 2, figsize=(14, 10), sharex=True, constrained_layout=True
    )
    styles = {
        "healthy_adaptive": ("Healthy / adaptive", "#228833", "-"),
        "healthy_frozen": ("Healthy / frozen", "#999999", ":"),
        "fault_adaptive": ("Fault / adaptive", "#0077bb", "-"),
        "fault_frozen": ("Fault / frozen", "#cc3311", "--"),
    }
    cfg = report["experiment"]
    t = traces["fault_adaptive"][:, 0]
    axes[0, 0].plot(t, traces["fault_adaptive"][:, 1], "k:", lw=1.5, label="Command")
    for name, trace in traces.items():
        label, color, line = styles[name]
        data = [
            trace[:, 2],
            trace[:, 2] - trace[:, 1],
            trace[:, 4],
            trace[:, 3],
            trace[:, 11],
            trace[:, 12],
        ]
        for ax, values in zip(axes.flat, data):
            ax.plot(t, values, color=color, ls=line, lw=1.25, label=label)
    titles = [
        ("Pitch-rate tracking", "q [deg/s]"),
        ("Tracking error", "q - command [deg/s]"),
        ("Actual stabilator position", "Surface [deg]"),
        ("Angle of attack", "Alpha [deg]"),
        ("Identifier change since event", "||theta - theta_event||"),
        ("Critic change since event", "||P - P_event||"),
    ]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.axvline(cfg["fault_time"], color="#555555", ls=":", lw=1)
        ax.set(title=title, ylabel=ylabel, xlim=(0, cfg["duration"]))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    reduction = report["fault_rmse_reduction_percent"]["after_fault"]
    fig.suptitle(
        f"Nonlinear F-16 / iADP | {100 * cfg['loss']:.0f}% servo command-gain loss at "
        f"{cfg['fault_time']:g} s\nPost-fault RMSE reduction vs frozen controller: {reduction:.1f}%",
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
            header=",".join(COLUMNS),
            comments="",
        )
    fig = plot_results(traces, report)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/f16-small-fault"))
    parser.add_argument("--loss", type=float, default=0.15)
    parser.add_argument("--fault-time", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--substeps", type=int, default=1, help="RK4 substeps per 20 ms control period"
    )
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        cfg = Experiment(
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

    traces, report = run_experiment(cfg)
    fig = save_results(traces, report, args.output)
    for name, metrics in report["metrics"].items():
        print(
            f"{name:18s} post-fault RMSE: {metrics['after_fault']['rmse_deg_s']:.6f} deg/s"
            f"  late: {metrics['late']['rmse_deg_s']:.6f} deg/s"
        )
    print(f"Saved metrics, full CSV traces and figures to {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
