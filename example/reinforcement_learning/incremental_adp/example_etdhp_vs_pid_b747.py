"""ET-DHP versus PID on the nonlinear 6-DoF B747, with an unseen engine loss.

Only the environment knows the event schedule. Both lateral controllers share
an observation-only longitudinal hold loop. ET-DHP uses a healthy local-LQR
initialization and keeps the native event-triggered learning active online.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from tensoraerospace.aerospacemodel.b747.nonlinear import NonlinearB747, trim
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile,
    EngineFailureEvent,
)
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.agent.pid import B747LongitudinalHold as LongitudinalHold
from tensoraerospace.agent.pid import LateralAircraftPID as LateralPID
from tensoraerospace.agent.pid.aircraft import B747_LATERAL_PID_GAINS
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

ALTITUDE = 20000.0
SPEED = 674.0
BOUND = 8.0
INTEGRAL_SCALE = 0.1
# Two healthy-only Nelder-Mead stages; full histories accompany the report.
NOMINAL_PID_GAINS = B747_LATERAL_PID_GAINS
STATE_NAMES = (
    "beta_deg",
    "p_deg_s",
    "r_deg_s",
    "phi_deg",
    "psi_deg",
    "roll_integral_scaled",
    "heading_integral_scaled",
)
TRACE_NAMES = (
    "time_s",
    "psi_deg",
    "phi_deg",
    "beta_deg",
    "speed_error_ft_s",
    "height_error_ft",
    "theta_error_deg",
    "p_deg_s",
    "q_deg_s",
    "r_deg_s",
    "aileron_deg",
    "rudder_deg",
    "elevator_deg",
    "throttle",
    "triggered",
    "actor_change",
    "critic_change",
    "model_change",
    "integral_phi_deg_s",
    "integral_psi_deg_s",
)


@dataclass(frozen=True)
class Experiment:
    duration: float = 500.0
    dt: float = 0.05
    fault_time: float = 137.0
    engine_fraction: float = 0.5
    engine_id: int = 1
    substeps: int = 1
    initial_heading_deg: float = 1.0
    initial_roll_deg: float = 0.3
    seed: int = 11

    def __post_init__(self):
        vals = (
            self.duration,
            self.dt,
            self.fault_time,
            self.engine_fraction,
            self.initial_heading_deg,
            self.initial_roll_deg,
        )
        if not all(np.isfinite(v) for v in vals) or self.dt <= 0 or self.duration <= 0:
            raise ValueError("Finite positive duration/dt required")
        if (
            not 0 <= self.fault_time < self.duration
            or not 0 <= self.engine_fraction <= 1
        ):
            raise ValueError("Invalid fault time/effectiveness")
        if (
            self.engine_id not in (1, 2, 3, 4)
            or self.substeps < 1
            or not isinstance(self.substeps, (int, np.integer))
        ):
            raise ValueError("Invalid engine/substeps")
        if not np.isclose(self.duration / self.dt, round(self.duration / self.dt)):
            raise ValueError("Duration must align with control steps")
        h = self.dt / self.substeps
        if not np.isclose(
            self.fault_time / h, round(self.fault_time / h), atol=1e-8, rtol=0
        ):
            raise ValueError("Fault time must align with a physics step")

    @property
    def steps(self):
        return round(self.duration / self.dt)


@lru_cache(maxsize=1)
def nominal_trim():
    result = trim(ALTITUDE, SPEED)
    if not result.converged or result.residual > 1e-7:
        raise RuntimeError("B747 nominal trim did not converge")
    return result


lateral_state = NonlinearB747.lateral_state


def make_env(cfg, *, fault):
    initial = nominal_trim().to_state()
    initial[[6, 8]] = np.deg2rad([cfg.initial_roll_deg, cfg.initial_heading_deg])
    event = (
        EngineFailureEvent(
            trigger_time=cfg.fault_time,
            engine_id=cfg.engine_id,
            thrust_fraction=cfg.engine_fraction,
        )
        if fault
        else None
    )
    return NonlinearB747Env(
        initial_state=initial,
        dt=cfg.dt / cfg.substeps,
        number_time_steps=cfg.steps * cfg.substeps,
        action_space="virtual",
        damage_profile=DamageProfile(events=[event]) if event else None,
    )


def nominal_transition(x, action, dt):
    tr = nominal_trim()
    model = NonlinearB747(tr.to_state(), dt=dt)
    return model.lateral_transition(x, action, [tr.elevator_rad, 0, 0, tr.throttle])


def healthy_linearization(dt):
    tr = nominal_trim()
    model = NonlinearB747(tr.to_state(), dt=dt)
    return model.lateral_linearization([tr.elevator_rad, 0, 0, tr.throttle])


def initialize_network(network, matrix, scale=0.02):
    """Embed a healthy local linear map in the two-tanh-layer network.

    This is an explicit model-based initialization, not a claim of model-free
    learning. The neural weights remain trainable and no LQR runs in flight.
    """
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        n = matrix.shape[1]
        network.backbone[0].weight[:n, :n].copy_(scale * torch.eye(n))
        network.backbone[2].weight[:n, :n].copy_(torch.eye(n))
        network.head.weight[:, :n].copy_(
            torch.as_tensor(matrix / scale, dtype=torch.float32)
        )


def make_agent(
    dt=0.05,
    seed=11,
    *,
    actor_lr=1e-6,
    critic_lr=1e-5,
    model_lr=1e-7,
    rho=0.02,
    floor=0.01,
    online_model_fit=False,
):
    q = np.array([0.1, 0.2, 0.2, 2, 5, 1, 2]) * 1e-3
    r = np.ones(2) * 0.2e-3
    cfg = ETDHPConfig(
        actor_hidden=(16, 16),
        critic_hidden=(16, 16),
        model_hidden=(16, 16),
        Q=q.tolist(),
        R=r.tolist(),
        gamma=1.0,
        num_epochs_per_trigger=1,
        u_bound=BOUND,
        rho=rho,
        trigger_floor=floor,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        model_lr=model_lr,
        online_model_fit=online_model_fit,
        seed=seed,
    )
    agent = ETDHPAgent(7, 2, config=cfg)
    if agent.writer is not None:
        agent.writer.close()
        agent.writer = None
    a, b = healthy_linearization(dt)
    p = solve_discrete_are(a, b, np.diag(q), np.diag(r))
    k = np.linalg.solve(np.diag(r) + b.T @ p @ b, b.T @ p @ a)
    initialize_network(agent.plant_model, np.column_stack([a, b]))
    initialize_network(agent.actor, -k / BOUND)
    initialize_network(agent.critic, 2 * p)
    return agent, {
        "A": a.tolist(),
        "B": b.tolist(),
        "K_initial": k.tolist(),
        "P_initial": p.tolist(),
        "config": asdict(cfg),
        "initialization": "healthy local LQR embedded in trainable two-tanh-layer networks",
    }


def parameter_vector(module):
    return (
        torch.cat([p.detach().reshape(-1) for p in module.parameters()])
        .cpu()
        .numpy()
        .astype(float)
    )


def rollout(cfg, algorithm, *, fault, gains=None, template=None):
    if algorithm not in ("etdhp", "pid"):
        raise ValueError("Algorithm must be etdhp or pid")
    if algorithm == "etdhp" and template is None:
        raise ValueError("An ETDHP template is required")
    env = make_env(cfg, fault=fault)
    obs, _ = env.reset(seed=cfg.seed)
    lon = LongitudinalHold(cfg.dt)
    integrals = np.zeros(2)
    agent = copy.deepcopy(template) if algorithm == "etdhp" else None
    pid = LateralPID(gains, cfg.dt) if algorithm == "pid" else None
    if agent is not None:
        agent.reset()
        networks = (agent.actor, agent.critic, agent.plant_model)
        initial = [parameter_vector(net) for net in networks]
    before_event = None
    rows = []
    status = "complete"
    reason = None
    for step in range(cfg.steps):
        # Diagnostic snapshots are never used by either control law.
        if (
            agent is not None
            and before_event is None
            and step * cfg.dt >= cfg.fault_time - 1e-10
        ):
            before_event = [parameter_vector(net) for net in networks]
        x = lateral_state(obs, integrals)
        action = agent.predict(x, time_step=step) if agent else pid.command(obs)
        elevator, throttle = lon.command(obs)
        control = np.r_[elevator, np.deg2rad(action), throttle]
        for _ in range(cfg.substeps):
            obs, _, terminated, truncated, _ = env.step(control)
        angles = lateral_state(obs)[3:5]
        integrals = np.clip(integrals + cfg.dt * angles, -100, 100)
        metrics = (
            agent.learn(lateral_state(obs, integrals), time_step=step, dt=cfg.dt)
            if agent
            else {"triggered": 0}
        )
        changes = (
            [
                float(np.linalg.norm(parameter_vector(n) - base))
                for n, base in zip(networks, initial)
            ]
            if agent
            else [0, 0, 0]
        )
        state = lateral_state(obs, integrals)
        row = [
            (step + 1) * cfg.dt,
            state[4],
            state[3],
            state[0],
            np.linalg.norm(obs[:3]) - SPEED,
            -obs[11] - ALTITUDE,
            np.rad2deg(obs[7] - nominal_trim().theta_rad),
            *np.rad2deg(obs[3:6]),
            *action,
            np.rad2deg(elevator),
            throttle,
            metrics["triggered"],
            *changes,
            *integrals,
        ]
        rows.append(row)
        if (
            not np.all(np.isfinite(row))
            or abs(state[3]) > 30
            or abs(state[4]) > 45
            or abs(state[0]) > 15
            or abs(row[4]) > 100
            or abs(row[5]) > 1500
        ):
            status = "failed"
            reason = "nonfinite state or flight-envelope guard"
            break
        if terminated or (truncated and step + 1 < cfg.steps):
            status = "failed"
            reason = "early environment end"
            break
    result = {
        "status": status,
        "reason": reason,
        "steps": len(rows),
        "events": env.damage_events_log,
        "triggers": int(sum(row[14] for row in rows)),
        "triggers_after_event": int(
            sum(row[14] for row in rows if row[0] > cfg.fault_time)
        ),
        "parameter_change_after_event": (
            [
                float(np.linalg.norm(parameter_vector(net) - old))
                for net, old in zip(networks, before_event)
            ]
            if before_event is not None
            else None
        ),
    }
    return np.asarray(rows), result


def metrics(trace, cfg):
    result = {}
    for name, start, end in [
        ("whole", 0, cfg.duration),
        ("before_fault", 0, cfg.fault_time),
        ("after_fault", cfg.fault_time, cfg.duration),
        ("last_100_s", max(cfg.fault_time, cfg.duration - 100), cfg.duration),
    ]:
        mask = (trace[:, 0] > start) & (trace[:, 0] <= end)
        x = trace[mask]
        result[name] = (
            {
                label: {
                    "rmse": float(np.sqrt(np.mean(x[:, i] ** 2))),
                    "peak_abs": float(np.max(np.abs(x[:, i]))),
                }
                for i, label in [
                    (1, "heading_deg"),
                    (2, "roll_deg"),
                    (3, "sideslip_deg"),
                    (4, "speed_ft_s"),
                    (5, "height_ft"),
                ]
            }
            if len(x)
            else {}
        )
    result["command_rms_deg"] = np.sqrt(np.mean(trace[:, 10:12] ** 2, axis=0)).tolist()
    result["command_peak_deg"] = np.max(np.abs(trace[:, 10:12]), axis=0).tolist()
    result["network_change_norms"] = trace[-1, 15:18].tolist()
    return result


def tune_pid(max_evaluations=150, *, initial_gains=None, gain_bounds=None):
    cfg = Experiment(
        duration=60.0, fault_time=30.0, initial_heading_deg=2.0, initial_roll_deg=1.0
    )
    bounds = np.array(
        gain_bounds
        if gain_bounds is not None
        else [[0.1, 40], [0.005, 5], [0.05, 30]] * 2
    )
    history = []

    def objective(log_gains):
        gains = np.exp(log_gains)
        trace, diag = rollout(cfg, "pid", fault=False, gains=gains)
        value = float(
            np.mean(
                trace[:, 1] ** 2
                + trace[:, 2] ** 2
                + 0.002 * np.sum(trace[:, 10:12] ** 2, axis=1)
            )
        )
        if diag["status"] != "complete":
            value += 1000
        history.append(
            {"gains": gains.tolist(), "cost": value, "status": diag["status"]}
        )
        return value

    result = minimize(
        objective,
        np.log(initial_gains if initial_gains is not None else [2, 0.1, 2, 5, 0.3, 4]),
        method="Nelder-Mead",
        bounds=np.log(bounds),
        options={"maxfev": max_evaluations, "xatol": 0.03, "fatol": 1e-5},
    )
    gains = np.exp(result.x)
    return gains, {
        "healthy_only": True,
        "experiment": asdict(cfg),
        "bounds": bounds.tolist(),
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.fun),
        "evaluations": len(history),
        "history": history,
    }


def retune_nominal_pid():
    first_gains, first = tune_pid(
        150, gain_bounds=[[0.1, 20], [0.005, 2], [0.05, 20]] * 2
    )
    gains, second = tune_pid(180, initial_gains=first_gains)
    return gains, {"stage_one": first, "expanded_stage": second}


def save_comparison(output, cfg, traces, report):
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for key, value in traces.items():
        np.savetxt(
            output / f"{key}.csv",
            value,
            delimiter=",",
            header=",".join(TRACE_NAMES),
            comments="",
        )
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    panels = [
        (1, "Heading error [deg]"),
        (2, "Bank angle [deg]"),
        (4, "Airspeed error [ft/s]"),
        (5, "Altitude error [ft]"),
        (10, "Aileron command [deg]"),
        (11, "Rudder command [deg]"),
    ]
    for algorithm, color in [("etdhp", "tab:blue"), ("pid", "tab:orange")]:
        x = traces[f"fault_{algorithm}"]
        for ax, (column, title) in zip(axes.flat, panels):
            ax.plot(
                x[:, 0], x[:, column], color=color, lw=0.85, label=algorithm.upper()
            )
            ax.set_ylabel(title)
        late = x[:, 0] > max(0, cfg.duration - 100)
        axes[3, 0].plot(
            x[late, 0], x[late, 1], color=color, lw=0.8, label=algorithm.upper()
        )
        axes[3, 0].set_ylabel("Late heading error [deg]")
    x = traces["fault_etdhp"]
    axes[3, 1].plot(x[:, 0], np.cumsum(x[:, 14]), label="ET-DHP updates")
    axes[3, 1].set_ylabel("Cumulative event triggers")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[:6]:
        ax.axvline(cfg.fault_time, color="gray", ls=":")
    fig.suptitle(
        f"Nonlinear B747: engine {cfg.engine_id}, {100*(1-cfg.engine_fraction):g}% thrust loss at {cfg.fault_time:g} s\nHealthy initialization/tuning; event schedule belongs to environment only"
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"comparison.{ext}", dpi=150)
    return fig


def search_learning(*, online_model_fit=False):
    """Select a positive learning rate on healthy 500 s episodes only."""
    cfg = Experiment(initial_heading_deg=2.0, initial_roll_deg=1.0)
    history = []
    for rate in (1e-6, 1e-7, 1e-8):
        template, _ = make_agent(
            actor_lr=rate,
            critic_lr=10 * rate,
            model_lr=1e-7,
            online_model_fit=online_model_fit,
        )
        trace, diagnostics = rollout(cfg, "etdhp", fault=False, template=template)
        cost = float(
            np.mean(
                trace[:, 1] ** 2
                + trace[:, 2] ** 2
                + 0.002 * np.sum(trace[:, 10:12] ** 2, axis=1)
            )
        )
        history.append(
            {
                "rate": rate,
                "cost": cost,
                "diagnostics": diagnostics,
                "metrics": metrics(trace, cfg),
            }
        )
        print("Healthy ET-DHP search:", rate, cost, diagnostics["status"], flush=True)
    successful = [r for r in history if r["diagnostics"]["status"] == "complete"]
    if not successful:
        raise RuntimeError("No healthy ET-DHP candidate completed the horizon")
    return min(successful, key=lambda r: r["cost"])["rate"], history


def run_comparison(cfg, *, gains=None, include_online_model=False, actor_lr=1e-6):
    """Run fresh, paired controllers; the shared template is never mutated."""
    if gains is None:
        gains = NOMINAL_PID_GAINS
    template, initialization = make_agent(
        cfg.dt, cfg.seed, actor_lr=actor_lr, critic_lr=10 * actor_lr
    )
    online_template, online_init = make_agent(
        cfg.dt,
        cfg.seed,
        actor_lr=1e-8,
        critic_lr=1e-7,
        model_lr=1e-7,
        online_model_fit=True,
    )
    report = {
        "experiment": asdict(cfg),
        "etdhp_initialization": initialization,
        "pid_gains": list(gains),
        "controller_knows_schedule": False,
        "actor_critic_learning": "active on state-triggered events for the entire episode",
        "plant_model_learning": "nominal model held throughout; optional online-model variant is separate",
        "longitudinal_hold": "identical observation-only PI/PD; no scheduled feedforward",
        "trace_columns": TRACE_NAMES,
        "runs": {},
    }
    traces = {}
    algorithms = (
        ("etdhp", "pid", "etdhp_online_model")
        if include_online_model
        else ("etdhp", "pid")
    )
    if include_online_model:
        report["online_model_initialization"] = online_init
    for fault in (False, True):
        for name in algorithms:
            key = ("fault_" if fault else "healthy_") + name
            algorithm = "pid" if name == "pid" else "etdhp"
            selected = online_template if name == "etdhp_online_model" else template
            trace, diagnostics = rollout(
                cfg, algorithm, fault=fault, gains=gains, template=selected
            )
            traces[key] = trace
            report["runs"][key] = {**diagnostics, "metrics": metrics(trace, cfg)}
            print(
                key,
                diagnostics["status"],
                "heading RMSE after event:",
                report["runs"][key]["metrics"]["after_fault"].get("heading_deg"),
                flush=True,
            )
    for name in algorithms:
        n = round(cfg.fault_time / cfg.dt)
        if min(len(traces[f"healthy_{name}"]), len(traces[f"fault_{name}"])) < n:
            continue
        np.testing.assert_array_equal(
            traces[f"healthy_{name}"][:n], traces[f"fault_{name}"][:n]
        )
    return traces, report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/b747-etdhp-vs-pid")
    )
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--fault-time", type=float, default=137.0)
    parser.add_argument("--engine-fraction", type=float, default=0.5)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument(
        "--search",
        action="store_true",
        help="repeat the healthy ET-DHP learning-rate search",
    )
    parser.add_argument(
        "--retune-pid",
        action="store_true",
        help="repeat both healthy PID tuning stages",
    )
    parser.add_argument(
        "--online-model-comparison",
        action="store_true",
        help="also evaluate online plant-model learning",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Experiment(
        duration=args.duration,
        fault_time=args.fault_time,
        engine_fraction=args.engine_fraction,
        substeps=args.substeps,
    )
    actor_lr, search_history = search_learning() if args.search else (1e-6, None)
    gains, pid_training = (
        retune_nominal_pid() if args.retune_pid else (NOMINAL_PID_GAINS, None)
    )
    traces, report = run_comparison(
        cfg,
        gains=gains,
        include_online_model=args.online_model_comparison,
        actor_lr=actor_lr,
    )
    report["healthy_learning_search"] = search_history
    report["pid_training"] = pid_training
    import matplotlib.pyplot as plt

    plt.close(save_comparison(args.output, cfg, traces, report))


if __name__ == "__main__":
    main()
