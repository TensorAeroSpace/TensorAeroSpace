"""Healthy-only ET-DHP tuning and independent nonlinear B747 evaluation.

The actor/critic remain trainable throughout the flight. Cost and trigger
parameters are fixed before testing faults; the plant and PID are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_step_response_b747 as steps,
    example_etdhp_vs_pid_b747 as base,
)


@dataclass(frozen=True)
class Tuning:
    q: tuple = (0.1, 0.2, 0.2, 2.0, 5.0, 1.0, 2.0)
    r: tuple = (0.2, 0.2)
    actor_lr: float = 1e-7
    critic_lr: float = 1e-6
    rho: float = 0.001
    trigger_floor: float = 0.0001
    epochs: int = 1
    embedding_scale: float = 0.02
    cost_scale: float = 0.001

    def __post_init__(self):
        object.__setattr__(self, "q", tuple(self.q))
        object.__setattr__(self, "r", tuple(self.r))
        scalars = [
            self.actor_lr,
            self.critic_lr,
            self.rho,
            self.trigger_floor,
            self.embedding_scale,
            self.cost_scale,
        ]
        if (
            len(self.q) != 7
            or len(self.r) != 2
            or not np.isfinite([*self.q, *self.r, *scalars]).all()
            or min(
                *self.q,
                *self.r,
                self.actor_lr,
                self.critic_lr,
                self.embedding_scale,
                self.cost_scale,
            )
            <= 0
            or not 0 < self.rho < 0.5
            or self.trigger_floor < 0
            or not isinstance(self.epochs, int)
            or self.epochs < 1
        ):
            raise ValueError(
                "Finite positive costs, active learning and valid trigger parameters required"
            )


# Selected before fault validation; all learning rates stay positive.
FAST_TUNED = Tuning(
    q=(
        0.00871534531155019,
        0.5874872073134174,
        0.11184034413376791,
        7.6925000139040645,
        92.16586099477253,
        7.033410616481587,
        0.5906411686898158,
    ),
    r=(0.013668067235660195, 0.06581167696853699),
    actor_lr=1e-07,
    critic_lr=1e-06,
    rho=0.01,
    trigger_floor=0.0001,
    epochs=1,
    embedding_scale=0.02,
)


# Lowest all-channel healthy score among the 12 balanced candidates.
TUNED = Tuning(
    q=(
        0.9073718885475248,
        0.8234691961911224,
        0.03975736005160585,
        153.57761735858224,
        36.007889593288255,
        16.44426339214549,
        16.594033001885215,
    ),
    r=(0.1605830391504966, 0.026911643586993605),
    actor_lr=1e-07,
    critic_lr=1e-06,
    rho=0.005,
    trigger_floor=0.0001,
    epochs=1,
    embedding_scale=0.02,
    cost_scale=0.0001,
)


def make_agent(settings: Tuning, *, dt=0.05, seed=11):
    agent, info = base.make_agent(
        dt=dt,
        seed=seed,
        actor_lr=settings.actor_lr,
        critic_lr=settings.critic_lr,
        rho=settings.rho,
        floor=settings.trigger_floor,
    )
    q, r = (
        np.asarray(settings.q) * settings.cost_scale,
        np.asarray(settings.r) * settings.cost_scale,
    )
    if (
        q.shape != (7,)
        or r.shape != (2,)
        or not np.isfinite(np.r_[q, r]).all()
        or min(*q, *r) <= 0
    ):
        raise ValueError("Positive finite Q[7] and R[2] are required")
    agent.cfg.Q, agent.cfg.R = q.tolist(), r.tolist()
    agent.cfg.num_epochs_per_trigger = settings.epochs
    agent.Q = torch.tensor(q, dtype=torch.float32)
    agent.R = torch.tensor(r, dtype=torch.float32)
    a, b = np.asarray(info["A"]), np.asarray(info["B"])
    p = solve_discrete_are(a, b, np.diag(q), np.diag(r))
    gain = np.linalg.solve(np.diag(r) + b.T @ p @ b, b.T @ p @ a)
    base.initialize_network(
        agent.actor, -gain / base.BOUND, scale=settings.embedding_scale
    )
    base.initialize_network(agent.critic, 2 * p, scale=settings.embedding_scale)
    base.initialize_network(
        agent.plant_model, np.column_stack([a, b]), scale=settings.embedding_scale
    )
    info.update(
        config=asdict(agent.cfg),
        K_initial=gain.tolist(),
        P_initial=p.tolist(),
        tuning=asdict(settings),
    )
    return agent, info


class PhysicsSubsteps:
    """Keep the 20 Hz controller while refining only the physical integrator."""

    def __init__(self, env, count):
        self.env, self.count = env, count

    def step(self, action):
        for _ in range(self.count):
            transition = self.env.step(action)
            if transition[2] or transition[3]:
                break
        return transition

    @property
    def damage_events_log(self):
        return self.env.damage_events_log


def run_case(
    settings: Tuning,
    *,
    algorithm="etdhp",
    fault=False,
    warmup=60.0,
    response=300.0,
    channel=0,
    amplitude=None,
    seed=11,
    fault_time=30.0,
    dt=0.05,
    step_time=10.0,
    engine_fraction=0.5,
    engine_id=1,
    substeps=1,
):
    protocol = steps.Protocol(
        warmup_s=warmup,
        step_time_s=step_time,
        response_s=response,
        fault_time_s=fault_time,
        seed=seed,
        dt=dt,
        tail_s=min(50.0, response / 3),
        confirmation_s=min(30.0, response / 3),
    )
    template, info = make_agent(settings, dt=dt, seed=seed)
    state = steps.TrialState(protocol, algorithm, fault=fault, template=template)
    state.cfg = replace(
        state.cfg,
        engine_fraction=engine_fraction,
        engine_id=engine_id,
        substeps=substeps,
    )
    env = base.make_env(state.cfg, fault=fault)
    state.obs, _ = env.reset(seed=seed)
    state.env = PhysicsSubsteps(env, substeps)
    initial_weights = state.weights()
    for _ in range(round(warmup / dt)):
        try:
            state.advance(steps.NOMINAL_REFERENCE)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return np.asarray([state.record(0.0, steps.NOMINAL_REFERENCE)]), {
                "status": "failed",
                "reason": str(exc),
                "stage": "warmup",
                "tuning": asdict(settings),
                "protocol": asdict(protocol),
                "algorithm": algorithm,
                "fault": fault,
                "channel": channel,
                "steps_completed": state.step,
            }
    start_weights = state.weights()
    start_triggers = state.triggers
    amplitude = steps.STEP_AMPLITUDES[channel] if amplitude is None else amplitude

    def reference(t):
        ref = steps.NOMINAL_REFERENCE.copy()
        if t >= protocol.step_time_s - 1e-9:
            ref[channel] += amplitude
        return ref

    rows = [state.record(0.0, reference(0.0))]
    status, reason = "complete", None
    for i in range(round(protocol.trial_s / dt)):
        try:
            state.advance(reference(i * dt))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            status, reason = "failed", str(exc)
            break
        t = (i + 1) * dt
        rows.append(state.record(t, reference(t)))
    trace = np.asarray(rows)
    result = dict(
        status=status,
        reason=reason,
        tuning=asdict(settings),
        protocol=asdict(protocol),
        algorithm=algorithm,
        fault=fault,
        channel=channel,
        amplitude=float(amplitude),
        engine_fraction=engine_fraction,
        engine_id=engine_id,
        substeps=substeps,
        triggers=state.triggers - start_triggers,
        events=state.env.damage_events_log,
        weight_change_after_warmup=[
            float(np.linalg.norm(a - b)) for a, b in zip(state.weights(), start_weights)
        ],
        weight_change_total=[
            float(np.linalg.norm(a - b))
            for a, b in zip(state.weights(), initial_weights)
        ],
    )
    if status == "complete":
        m = steps.step_metrics(
            trace[:, 0],
            trace[:, 2 + channel],
            step_time=protocol.step_time_s,
            initial_reference=steps.NOMINAL_REFERENCE[channel],
            final_reference=steps.NOMINAL_REFERENCE[channel] + amplitude,
            tail_seconds=protocol.tail_s,
            confirmation_seconds=protocol.confirmation_s,
        )
        post = trace[:, 0] >= protocol.step_time_s
        time = trace[post, 0] - protocol.step_time_s
        errors = trace[post, 2:6] - trace[post, 6:10]
        result["metrics"] = m
        result["all_channel_iae"] = np.trapz(np.abs(errors), time, axis=0).tolist()
        result["all_channel_rmse"] = np.sqrt(
            np.trapz(errors**2, time, axis=0) / response
        ).tolist()
        result["command_energy"] = float(
            np.trapz(np.sum(trace[post, 11:13] ** 2, axis=1), time)
        )
        result["command_peak"] = np.max(np.abs(trace[post, 11:13]), axis=0).tolist()
        result["command_total_variation"] = np.sum(
            np.abs(np.diff(trace[post, 11:13], axis=0)), axis=0
        ).tolist()
        result["beta_peak"] = float(np.max(np.abs(trace[post, 10])))
        result["roll_peak"] = float(np.max(np.abs(trace[post, 3])))
        result["trigger_fraction"] = result["triggers"] / (protocol.trial_s / dt)
    return trace, result


def tuning_score(result):
    """Preset healthy cost: tracking, cross-axis excursions, tail, effort, events."""
    if result["status"] != "complete":
        return 1e6
    m = result["metrics"]
    amp = abs(result["amplitude"])
    other = 1 - result["channel"]
    settle = m["settling_2pct_s"]
    return (
        m["iae"] / amp
        + 0.5 * result["all_channel_iae"][other] / amp
        + 0.25 * m["ise"] / amp**2
        + 0.15 * (settle if settle is not None else result["protocol"]["response_s"])
        + 1.2 * max(m["overshoot_pct"] - 8, 0)
        + 1000 * (abs(m["tail_bias"]) + m["tail_std"]) / amp
        + 0.008 * result["command_energy"]
        + 2 * result["trigger_fraction"]
    )


def save_case(path, trace, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path.with_suffix(".csv"),
        trace,
        delimiter=",",
        header=",".join(steps.TRACE_COLUMNS),
        comments="",
    )
    path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


# These additional cases were fixed before evaluating the balanced profile.
# They differ from the earlier diagnostic cases (engine ID, time, amplitude).
HELD_OUT_CASES = {
    "right_full_heading": dict(
        channel=0,
        amplitude=-0.75,
        fault=True,
        warmup=350.0,
        fault_time=93.5,
        engine_id=4,
        engine_fraction=0.0,
    ),
    "right_full_roll": dict(
        channel=1,
        amplitude=-0.2,
        fault=True,
        warmup=350.0,
        fault_time=93.5,
        engine_id=4,
        engine_fraction=0.0,
    ),
    "inboard_heading": dict(
        channel=0,
        amplitude=1.25,
        fault=True,
        warmup=450.0,
        fault_time=287.5,
        engine_id=2,
        engine_fraction=0.3,
    ),
    "inboard_roll": dict(
        channel=1,
        amplitude=0.35,
        fault=True,
        warmup=450.0,
        fault_time=287.5,
        engine_id=2,
        engine_fraction=0.3,
    ),
    "fine_physics_heading": dict(
        channel=0,
        amplitude=1.0,
        fault=True,
        warmup=1000.0,
        fault_time=137.0,
        substeps=2,
    ),
    "fine_physics_roll": dict(
        channel=1,
        amplitude=0.5,
        fault=True,
        warmup=1000.0,
        fault_time=137.0,
        substeps=2,
    ),
}
HOLD_CASES = {
    "healthy_hold": dict(duration=500.0, fault_time=137.0),
    "standard_hold": dict(duration=500.0, fault_time=137.0),
    "early_full_hold": dict(
        duration=500.0,
        fault_time=20.0,
        engine_fraction=0.0,
        initial_heading_deg=2.0,
        initial_roll_deg=1.0,
    ),
}


def runtime_digest():
    """Hash relevant syntax trees without relying on compiled line numbers."""
    import ast
    import hashlib

    parts = [steps.source_digest()]
    names = {"Tuning", "make_agent", "PhysicsSubsteps", "run_case"}
    module = ast.parse(Path(__file__).read_text())
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            parts.append(ast.dump(node, include_attributes=False))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _validation_job(job):
    name, algorithm, kind, kwargs, output, rerun = job
    torch.set_num_threads(1)
    path = Path(output) / name
    request = dict(
        algorithm=algorithm,
        kind=kind,
        kwargs=kwargs,
        tuning=asdict(TUNED),
        source_digest=runtime_digest(),
    )
    # JSON normalization makes tuple/list equality independent of serialization.
    request = json.loads(json.dumps(request))
    if (
        not rerun
        and path.with_suffix(".json").exists()
        and path.with_suffix(".csv").exists()
    ):
        cached = json.loads(path.with_suffix(".json").read_text())
        if cached.get("request") == request:
            return name, cached
    if kind == "step":
        trace, result = run_case(
            TUNED, algorithm=algorithm, response=500.0, step_time=20.0, **kwargs
        )
        result["request"] = request
        save_case(path, trace, result)
    else:
        cfg = base.Experiment(**kwargs)
        agent, _ = make_agent(TUNED)
        trace, diagnostics = base.rollout(
            cfg,
            algorithm,
            fault=not name.startswith("healthy_hold"),
            gains=base.NOMINAL_PID_GAINS,
            template=agent,
        )
        result = dict(
            **diagnostics,
            experiment=asdict(cfg),
            metrics=base.metrics(trace, cfg),
            algorithm=algorithm,
            kind=kind,
            request=request,
        )
        np.savetxt(
            path.with_suffix(".csv"),
            trace,
            delimiter=",",
            header=",".join(base.TRACE_NAMES),
            comments="",
        )
        path.with_suffix(".json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    return name, result


def run_validation(output, *, stage="standard", workers=2, rerun=False):
    """Execute reproducible standard steps or independent stress/physics checks."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    if stage == "standard":
        for fault in (False, True):
            for algorithm in ("etdhp", "pid"):
                for channel in range(4):
                    name = (
                        ("fault" if fault else "healthy")
                        + "_"
                        + algorithm
                        + "_"
                        + steps.CHANNELS[channel]
                    )
                    kwargs = dict(
                        channel=channel, fault=fault, warmup=1000.0, fault_time=137.0
                    )
                    jobs.append((name, algorithm, "step", kwargs, output, rerun))
    elif stage == "held_out":
        for name, kwargs in HELD_OUT_CASES.items():
            for algorithm in ("etdhp", "pid"):
                jobs.append(
                    (name + "_" + algorithm, algorithm, "step", kwargs, output, rerun)
                )
        for name, kwargs in HOLD_CASES.items():
            for algorithm in ("etdhp", "pid"):
                jobs.append(
                    (name + "_" + algorithm, algorithm, "hold", kwargs, output, rerun)
                )
    else:
        raise ValueError("stage must be standard or held_out")
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "parameters": asdict(TUNED),
                "jobs": [
                    dict(name=j[0], algorithm=j[1], kind=j[2], kwargs=j[3])
                    for j in jobs
                ],
            },
            indent=2,
        )
        + "\n"
    )
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_validation_job, job) for job in jobs]):
            name, result = future.result()
            results[name] = result
            print(
                name,
                result["status"],
                result.get("metrics", {}).get("settling_2pct_s"),
                flush=True,
            )
            (output / "summary.json").write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n"
            )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["standard", "held_out"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("outputs/b747-etdhp-vs-pid/tuning/final") / args.stage
    run_validation(output, stage=args.stage, workers=args.workers, rerun=args.rerun)


if __name__ == "__main__":
    main()
