"""Reproduce AIDI sensor/actuator alignment and nonlinear F-16 rollouts.

A fixed-parameter controller is a diagnostic baseline from episode start.
The adaptive controller never receives fault time or severity. Native F-16
actuators and damage events remain active. Limits below are conservative
experiment validity guards, not a certified aircraft flight envelope.
Exit status is 1 when any requested rollout fails its guards or horizon.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (  # noqa: E402
    stab_efficiency_step,
)
from tensoraerospace.agent.aidi.utils import reconstruct_n_z  # noqa: E402
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16  # noqa: E402
from tensoraerospace.scripts.benchmark_aidi import (  # noqa: E402
    _build_agent,
    _solve_trim,
)


def observation(x, speed, *, legacy=False):
    return dict(
        omega=x[[2, 4, 3]] * [1, 1, 1 if legacy else -1],
        alpha=float(x[0]),
        beta=float(x[1]),
        theta=float(x[7]),
        phi=float(x[5]),
        V=speed,
        state=x.copy(),
    )


def rollout(version, seed, scenario, duration, physics_dt, baseline=None):
    control_dt = 0.01
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt must be finite and positive")
    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("physics_dt must divide 0.01 s")
    rng = np.random.default_rng(seed)
    alpha, stab = _solve_trim()
    x = np.zeros(14)
    x[0] = x[7] = alpha
    x[8] = stab
    x[[0, 1, 5, 7]] += np.deg2rad(rng.uniform(-0.1, 0.1, 4))
    x[[2, 3, 4]] += np.deg2rad(rng.uniform(-0.01, 0.01, 3))
    agent = _build_agent("adaptive")
    if version == "before":
        agent = baseline.AIDIAgent(
            3,
            3,
            baseline.F16NonlinearOnboardCE(),
            baseline.AIDIConfig(
                **{**dataclasses.asdict(agent.cfg), "rate_kp": (0.0, 0.0, 0.0)}
            ),
        )
        # Give both implementations the same physical trim; old API has no
        # initial_action argument. No weights/formulas in the snapshot change.
        agent.reset()
        agent._u_prev = x[[8, 10, 12]].copy()
        agent._last_u_cmd = agent._u_prev.copy()
    else:
        agent.reset(initial_action=x[[8, 10, 12]])
    n = round(duration / control_dt)
    fault_time = 17.35
    env = NonlinearAngularF16(
        initial_state=x,
        number_time_steps=n * substeps + 2,
        dt=physics_dt,
        integrator="rk4",
        airspeed=120.0,
        damage_profile=(
            stab_efficiency_step(fault_time, mu=0.75)
            if scenario == "command_loss_25pct"
            else None
        ),
    )
    x, _ = env.reset()
    speed = float(env.model.param.V)
    theta0 = agent.rls.theta.copy()
    trace = []
    status, failure, events = "completed", None, []
    updates_at_fault = None
    for k in range(n):
        t = k * control_dt
        # Small, repeatable manoeuvre, identical for all versions and seeds.
        phi_ref = np.deg2rad(2.0) * np.sin(2 * np.pi * t / 40.0)
        refs = dict(C_star=1.0, phi_cmd=phi_ref, beta_cmd=0.0, V_cmd=speed)
        old_x = x.copy()
        try:
            command = agent.predict(
                observation(x, speed, legacy=version == "before"), refs
            )
            applied_mean = np.zeros(3)
            for j in range(substeps):
                previous_actuators = x[[8, 10, 12]].copy()
                x, _, terminated, truncated, info = env.step(np.rad2deg(command))
                applied_mean += (previous_actuators + x[[8, 10, 12]]) / (2 * substeps)
                if info.get("damage_events_triggered"):
                    events.append(
                        dict(
                            time=t + (j + 1) * physics_dt,
                            events=info.get("damage_events_triggered"),
                        )
                    )
                if (terminated or truncated) and k < n - 1:
                    raise RuntimeError("environment ended before requested horizon")
            if not np.all(np.isfinite(x)):
                raise RuntimeError("non-finite plant state")
            if version == "before":
                agent.learn(observation(x, speed, legacy=True), refs)
            else:
                agent.learn(
                    observation(x, speed),
                    refs,
                    applied_action=applied_mean,
                    adapt=version != "fixed",
                )
            if not (
                np.isfinite(agent.rls.theta).all() and np.isfinite(agent.rls.P).all()
            ):
                raise RuntimeError("non-finite identifier")
            p = env.model.param
            if np.any(
                abs(x[[8, 10, 12]])
                > np.array([p.maxabsstab, p.maxabsail, p.maxabsdir]) + 1e-9
            ):
                raise RuntimeError("actuator position outside physical bounds")
            if np.any(
                abs(x[[9, 11, 13]])
                > np.array([p.maxabsdstab, p.maxabsdail, p.maxabsddir]) + 1e-9
            ):
                raise RuntimeError("actuator rate outside physical bounds")
            nz = reconstruct_n_z(
                x[0], (x[0] - old_x[0]) / control_dt, x[4], speed, x[7], x[5]
            )
            cstar = nz + speed / agent.cfg.cstar_V_co * x[4]
            trace.append(
                [
                    t + control_dt,
                    cstar - 1,
                    np.rad2deg(x[5] - phi_ref),
                    np.rad2deg(x[1]),
                    *np.rad2deg(x[[0, 5, 7, 2, 4, 3]]),
                    float(np.max(abs(agent.rls.theta - theta0))),
                ]
            )
            if updates_at_fault is None and t + control_dt >= fault_time:
                updates_at_fault = agent.rls.num_updates
            if not (-10 <= np.rad2deg(x[0]) <= 35):
                raise RuntimeError(
                    "alpha outside experiment validity bounds [-10,35] deg"
                )
            if np.any(abs(np.rad2deg(x[[1, 5, 7]])) > [15, 45, 45]):
                raise RuntimeError(
                    "beta/roll/pitch outside experiment validity bounds [15,45,45] deg"
                )
            if np.max(abs(np.rad2deg(x[[2, 3, 4]]))) > 100:
                raise RuntimeError("body rate exceeds 100 deg/s validity guard")
        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            status, failure = "failed", str(exc)
            break
    a = np.asarray(trace, dtype=float).reshape(-1, 11)
    metrics = {}
    for name, mask in [
        ("after_warmup", a[:, 0] >= 2),
        ("post_fault", a[:, 0] >= fault_time),
    ]:
        window = a[mask]
        metrics[name] = (
            None
            if not len(window)
            else dict(
                samples=len(window),
                cstar_rmse=float(np.sqrt(np.mean(window[:, 1] ** 2))),
                roll_rmse_deg=float(np.sqrt(np.mean(window[:, 2] ** 2))),
                beta_rmse_deg=float(np.sqrt(np.mean(window[:, 3] ** 2))),
            )
        )
    return dict(
        version=version,
        seed=seed,
        scenario=scenario,
        duration_requested=duration,
        physics_dt=physics_dt,
        control_dt=control_dt,
        status=status,
        failure=failure,
        completed_time=float(a[-1, 0]) if len(a) else 0,
        metrics=metrics,
        max_abs_alpha_roll_pitch_deg=(
            np.max(abs(a[:, 4:7]), axis=0).tolist() if len(a) else None
        ),
        theta_final=agent.rls.theta.tolist(),
        theta_change_max=float(np.max(abs(agent.rls.theta - theta0))),
        updates=agent.rls.num_updates,
        updates_after_fault=(
            None
            if updates_at_fault is None
            else agent.rls.num_updates - updates_at_fault
        ),
        fault_events=events,
        trace_columns=[
            "time",
            "cstar_error",
            "roll_error_deg",
            "beta_error_deg",
            "alpha_deg",
            "roll_deg",
            "pitch_deg",
            "p_deg_s",
            "q_deg_s",
            "wy_native_deg_s",
            "theta_change",
        ],
        trace=a[::10].tolist(),
    )


def angular_physics():
    """Cross-check angular env units/integration against DOP853, sharing the RHS."""
    from scipy.integrate import solve_ivp

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )

    alpha, stab = _solve_trim()
    results = []
    for dt in (0.02, 0.01, 0.005):
        x = np.zeros(14)
        x[0] = x[7] = alpha
        x[8] = stab
        n = round(2 / dt)
        env = NonlinearAngularF16(
            x, number_time_steps=n + 2, dt=dt, integrator="rk4", airspeed=120.0
        )
        env.reset()
        p = default_parameters()
        reference = x.copy()
        errors = []
        for k in range(n):
            u = np.array([stab, 0, 0]) + np.deg2rad([0.2, 0.1, -0.1]) * np.sin(
                2 * np.pi * 0.7 * k * dt
            )
            actual, _, term, trunc, _ = env.step(np.rad2deg(u))
            assert not term and not trunc
            solution = solve_ivp(
                lambda t, state: f16_ode_6dof(state, u, t, p),
                (k * dt, (k + 1) * dt),
                reference,
                method="DOP853",
                rtol=1e-11,
                atol=1e-12,
            )
            assert solution.success
            reference = solution.y[:, -1]
            np.testing.assert_array_equal(actual, env.model.current_state)
            assert np.isfinite(actual).all()
            errors.append(np.abs(actual - reference))
        error = np.max(errors, axis=0)
        results.append(
            dict(
                dt=dt,
                max_state_error=float(np.max(error)),
                max_body_rate_error=float(np.max(error[2:5])),
                steps=n,
            )
        )
    assert results[-1]["max_state_error"] < results[0]["max_state_error"] / 50
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--versions", default="after,fixed")
    parser.add_argument("--scenarios", default="nominal,command_loss_25pct")
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.physics_only:
        result = dict(angular_physics=angular_physics())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    if args.duration <= 2 or not np.isfinite(args.duration):
        parser.error("duration must be finite and greater than 2 s")
    baseline = None
    versions = args.versions.split(",")
    scenarios = args.scenarios.split(",")
    if any(v not in {"before", "after", "fixed"} for v in versions):
        parser.error("unknown version")
    if any(s not in {"nominal", "command_loss_25pct"} for s in scenarios):
        parser.error("unknown scenario")
    if "before" in versions:
        if args.baseline_package is None:
            parser.error(
                "before requires --baseline-package (an unmodified aidi package)"
            )
        sys.path.insert(0, str(args.baseline_package.resolve().parent))
        baseline = importlib.import_module(args.baseline_package.name)
    result = dict(
        cases=[],
        units="angles in degrees; C* dimensionless",
        faults="command gain, not aerodynamic coefficient loss",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for version in versions:
        for scenario in scenarios:
            for seed in map(int, args.seeds.split(",")):
                case = rollout(
                    version, seed, scenario, args.duration, args.physics_dt, baseline
                )
                result["cases"].append(case)
                args.output.write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in case.items()
                            if k not in {"trace", "trace_columns", "theta_final"}
                        }
                    ),
                    flush=True,
                )
    return int(any(case["status"] != "completed" for case in result["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
