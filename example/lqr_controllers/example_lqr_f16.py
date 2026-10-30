"""Discrete LQR tracking on the nonlinear longitudinal F-16.

Run from the repository root:
    python -m example.lqr_controllers.example_lqr_f16 --output outputs/lqr-f16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

import tensoraerospace.envs  # noqa: F401 — registers gym envs
from example.reinforcement_learning.incremental_adp import (
    example_etdhp_damage_f16 as environment,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.model import (
    LongitudinalF16,
)
from tensoraerospace.agent.lqr import (
    LQRAgent,
    LQRMRACAgent,
    LQRMRACConfig,
    LQRPMRACAgent,
    LQRPMRACConfig,
)


@dataclass(frozen=True)
class F16Setup:
    alpha_trim_rad: float
    stab_trim_rad: float
    alphas_grid_deg: np.ndarray
    stabs_grid_deg: np.ndarray

    @property
    def feedforward(self):
        return environment.make_feedforward_fn(
            None,
            np.rad2deg(self.alpha_trim_rad),
            self.alphas_grid_deg,
            self.stabs_grid_deg,
        )


def make_setup():
    alpha_trim, stab_trim = environment.compute_trim()
    alphas, stabs = environment.build_feedforward_table(np.rad2deg(alpha_trim))
    return F16Setup(alpha_trim, stab_trim, alphas, stabs)


def residual_transition(setup, state_deg, action_deg):
    physical = np.array(
        [
            setup.alpha_trim_rad + np.deg2rad(state_deg[0]),
            np.deg2rad(state_deg[1]),
            setup.stab_trim_rad + np.deg2rad(state_deg[2]),
            np.deg2rad(state_deg[3]),
        ]
    )
    model = LongitudinalF16(physical, dt=DT, integrator="euler")
    control = setup.stab_trim_rad + np.deg2rad(action_deg)
    next_physical = model.run_step(np.array([control])).reshape(-1)
    return np.rad2deg(next_physical) - np.array(
        [np.rad2deg(setup.alpha_trim_rad), 0.0, np.rad2deg(setup.stab_trim_rad), 0.0]
    )


def make_lqr(setup, q_diag=(0.1, 10.0, 0.0, 0.0), r_weight=0.1):
    zero = np.zeros(4)
    h = 1e-3
    a = np.column_stack(
        [
            (
                residual_transition(setup, h * basis, 0.0)
                - residual_transition(setup, -h * basis, 0.0)
            )
            / (2 * h)
            for basis in np.eye(4)
        ]
    )
    b = np.column_stack(
        [
            (residual_transition(setup, zero, h) - residual_transition(setup, zero, -h))
            / (2 * h)
        ]
    )
    q = np.diag(q_diag)
    r = np.diag([r_weight])
    agent = LQRAgent(a, b, q, r)
    output = np.array([[0.0, 1.0, 0.0, 0.0]])
    dc_gain = output @ np.linalg.solve(np.eye(4) - a + b @ agent.K, b)
    prefilter = 1.0 / dc_gain[0, 0]
    return agent, {
        "A": a.tolist(),
        "B": b.tolist(),
        "K": agent.K.tolist(),
        "q_reference_scale": float(prefilter / agent.K[0, 1]),
        "spectral_radius": float(max(abs(np.linalg.eigvals(a - b @ agent.K)))),
        "Q": q.tolist(),
        "R": r.tolist(),
    }


DT = 0.02
MRAC_GAIN = 0.002
PMRAC_GAIN = 0.002
MRAC_DEADZONE = 1e-4
TRACE_COLUMNS = (
    "time_s",
    "q_reference_deg_s",
    "q_deg_s",
    "alpha_deg",
    "surface_deg",
    "surface_rate_deg_s",
    "residual_command_deg",
    "applied_surface_delta_deg",
    "reference_error_norm",
    "adaptive_command_deg",
    "parameter_norm",
    "prediction_error_norm",
)


@dataclass(frozen=True)
class Experiment:
    duration: float = 20.0
    amplitude_deg: float = 2.0
    frequency_hz: float = 0.1
    phase: float = 0.0
    fault_time: float = 10.0
    loss_fraction: float = 0.0
    fault_kind: str = "section_loss"
    seed: int = 1

    @property
    def steps(self):
        return round(self.duration / DT)


def q_reference(cfg):
    time = np.arange(cfg.steps + 1) * DT
    q = (
        cfg.amplitude_deg
        / 20.0
        * (
            0.5 * np.sin(2 * np.pi * cfg.frequency_hz * time + cfg.phase)
            + 0.15 * np.sin(6.2 * np.pi * cfg.frequency_hz * time + cfg.phase)
        )
    )
    return np.deg2rad(q)


def damage_profile(cfg):
    if cfg.loss_fraction <= 0:
        return None
    if cfg.fault_kind == "actuator":
        return DamageProfile(
            events=[
                DamageEvent(
                    trigger_time=cfg.fault_time,
                    event_type="control_failure",
                    payload={
                        "surface": "stab_left",
                        "mode": "efficiency_loss",
                        "efficiency": 1.0 - cfg.loss_fraction,
                    },
                    label="collective_stabilator_efficiency_loss",
                )
            ]
        )
    if cfg.fault_kind != "section_loss":
        raise ValueError(f"Unknown fault kind: {cfg.fault_kind}")
    return DamageProfile(
        events=[
            DamageEvent(
                trigger_time=cfg.fault_time,
                event_type="section_loss",
                payload={"section": section, "loss_fraction": cfg.loss_fraction},
                label=f"{section}_{cfg.loss_fraction:g}",
            )
            for section in ("left_tip", "right_tip")
        ]
    )


def make_controller(setup, method, q_diag=(0.1, 10.0, 0.0, 0.0), r_weight=0.1):
    baseline, model = make_lqr(setup, q_diag=q_diag, r_weight=r_weight)
    if method == "lqr":
        return baseline, model
    if method.startswith("pmrac"):
        agent = LQRPMRACAgent(
            baseline,
            LQRPMRACConfig(
                dt=DT,
                adaptation_gain=PMRAC_GAIN,
                deadzone=MRAC_DEADZONE,
                normalization=1.0,
                parameter_limit=20.0,
                adaptive_limit=1.0,
                adaptation_enabled=True,
            ),
        )
    else:
        agent = LQRMRACAgent(
            baseline,
            LQRMRACConfig(
                dt=DT,
                adaptation_gain=MRAC_GAIN,
                deadzone=MRAC_DEADZONE,
                normalization=1.0,
                parameter_limit=20.0,
                adaptive_limit=1.0,
                adaptation_enabled=True,
            ),
        )
    return agent, model


def rollout(
    setup,
    cfg,
    method,
    reference_override=None,
    q_diag=(0.1, 10.0, 0.0, 0.0),
    r_weight=0.1,
):
    methods = ("lqr", "mrac_frozen", "mrac_online", "pmrac_frozen", "pmrac_online")
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    agent, model = make_controller(setup, method, q_diag=q_diag, r_weight=r_weight)
    initial = np.array([setup.alpha_trim_rad, 0.0, setup.stab_trim_rad, 0.0])
    alpha_reference = np.full((1, cfg.steps + 1), setup.alpha_trim_rad)
    reference = (
        q_reference(cfg)
        if reference_override is None
        else np.asarray(reference_override)
    )
    if reference.ndim != 1 or len(reference) < cfg.steps + 1:
        raise ValueError("reference_override must contain at least steps + 1 samples")
    env = gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=cfg.steps + 2,
        initial_state=[setup.alpha_trim_rad, 0.0, setup.stab_trim_rad, 0.0],
        reference_signal=alpha_reference,
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=DT,
        integrator="rk4",
        feedforward_fn=setup.feedforward,
        damage_profile=damage_profile(cfg),
    ).unwrapped
    observation, _ = env.reset(seed=cfg.seed)
    state = initial.copy()
    if hasattr(agent, "reset"):
        agent.reset(np.zeros(4))
    rows = []
    previous_command = 0.0
    try:
        for k in range(cfg.steps):
            if method.endswith("_frozen") and k == round(cfg.fault_time / DT):
                agent.cfg.adaptation_enabled = False
            state_deg = np.array(
                [
                    np.rad2deg(state[0] - setup.alpha_trim_rad),
                    np.rad2deg(state[1]),
                    np.rad2deg(state[2] - setup.stab_trim_rad),
                    np.rad2deg(state[3]),
                ]
            )
            reference_deg = np.array([0.0, np.rad2deg(reference[k]), 0.0, 0.0])
            reference_deg[1] *= model["q_reference_scale"]
            command = float(agent.predict(state_deg, reference_deg)[0])
            command = float(
                np.clip(command, previous_command - 60 * DT, previous_command + 60 * DT)
            )
            command = float(np.clip(command, -10.0, 10.0))
            previous_surface = state[2]
            observation, _, terminated, truncated, _ = env.step(np.array([command]))
            next_state = env.model.current_state.copy()
            applied = 0.5 * (previous_surface + next_state[2])
            next_deg = np.array(
                [
                    np.rad2deg(next_state[0] - setup.alpha_trim_rad),
                    np.rad2deg(next_state[1]),
                    np.rad2deg(next_state[2] - setup.stab_trim_rad),
                    np.rad2deg(next_state[3]),
                ]
            )
            if hasattr(agent, "update"):
                agent.update(state_deg, next_deg, reference_deg)
            diagnostics = (
                agent.diagnostics
                if hasattr(agent, "diagnostics")
                else {
                    "reference_error": np.zeros(4),
                    "adaptive_command": np.zeros(1),
                    "parameter_norm": 0.0,
                }
            )
            if not np.isfinite(next_state).all():
                raise FloatingPointError(f"Nonfinite state at {k}")
            if abs(next_state[1]) > np.deg2rad(10) or abs(
                next_state[0] - initial[0]
            ) > np.deg2rad(10):
                raise RuntimeError(f"Left the demonstration flight envelope at {k}")
            if (terminated or truncated) and k + 1 < cfg.steps:
                raise RuntimeError(f"Environment ended early at {k}")
            rows.append(
                [
                    (k + 1) * DT,
                    np.rad2deg(reference[k + 1]),
                    next_deg[1],
                    np.rad2deg(next_state[0]),
                    np.rad2deg(next_state[2]),
                    next_deg[3],
                    command,
                    np.rad2deg(applied - setup.stab_trim_rad),
                    np.linalg.norm(diagnostics["reference_error"]),
                    np.rad2deg(diagnostics["adaptive_command"][0]),
                    diagnostics["parameter_norm"],
                    np.linalg.norm(diagnostics.get("prediction_error", np.zeros(4))),
                ]
            )
            state = next_state
            previous_command = command
        status = "complete"
        reason = None
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        status, reason = "failed", str(exc)
    finally:
        env.close()
    report = {
        "experiment": asdict(cfg),
        "method": method,
        "status": status,
        "reason": reason,
        "model": model,
        "trace_columns": list(TRACE_COLUMNS),
    }
    return np.asarray(rows).reshape(-1, len(TRACE_COLUMNS)), report


def metrics(trace):
    if not len(trace):
        return {"q_rmse_deg_s": None, "command_variation_deg": None}
    late = trace[len(trace) // 2 :, 2] - trace[len(trace) // 2 :, 1]
    return {
        "q_rmse_deg_s": float(np.sqrt(np.mean(late**2))),
        "command_variation_deg": float(np.sum(np.abs(np.diff(trace[:, 6])))),
    }


def save_results(output, traces, reports):
    output.mkdir(parents=True, exist_ok=True)
    for method, trace in traces.items():
        np.savez_compressed(output / f"{method}.npz", trace=trace)
        reports[method]["metrics"] = metrics(trace)
    (output / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, constrained_layout=True
    )
    for method, trace in traces.items():
        if not len(trace):
            continue
        axes[0].plot(trace[:, 0], trace[:, 2], label=method, lw=1.5)
        axes[1].plot(trace[:, 0], trace[:, 2] - trace[:, 1], label=method, lw=1.5)
        axes[2].plot(trace[:, 0], trace[:, 6], label=method, lw=1.5)
    reference = next(trace for trace in traces.values() if len(trace))
    axes[0].plot(reference[:, 0], reference[:, 1], "k--", lw=1.2, label="Задание")
    axes[0].set_ylabel("q, град/с")
    axes[1].set_ylabel("Ошибка q, град/с")
    axes[2].set_ylabel("Команда, град")
    axes[2].set_xlabel("Время, с")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/lqr-f16"))
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    setup = make_setup()
    traces, reports = {}, {}
    for fault in (False, True):
        cfg = Experiment(
            duration=args.duration,
            fault_time=args.duration / 2,
            loss_fraction=0.3 if fault else 0.0,
        )
        method = "lqr_fault" if fault else "lqr_nominal"
        trace, report = rollout(setup, cfg, "lqr")
        traces[method], reports[method] = trace, report
    save_results(args.output, traces, reports)


if __name__ == "__main__":
    main()
