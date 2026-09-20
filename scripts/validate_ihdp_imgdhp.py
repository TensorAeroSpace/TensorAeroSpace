"""Numerical IM-GDHP regression using the public SDK (not a flight certificate).

IHDP reproducible cases live in example_ihdp_nonlinear_{f16,b737}.ipynb.
This CI/research utility records every evaluated episode, final policies,
held-out signals, physical envelopes, and failures without selecting a best
checkpoint. The example notebook contains the complete equivalent SDK loop.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import tensoraerospace  # noqa: F401 -- registers Gymnasium environments
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig
from tensoraerospace.benchmark import ControlBenchmark

DT = 0.01
N = 1200
TIME = np.arange(N) * DT
REFERENCE = np.deg2rad(2 * np.sin(2 * np.pi * TIME / 4))[None, :]


def make_env(reference):
    return gym.make(
        "LinearLongitudinalF16-v0",
        initial_state=np.zeros(4),
        reference_signal=reference,
        number_time_steps=reference.shape[1],
    ).unwrapped


def run_episode(agent, reference, *, learning, noise=0.0):
    env = make_env(reference)
    obs, _ = env.reset()
    agent.reset()
    agent.cfg.exploration_noise_std = noise
    observations, commands = [], []
    try:
        for k in range(reference.shape[1] - 1):
            action = agent.predict(
                np.asarray(obs).ravel(), reference, k, deterministic=noise == 0
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if not np.isfinite(obs).all() or abs(
                float(np.asarray(obs).ravel()[0])
            ) > np.deg2rad(25):
                raise RuntimeError(f"alpha envelope exceeded at step {k}")
            if learning:
                metrics = agent.learn(np.asarray(obs).ravel(), reference, k)
                for key in ("critic_loss", "actor_loss"):
                    if (
                        k > 0
                        and agent._total_steps
                        > agent.cfg.warmup_steps + agent.cfg.critic_only_steps + 1
                        and not np.isfinite(metrics[key])
                    ):
                        raise RuntimeError(f"nonfinite {key} at step {k}")
            observations.append(np.asarray(obs).ravel().copy())
            commands.append(action.copy())
            if terminated or truncated:
                break
    finally:
        env.close()
    y, u = np.asarray(observations), np.asarray(commands)
    err = np.rad2deg(y[:, 0] - reference[0, 1 : len(y) + 1])
    return (
        {
            "rmse_deg": float(np.sqrt(np.mean(err**2))),
            "mae_deg": float(np.mean(np.abs(err))),
            "peak_alpha_deg": float(np.max(np.abs(np.rad2deg(y[:, 0])))),
            "peak_rate_deg_s": float(np.max(np.abs(np.rad2deg(y[:, 1])))),
            "peak_command_deg": float(np.max(np.abs(u))),
            "saturation_fraction": float(np.mean(np.abs(u) >= 0.99 * agent.cfg.u_max)),
            "samples": len(y),
        },
        y,
        u,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--actor-lr", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=0.01)
    parser.add_argument("--u-max", type=float, default=3.0)
    parser.add_argument("--costate-ratio", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=0.99995)
    parser.add_argument("--training-signal", choices=("sine", "step"), default="sine")
    parser.add_argument(
        "--q-weight",
        type=float,
        default=None,
        help="Include measured pitch-rate error with this physical cost weight",
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--cov-init", type=float, default=1e4)
    parser.add_argument("--weight-limit", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        scale = 180 / np.pi
        cfg = IMGDHPConfig(
            actor_hidden=(16,),
            critic_hidden=(16,),
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            actor_lr_decay=args.decay,
            critic_lr_decay=args.decay,
            actor_lr_min=1e-5,
            critic_lr_min=1e-5,
            track_Q=(
                (200 / scale**2,)
                if args.q_weight is None
                else (200 / scale**2, args.q_weight / scale**2)
            ),
            beta_lambda=args.costate_ratio / scale**2,
            history_length=4,
            obs_scale=(scale, scale),
            gamma=args.gamma,
            warmup_steps=200,
            critic_only_steps=400,
            max_grad_norm=5.0,
            cov_init=args.cov_init,
            weight_limit=args.weight_limit,
            forgetting=0.999,
            u_max=args.u_max,
            seed=seed,
        )
        indices = [0] if args.q_weight is None else [0, 1]
        agent = IMGDHPAgent(
            2, 1, reference_size=len(indices), tracking_indices=indices, config=cfg
        )

        def channels(alpha):
            return (
                alpha if len(indices) == 1 else np.vstack((alpha, np.zeros_like(alpha)))
            )

        training_reference = channels(
            REFERENCE
            if args.training_signal == "sine"
            else np.deg2rad(np.where(TIME >= 2.0, 1.0, 0.0))[None, :]
        )
        record = {
            "seed": seed,
            "tracking_indices": indices,
            "training_signal": args.training_signal,
            "config": dataclasses.asdict(cfg),
            "episodes": [],
            "evaluation": [],
        }
        try:
            for ep in range(args.episodes):
                metrics, _, _ = run_episode(
                    agent, training_reference, learning=True, noise=2 if ep < 2 else 0.3
                )
                record["episodes"].append(metrics)
                if ep in (1, 4, 9) or (ep + 1) % 10 == 0 or ep + 1 == args.episodes:
                    evaluation_reference = channels(
                        np.deg2rad(np.where(np.arange(2000) * DT >= 2.0, 1.0, 0.0))[
                            None, :
                        ]
                    )
                    result, evaluation_y, _ = run_episode(
                        agent, evaluation_reference, learning=False
                    )
                    result["benchmark"] = ControlBenchmark().benchmarking_step_response(
                        evaluation_reference[0, 1 : len(evaluation_y) + 1],
                        evaluation_y[:, 0],
                        0.0,
                        DT,
                    )
                    record["evaluation"].append({"episode": ep + 1, **result})
                    print(seed, ep + 1, result, flush=True)
            result, y, u = run_episode(agent, training_reference, learning=False)
            record["final"] = result
            np.savez_compressed(
                args.output.with_name(f"{args.output.stem}-seed{seed}.npz"),
                time=TIME[1 : len(y) + 1],
                reference=training_reference[0, 1 : len(y) + 1],
                observation=y,
                command=u,
            )
            test_time = np.arange(2000) * DT
            step_reference = np.deg2rad(np.where(test_time >= 2.0, 1.0, 0.0))[None, :]
            record["additional_step"], step_y, _ = run_episode(
                agent, channels(step_reference), learning=False
            )
            transient = ControlBenchmark().benchmarking_step_response(
                step_reference[0, 1 : len(step_y) + 1], step_y[:, 0], 0.0, DT
            )
            record["additional_step"]["benchmark"] = transient
            online_metrics, online_y, _ = run_episode(
                copy.deepcopy(agent), channels(step_reference), learning=True
            )
            online_metrics["benchmark"] = ControlBenchmark().benchmarking_step_response(
                step_reference[0, 1 : len(online_y) + 1], online_y[:, 0], 0.0, DT
            )
            record["step_online"] = online_metrics
            record["additional_step"]["command_overshoot_percent"] = transient[
                "command_overshoot"
            ]
            record["additional_step"]["command_settling_time_5percent_s"] = transient[
                "command_settling_time"
            ]
            held_out = np.deg2rad(1.5 * np.sin(2 * np.pi * TIME / 5 + 0.4))[None, :]
            record["held_out"], _, _ = run_episode(
                agent, channels(held_out), learning=False
            )
            long_time = np.arange(6000) * DT
            long_ref = np.deg2rad(2 * np.sin(2 * np.pi * long_time / 4))[None, :]
            record["long_frozen"], _, _ = run_episode(
                agent, channels(long_ref), learning=False
            )
            record["long_online"], _, _ = run_episode(
                copy.deepcopy(agent), channels(long_ref), learning=True
            )
            mixed_reference = np.deg2rad(
                0.8 * np.sin(2 * np.pi * test_time / 3.7)
                + 0.4 * np.cos(2 * np.pi * test_time / 6.3)
                - 0.4
            )[None, :]
            record["additional_multisine"], _, _ = run_episode(
                agent, channels(mixed_reference), learning=False
            )
            record["checkpoint"] = agent.save(
                args.output.parent / f"{args.output.stem}-seed{seed}",
                save_gradients=True,
            )
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            record["failure"] = str(exc)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(
            "RESULT",
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("episodes", "config")
                }
            ),
            flush=True,
        )
    return int(any("failure" in record for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
