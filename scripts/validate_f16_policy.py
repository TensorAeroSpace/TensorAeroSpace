"""Train/evaluate SAC pitch tracking on nonlinear F-16 with symmetric servo loss.

This is a bounded simulation benchmark, with true actuator effectiveness supplied
as an observation. It does not validate fault diagnosis or real-flight robustness.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.agent.sac import SAC
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    torch.set_num_threads(1)
    trim = find_trim(120.0, 3000.0)
    assert trim.converged
    trim_stab = float(np.rad2deg(trim.stab_rad))
    dt, horizon, residual_limit = 0.02, 3.0, 4.0

    class TrackingEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.steps = 0

        def target(self):
            return trim.alpha_rad + (
                np.deg2rad(self.amplitude) if self.elapsed >= self.start else 0.0
            )

        def effectiveness(self):
            failure = self.env.damage_manager.state.control_failures.get("stab_left")
            return 1.0 if failure is None else float(failure.efficiency)

        def observation(self):
            x = self.env.model.current_state
            return np.array(
                [
                    np.rad2deg(self.target() - x[7]) / 5,
                    np.rad2deg(x[4]) / 15,
                    np.rad2deg(x[0] - trim.alpha_rad) / 5,
                    (np.rad2deg(x[8]) - trim_stab) / 5,
                    np.rad2deg(self.target() - trim.alpha_rad) / 5,
                    (x[15] - 120) / 10,
                    self.effectiveness(),
                ],
                dtype=np.float32,
            )

        def reset(self, *, case=None, **kwargs):
            if case is None:
                amplitude = self.rng.uniform(-2.5, 2.5)
                start = self.rng.uniform(0.3, 0.8)
                effectiveness = (
                    1.0 if self.rng.random() < 0.4 else self.rng.uniform(0.6, 0.95)
                )
                failure_time = self.rng.uniform(0.6, 1.4)
            else:
                amplitude, start, effectiveness, failure_time = case
            self.amplitude, self.start, self.elapsed = amplitude, start, 0.0
            events = (
                []
                if effectiveness == 1
                else [
                    DamageEvent(
                        failure_time,
                        "control_failure",
                        {
                            "surface": side,
                            "mode": "efficiency_loss",
                            "efficiency": effectiveness,
                        },
                    )
                    for side in ["stab_left", "stab_right"]
                ]
            )
            self.env = NonlinearAngularF16(
                trim.x0,
                round(horizon / dt),
                dt=dt,
                integrator="rk4",
                split_stab=True,
                track_altitude=True,
                thrust_mode="control",
                damage_profile=DamageProfile(events),
            )
            self.env.reset()
            return self.observation(), {}

        def step(self, action):
            a = float(np.asarray(action).reshape(-1)[0])
            stab = trim_stab + residual_limit * np.clip(a, -1, 1)
            x, _, _, truncated, info = self.env.step(
                [stab, stab, 0.0, 0.0, trim.T_thrust]
            )
            self.steps += 1
            self.elapsed = self.env._step_index * dt
            if not np.isfinite(x).all():
                raise FloatingPointError("Nonfinite aircraft state")
            error = float(np.rad2deg(self.target() - x[7]))
            pitch_rate = float(np.rad2deg(x[4]))
            failed = bool(
                abs(np.rad2deg(x[7] - trim.alpha_rad)) > 20
                or abs(np.rad2deg(x[0])) > 35
                or abs(x[5]) > np.deg2rad(30)
                or not 60 < x[15] < 200
            )
            reward = -((error / 2) ** 2) - 0.01 * (pitch_rate / 5) ** 2 - 0.001 * a * a
            if failed:
                reward -= 10
            return self.observation(), float(reward), failed, bool(truncated), info

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = TrackingEnv(args.seed)
    agent = SAC(
        env,
        hidden_size=64,
        batch_size=128,
        memory_capacity=50000,
        seed=args.seed,
        alpha=0.01,
        log_every_updates=100,
        log_dir=args.output.with_suffix(""),
    )
    agent.writer.assert_contract_satisfied = lambda: None
    cases = [
        (amp, 0.5, eff, failure_time)
        for amp in [-2.0, -1.0, 1.0, 2.0]
        for eff, failure_time in [(1.0, 1.015), (0.75, 1.015), (0.6, 0.515)]
    ]

    def policy(obs):
        return agent.select_action(obs, evaluate=True)

    def pd(obs):
        desired = (trim_stab - 1.2 * obs[0] * 5 + 0.5 * obs[1] * 15) / obs[6]
        return np.array([np.clip((desired - trim_stab) / residual_limit, -1, 1)])

    def evaluate(controller):
        evaluation = TrackingEnv(20260917)
        errors, failures, traces, speeds, pitches = [], 0, [], [], []
        for case in cases:
            obs, _ = evaluation.reset(case=case)
            squares = []
            trace = []
            for _ in range(round(horizon / dt)):
                action = controller(obs)
                obs, _, term, trunc, _ = evaluation.step(action)
                squares.append(float(obs[0] * 5) ** 2)
                x = evaluation.env.model.current_state
                speeds.append(float(x[15]))
                pitches.append(float(abs(np.rad2deg(x[7] - trim.alpha_rad))))
                trace.append(
                    [
                        evaluation.elapsed,
                        float(np.rad2deg(x[7])),
                        float(np.rad2deg(evaluation.target())),
                        float(np.asarray(action).reshape(-1)[0]),
                    ]
                )
                if term or trunc:
                    failures += int(term)
                    break
            errors.append(float(np.sqrt(np.mean(squares))))
            traces.append(trace)
        return {
            "rmse_deg_mean": float(np.mean(errors)),
            "episode_rmse_deg": errors,
            "failed_episodes": failures,
            "episodes": len(cases),
            "speed_range_m_s": [min(speeds), max(speeds)],
            "max_pitch_offset_deg": max(pitches),
            "cases": cases,
            "traces": traces,
        }

    result = {
        "seed": args.seed,
        "repo": str(args.repo.resolve()),
        "config": {
            "frames": args.frames,
            "dt": dt,
            "horizon": horizon,
            "residual_limit_deg": residual_limit,
            "true_efficiency_observed": True,
        },
        "pd": evaluate(pd),
        "trim_control": evaluate(lambda obs: np.zeros(1)),
    }
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        agent.policy.load_state_dict(saved["policy"])
        result["evaluated"] = evaluate(policy)
        result["checkpoint"] = str(args.checkpoint)
    else:
        result["untrained"] = evaluate(policy)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        original_update = agent.update_parameters
        max_losses = np.zeros(5)

        def checked_update(*pos, **kw):
            losses = np.asarray(original_update(*pos, **kw))
            if not np.isfinite(losses).all():
                raise FloatingPointError("Nonfinite SAC loss")
            max_losses[:] = np.maximum(max_losses, np.abs(losses))
            return tuple(losses)

        agent.update_parameters = checked_update
        curves = []
        next_eval = 5000
        started = time.monotonic()
        while env.steps < args.frames:
            agent.train(
                num_episodes=1, max_steps=args.frames - env.steps, verbose=False
            )
            if env.steps >= next_eval:
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                metrics = evaluate(policy)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                curves.append(
                    {
                        "step": env.steps,
                        "rmse_deg": metrics["rmse_deg_mean"],
                        "failures": metrics["failed_episodes"],
                    }
                )
                print(json.dumps(curves[-1]), flush=True)
                next_eval += 5000
        result["elapsed_s"] = time.monotonic() - started
        result["evaluated"] = evaluate(policy)
        result["learning_curve"] = curves
        result["max_abs_losses"] = max_losses.tolist()
        result["environment_steps"] = env.steps
        result["parameters_finite"] = all(
            torch.isfinite(p).all().item()
            for network in [agent.policy, agent.critic]
            for p in network.parameters()
        )
        if not result["parameters_finite"]:
            raise FloatingPointError("Nonfinite parameters")
        torch.save(
            {"policy": agent.policy.state_dict(), "critic": agent.critic.state_dict()},
            args.output.with_suffix(".pt"),
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    agent.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in result["evaluated"].items()
                if k not in ["traces", "cases", "episode_rmse_deg"]
            }
        )
    )


if __name__ == "__main__":
    main()
