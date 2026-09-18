"""Train/evaluate SAC on ImprovedUltrastickEnv; retain finite-loss and tracking metrics.

Uses native observations/reward (reward multiplied by 100 for SAC) and fixes the
reference during each two-second episode. The second action has no plant authority.
"""

from __future__ import annotations

import argparse
import itertools
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
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.agent.sac import SAC
    from tensoraerospace.envs.ultrastick import ImprovedUltrastickEnv

    torch.set_num_threads(1)
    horizon = round(args.duration / args.dt)
    cases = list(itertools.product([-3.0, 3.0], [-2.0, 0.0, 2.0], [-2.0, 2.0]))

    class Tracking(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.steps = 0

        def reset(self, *, case=None, **kwargs):
            target, theta, q = (
                self.rng.uniform([-3, -2, -2], [3, 2, 2]) if case is None else case
            )
            self.target = float(target)
            initial = np.array([0, 0, np.deg2rad(theta), np.deg2rad(q), 0])
            self.env = ImprovedUltrastickEnv(
                initial,
                np.full((1, horizon + 1), np.deg2rad(target)),
                horizon + 1,
                dt=args.dt,
                use_initial_action_on_first_step=False,
            )
            return self.env.reset()

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.steps += 1
            x = self.env.model.xt.reshape(-1)
            if not np.all(np.isfinite(x)) or not np.isfinite(reward):
                raise FloatingPointError("Nonfinite rollout")
            failed = bool(abs(x[2]) > np.deg2rad(10) or abs(x[3]) > np.deg2rad(60))
            info["failed"] = failed
            return obs, 100 * reward, bool(terminated or failed), truncated, info

    env = Tracking(args.seed)
    agent = SAC(
        env,
        device="cpu",
        seed=args.seed,
        hidden_size=64,
        batch_size=128,
        memory_capacity=50000,
        alpha=0.01,
        log_every_updates=100,
        log_dir=args.output.with_suffix(""),
    )
    contract = agent.writer.assert_contract_satisfied
    agent.writer.assert_contract_satisfied = lambda: None

    def policy(obs):
        return agent.select_action(obs, evaluate=True)

    def evaluate(controller):
        evaluator = Tracking(20260917)
        errors, peaks, failures, traces, returns = [], np.zeros(5), 0, [], []
        for case in cases:
            obs, _ = evaluator.reset(case=case)
            err, trace, total = [], [], 0.0
            for step in range(horizon):
                obs, reward, terminated, truncated, info = evaluator.step(
                    controller(obs)
                )
                x = evaluator.env.model.xt.reshape(-1)
                error = float(np.rad2deg(x[2]) - case[0])
                err.append(error)
                peaks = np.maximum(peaks, abs(x))
                total += reward
                if step % max(1, horizon // 100) == 0:
                    trace.append(
                        [
                            float((step + 1) * args.dt),
                            float(np.rad2deg(x[2])),
                            float(np.rad2deg(x[3])),
                            float(info["elevator_deg"]),
                        ]
                    )
                if terminated or truncated:
                    failures += int(info["failed"])
                    break
            errors.append(float(np.sqrt(np.mean(np.square(err)))))
            returns.append(total)
            traces.append(trace)
        return dict(
            mean_rmse_deg=float(np.mean(errors)),
            episode_rmse_deg=errors,
            failed_episodes=failures,
            episodes=len(cases),
            max_abs_state=peaks.tolist(),
            mean_return=float(np.mean(returns)),
            traces=traces,
        )

    def pd(obs):
        # Positive elevator produces negative pitching moment in the published model.
        error_deg, q_deg_s = float(obs[0] * 30), float(obs[1] * 30)
        return np.array([np.clip((-0.7 * error_deg + 0.15 * q_deg_s) / 15, -1, 1), -1])

    result = dict(
        repo=str(args.repo.resolve()),
        seed=args.seed,
        config=dict(
            dt_s=args.dt,
            horizon_s=args.duration,
            cases=cases,
            frames=args.frames,
            reward_multiplier=100,
            hidden_size=64,
            alpha=0.01,
        ),
        zero=evaluate(lambda obs: np.array([0, -1])),
        pd=evaluate(pd),
    )
    if args.checkpoint:
        agent.policy.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)["policy"]
        )
        result["checkpoint"] = str(args.checkpoint)
    else:
        result["untrained"] = evaluate(policy)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        original_update = agent.update_parameters
        max_losses = np.zeros(5)

        def checked_update(*pos, **kwargs):
            losses = np.asarray(original_update(*pos, **kwargs))
            if not np.all(np.isfinite(losses)):
                raise FloatingPointError("Nonfinite training loss")
            max_losses[:] = np.maximum(max_losses, abs(losses))
            return tuple(losses)

        agent.update_parameters = checked_update
        curves, next_eval = [], 5000
        started = time.monotonic()
        while env.steps < args.frames:
            agent.train(1, max_steps=args.frames - env.steps, verbose=False)
            if env.steps >= next_eval:
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                metrics = evaluate(policy)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                curves.append(
                    dict(
                        frames=env.steps,
                        rmse_deg=metrics["mean_rmse_deg"],
                        failures=metrics["failed_episodes"],
                    )
                )
                print(json.dumps(curves[-1]), flush=True)
                next_eval += 5000
        contract()
        names = ["policy", "critic", "critic_target"]
        finite = all(
            torch.isfinite(p).all().item()
            for name in names
            for p in getattr(agent, name).parameters()
        )
        if not finite:
            raise FloatingPointError("Nonfinite trained parameters")
        result.update(
            learning_curve=curves,
            max_abs_losses=max_losses.tolist(),
            parameters_finite=finite,
            elapsed_s=time.monotonic() - started,
        )
        torch.save(
            {name: getattr(agent, name).state_dict() for name in names},
            args.output.with_suffix(".pt"),
        )
    result["evaluated"] = evaluate(policy)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    agent.close()
    print(json.dumps({k: v for k, v in result["evaluated"].items() if k != "traces"}))


if __name__ == "__main__":
    main()
