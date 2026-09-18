"""Compare legacy B747 reward/time-limit revisions using a shared SAC agent.

Load the environment from --env-source. Normalize observed degrees and actions,
provide the reference error to the policy, and scale native rewards by 100.
The physical model and SAC implementation are both fixed by --repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--env-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=0.0,
        help="Minimum native-reward penalty on external envelope termination; 0 reproduces the original protocol.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.agent.sac import SAC

    spec = importlib.util.spec_from_file_location("validation_b747", args.env_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Env = module.LinearLongitudinalB747

    torch.set_num_threads(1)
    horizon = round(args.duration / args.dt)
    cases = list(itertools.product([-2.0, 2.0], [-1.0, 0.0, 1.0], [-1.0, 1.0]))

    class Tracking(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.steps = 0
            self.failures = 0
            self.episodes = 0

        def reset(self, *, case=None, **kwargs):
            target, theta, q = (
                self.rng.uniform([-2, -1, -1], [2, 1, 1]) if case is None else case
            )
            self.target = float(target)
            initial = np.array([0, 0, np.deg2rad(q), np.deg2rad(theta)])
            self.env = Env(
                initial,
                np.full((1, horizon + 1), np.deg2rad(target)),
                horizon + 1,
                dt=args.dt,
            )
            self.env.reset()
            return self.observation(), {}

        def observation(self):
            x = self.env.model.xt.reshape(-1)
            last = (
                self.env.model.store_input[0, self.env.model.time_step - 1]
                if self.env.model.time_step
                else 0.0
            )
            return np.clip(
                [
                    (np.deg2rad(self.target) - x[3]) / np.deg2rad(20),
                    x[2] / np.deg2rad(5),
                    x[3] / np.deg2rad(20),
                    last / np.deg2rad(25),
                ],
                -1,
                1,
            ).astype(np.float32)

        def step(self, action):
            _, reward, terminated, truncated, info = self.env.step(
                np.asarray(action) * 25
            )
            obs = self.observation()
            self.steps += 1
            x = self.env.model.xt.reshape(-1)
            if not np.all(np.isfinite(x)) or not np.isfinite(reward):
                raise FloatingPointError("Nonfinite rollout")
            failed = bool(abs(x[3]) > np.deg2rad(10) or abs(x[2]) > np.deg2rad(5))
            info["failed"] = failed
            if failed and args.failure_penalty > 0:
                reward = min(reward, -args.failure_penalty)
            self.failures += int(failed)
            self.episodes += int(terminated or failed or truncated)
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
        errors, peaks, failures, traces, returns = [], np.zeros(4), 0, [], []
        for case in cases:
            obs, _ = evaluator.reset(case=case)
            err, trace, total = [], [], 0.0
            for step in range(horizon):
                obs, reward, terminated, truncated, info = evaluator.step(
                    controller(obs)
                )
                x = evaluator.env.model.xt.reshape(-1)
                error = float(np.rad2deg(x[3]) - case[0])
                err.append(error)
                peaks = np.maximum(peaks, abs(x))
                total += reward
                if step % max(1, horizon // 100) == 0:
                    trace.append(
                        [
                            float((step + 1) * args.dt),
                            float(np.rad2deg(x[3])),
                            float(np.rad2deg(x[2])),
                            float(
                                np.rad2deg(
                                    evaluator.env.model.store_input[
                                        0, evaluator.env.model.time_step - 1
                                    ]
                                )
                            ),
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
        error_deg, q_deg_s = float(obs[0] * 20), float(obs[1] * 5)
        return np.array([np.clip((-2 * error_deg + 1.5 * q_deg_s) / 25, -1, 1)])

    result = dict(
        repo=str(args.repo.resolve()),
        environment_source=str(args.env_source.resolve()),
        environment_sha256=hashlib.sha256(args.env_source.read_bytes()).hexdigest(),
        seed=args.seed,
        config=dict(
            dt_s=args.dt,
            horizon_s=args.duration,
            cases=cases,
            frames=args.frames,
            reward_multiplier=100,
            external_failure_penalty=args.failure_penalty,
            hidden_size=64,
            alpha=0.01,
        ),
        zero=evaluate(lambda obs: np.array([0])),
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
            training_failures=env.failures,
            training_episodes=env.episodes,
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
