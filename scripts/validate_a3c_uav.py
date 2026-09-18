"""Paired A3C training on the same corrected UAV, with frozen six-state checks.

Runs the actual Worker inline for reproducible single-worker comparisons. It does
not establish convergence or performance with asynchronous multi-worker scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import queue
import random
import sys
import time
from pathlib import Path

from validate_uav_physics import source_matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plant-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-six-state", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    from scipy.linalg import expm

    from tensoraerospace.agent.a3c import pytorch as a3c
    from tensoraerospace.agent.a3c.shared_optim import SharedAdam

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    def load_file(name, relative):
        path = args.plant_repo / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    plant_module = load_file("audit_uav_plant", "tensoraerospace/aerospacemodel/uav.py")
    env_module = load_file("audit_uav_env", "tensoraerospace/envs/uav.py")
    env_module.LongitudinalUAV = plant_module.LongitudinalUAV
    dt, limit_deg = 0.02, 3.0
    horizon = round(args.duration / dt)
    cases = list(itertools.product([-2.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, 2.0]))
    a6, b6 = source_matrices()
    block = np.zeros((7, 7))
    block[:6, :6] = a6
    block[:6, 6] = b6
    zoh = expm(block * dt)
    ad6, bd6 = zoh[:6, :6], zoh[:6, 6]

    class Tracking(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def __init__(self, seed, evaluating=False):
            self.rng = np.random.default_rng(seed)
            self.steps = 0
            self.evaluating = evaluating
            self.failed_training_episodes = 0

        def observation(self):
            return np.array(
                [
                    self.x[0],
                    self.x[1],
                    np.rad2deg(self.x[2]) / 20,
                    np.rad2deg(self.x[3]) / 5,
                    (self.target - np.rad2deg(self.x[3])) / 5,
                    self.applied_deg / limit_deg,
                ],
                dtype=np.float32,
            )

        def reset(self, *, case=None, **kwargs):
            target, theta, pitch_rate = (
                self.rng.uniform([-2, -1, -2], [2, 1, 2]) if case is None else case
            )
            self.target = float(target)
            self.elapsed = 0
            self.applied_deg = 0.0
            self.had_failure = False
            initial = np.array([0.0, 0.0, np.deg2rad(pitch_rate), np.deg2rad(theta)])
            if args.full_six_state:
                self.x = np.r_[initial, 0.0, 0.0]
            else:
                self.env = env_module.LinearLongitudinalUAV(
                    initial,
                    np.full((1, horizon + 1), np.deg2rad(target)),
                    horizon + 1,
                    state_space=["u", "w", "q", "theta"],
                    tracking_states=["theta"],
                    dt=dt,
                )
                self.env.reset()
                self.x = self.env.model.xt.reshape(-1)
            return self.observation(), {}

        def step(self, action):
            command_deg = float(
                np.clip(np.asarray(action).reshape(-1)[0], -1, 1) * limit_deg
            )
            if args.full_six_state:
                self.applied_deg = float(
                    np.clip(
                        command_deg,
                        self.applied_deg - 60 * dt,
                        self.applied_deg + 60 * dt,
                    )
                )
                self.x = ad6 @ self.x + bd6 * np.deg2rad(self.applied_deg)
            else:
                self.env.step([np.deg2rad(command_deg)])
                self.x = self.env.model.xt.reshape(-1)
                self.applied_deg = float(
                    np.rad2deg(
                        self.env.model.store_input[0, self.env.model.time_step - 1]
                    )
                )
            self.steps += 1
            self.elapsed += 1
            if not np.all(np.isfinite(self.x)):
                raise FloatingPointError("Nonfinite UAV trajectory")
            theta, q = float(np.rad2deg(self.x[3])), float(np.rad2deg(self.x[2]))
            failed = bool(abs(theta) > 10 or abs(q) > 60)
            if failed and not self.had_failure:
                self.failed_training_episodes += 1
                self.had_failure = True
            cost = (
                ((theta - self.target) / 5) ** 2
                + 0.01 * (q / 20) ** 2
                + 0.002 * (self.applied_deg / limit_deg) ** 2
            )
            # Training uses full finite episodes; failures are counted and reported.
            return (
                self.observation(),
                float(-cost),
                bool(failed and self.evaluating),
                self.elapsed >= horizon,
                {"failed": failed},
            )

    net = a3c.Net(6, 1)
    net.share_memory()
    optimizer = SharedAdam(net.parameters(), lr=1e-4)

    def policy(obs):
        with torch.no_grad():
            return net(torch.from_numpy(obs[None]))[0][0].numpy()

    def pd(obs):
        return np.array([(0.7 * (-obs[4] * 5) + 0.15 * obs[2] * 20) / limit_deg])

    def evaluate(controller):
        env = Tracking(20260917, evaluating=True)
        errors, failures, traces, peaks = [], 0, [], np.zeros(4)
        for case in cases:
            obs, _ = env.reset(case=case)
            err = []
            trace = []
            for step in range(horizon):
                obs, _, terminated, truncated, info = env.step(controller(obs))
                x = env.x[:4]
                peaks = np.maximum(peaks, abs(x))
                err.append(float(np.rad2deg(x[3]) - case[0]))
                if step % max(1, horizon // 100) == 0:
                    trace.append(
                        [
                            (step + 1) * dt,
                            float(np.rad2deg(x[3])),
                            float(np.rad2deg(x[2])),
                            env.applied_deg,
                        ]
                    )
                if terminated or truncated:
                    failures += int(info["failed"])
                    break
            errors.append(float(np.sqrt(np.mean(np.square(err)))))
            traces.append(trace)
        return dict(
            mean_rmse_deg=float(np.mean(errors)),
            episode_rmse_deg=errors,
            failed_episodes=failures,
            episodes=len(cases),
            max_abs_state=peaks.tolist(),
            traces=traces,
        )

    result = dict(
        repo=str(args.repo.resolve()),
        seed=args.seed,
        plant_repo=str(args.plant_repo.resolve()),
        plant_sha256={
            name: hashlib.sha256((args.plant_repo / name).read_bytes()).hexdigest()
            for name in [
                "tensoraerospace/aerospacemodel/uav.py",
                "tensoraerospace/envs/uav.py",
            ]
        },
        config=dict(
            dt_s=dt,
            duration_s=args.duration,
            cases=cases,
            action_limit_deg=limit_deg,
            episodes=args.episodes,
            learning_rate=1e-4,
            gamma=0.99,
            update_interval=10,
            full_six_state=args.full_six_state,
            worker_mode="inline, one worker",
        ),
        zero=evaluate(lambda obs: np.zeros(1)),
        pd=evaluate(pd),
    )
    if args.checkpoint:
        net.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
        result["checkpoint"] = str(args.checkpoint)
    else:
        if args.full_six_state:
            raise ValueError("Use --checkpoint for the six-state validation")
        result["untrained"] = evaluate(policy)
        env = Tracking(args.seed)
        original = a3c.push_and_pull
        max_losses = {}
        curves = []
        updates = 0
        next_eval = 10000

        def checked(*pos, **kw):
            nonlocal updates, next_eval
            metrics = original(*pos, **kw)
            updates += 1
            for key, val in metrics.items():
                if not np.isfinite(val):
                    raise FloatingPointError("Nonfinite A3C loss")
                max_losses[key] = max(max_losses.get(key, 0.0), abs(val))
            if not all(torch.isfinite(p).all() for p in net.parameters()):
                raise FloatingPointError("Nonfinite weights")
            if env.steps >= next_eval:
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                metrics_eval = evaluate(policy)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                curves.append(
                    dict(
                        steps=env.steps,
                        rmse_deg=metrics_eval["mean_rmse_deg"],
                        failures=metrics_eval["failed_episodes"],
                    )
                )
                print(json.dumps(curves[-1]), flush=True)
                next_eval += 10000
            return metrics

        a3c.push_and_pull = checked
        worker = a3c.Worker(
            env=env,
            gnet=net,
            opt=optimizer,
            global_ep=mp.Value("i", 0),
            global_ep_r=mp.Value("d", 0),
            res_queue=queue.SimpleQueue(),
            name=0,
            num_actions=1,
            num_observations=6,
            MAX_EP=args.episodes,
            MAX_EP_STEP=horizon,
            GAMMA=0.99,
            update_global_iter=10,
        )
        started = time.monotonic()
        worker.run()
        result.update(
            learning_curve=curves,
            max_abs_losses=max_losses,
            parameters_finite=True,
            training_steps=env.steps,
            training_updates=updates,
            training_failed_episodes=env.failed_training_episodes,
            elapsed_s=time.monotonic() - started,
        )
        torch.save(net.state_dict(), args.output.with_suffix(".pt"))
    result["evaluated"] = evaluate(policy)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result["evaluated"].items() if k != "traces"}))


if __name__ == "__main__":
    main()
