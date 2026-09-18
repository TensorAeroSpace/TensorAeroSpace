"""SAC regulation of small normalized ComSat perturbations, compared with LQR.

nominal_rho is only a coordinate offset. Dynamics and reported errors use the
paper's dimensionless variables, never km, m/s or Newtons.
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offset", type=float, default=6371.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.linalg import solve_continuous_are

    from tensoraerospace.agent.sac import SAC
    from tensoraerospace.envs.comsat import ImprovedComSatEnv

    torch.set_num_threads(1)
    dt, horizon, input_limit = 0.1, 100, 0.05
    scales = np.array([0.05, 0.01, 0.01])
    q = np.diag(np.array([1.0, 0.5, 0.5]) / scales**2)
    r = np.array([[0.001 / input_limit**2]])
    a = np.array([[0.0, 1.0, 0.0], [0.01036, 0.0, 0.7757], [0.0, -0.01775, 0.0]])
    b = np.array([[0.0], [0.0], [0.1513]])
    gain = np.linalg.solve(r, b.T @ solve_continuous_are(a, b, q, r))
    cases = [
        list(x)
        for x in itertools.product([-0.02, 0.02], [-0.004, 0.004], [-0.002, 0.002])
    ]
    cases += [[-0.04, 0, 0], [0.04, 0, 0], [0, 0, -0.004], [0, 0, 0.004]]

    class Regulation(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.steps = 0

        def state(self):
            return self.env.state - np.array([args.offset, 0, 0])

        def obs(self):
            return np.append(
                self.state() / scales,
                self.env.previous_action * self.env.max_thrust / input_limit,
            ).astype(np.float32)

        def reset(self, *, case=None, **kwargs):
            initial = (
                self.rng.uniform(-1, 1, 3) * np.array([0.025, 0.005, 0.0025])
                if case is None
                else np.array(case)
            )
            initial[0] += args.offset
            self.env = ImprovedComSatEnv(
                initial,
                np.zeros((1, horizon + 1)),
                horizon + 1,
                nominal_rho=args.offset,
                dt=dt,
                use_initial_action_on_first_step=False,
            )
            self.env.reset()
            self.elapsed = 0
            return self.obs(), {}

        def step(self, action):
            self.env.step(np.asarray(action) * input_limit / self.env.max_thrust)
            self.steps += 1
            self.elapsed += 1
            state = self.state()
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite ComSat state")
            control = self.env.model.store_input[0, self.env.model.time_step - 1]
            cost = float(state @ q @ state + 0.001 * (control / input_limit) ** 2)
            failed = bool(np.any(abs(state) > np.array([0.2, 0.05, 0.02])))
            reward = -cost - (10 if failed else 0)
            return (
                self.obs(),
                reward,
                failed,
                self.elapsed >= horizon,
                {"state_cost": float(state @ q @ state)},
            )

    env = Regulation(args.seed)
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

    def lqr(obs):
        return np.clip(-(gain @ (obs[:3] * scales)) / input_limit, -1, 1)

    def evaluate(controller):
        evaluator = Regulation(20260917)
        errors = []
        costs = []
        failed = 0
        peaks = np.zeros(3)
        traces = []
        for case in cases:
            obs, _ = evaluator.reset(case=case)
            squared = np.zeros(3)
            total_cost = 0.0
            steps = 0
            trace = []
            for _ in range(horizon):
                action = controller(obs)
                if not np.isfinite(action).all():
                    raise FloatingPointError("Nonfinite action")
                obs, _, term, trunc, info = evaluator.step(action)
                state = evaluator.state()
                squared += state**2
                total_cost += info["state_cost"]
                steps += 1
                peaks = np.maximum(peaks, abs(state))
                trace.append(state.tolist())
                if term or trunc:
                    failed += int(term)
                    break
            errors.append(np.sqrt(squared / steps).tolist())
            costs.append(total_cost / steps)
            traces.append(trace)
        return {
            "mean_state_cost": float(np.mean(costs)),
            "mean_rmse": np.mean(errors, axis=0).tolist(),
            "episode_state_cost": costs,
            "episode_rmse": errors,
            "failed_episodes": failed,
            "episodes": len(cases),
            "max_abs_state": peaks.tolist(),
            "traces": traces,
        }

    result = {
        "repo": str(args.repo.resolve()),
        "seed": args.seed,
        "offset": args.offset,
        "config": {
            "frames": args.frames,
            "dt_tau": dt,
            "horizon_tau": dt * horizon,
            "input_limit_normalized": input_limit,
            "scales": scales.tolist(),
            "q": q.tolist(),
            "r": r.tolist(),
            "lqr_gain": gain.tolist(),
            "cases": cases,
        },
        "zero": evaluate(lambda obs: np.zeros(1)),
        "lqr": evaluate(lqr),
    }
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        agent.policy.load_state_dict(saved["policy"])
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
                raise FloatingPointError("Nonfinite training loss")
            max_losses[:] = np.maximum(max_losses, abs(losses))
            return tuple(losses)

        agent.update_parameters = checked_update
        curves = []
        next_eval = 5000
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
                    {
                        "frames": env.steps,
                        "mean_state_cost": metrics["mean_state_cost"],
                        "failures": metrics["failed_episodes"],
                    }
                )
                print(json.dumps(curves[-1]), flush=True)
                next_eval += 5000
        contract()
        result.update(
            learning_curve=curves,
            environment_steps=env.steps,
            max_abs_losses=max_losses.tolist(),
            elapsed_s=time.monotonic() - started,
        )
        names = ["policy", "critic", "critic_target"]
        result["parameters_finite"] = all(
            torch.isfinite(p).all().item()
            for name in names
            for p in getattr(agent, name).parameters()
        )
        if not result["parameters_finite"]:
            raise FloatingPointError("Nonfinite parameters")
        torch.save(
            {name: getattr(agent, name).state_dict() for name in names},
            args.output.with_suffix(".pt"),
        )
    result["evaluated"] = evaluate(policy)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    agent.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in result["evaluated"].items()
                if k not in ["traces", "episode_state_cost", "episode_rmse"]
            }
        )
    )


if __name__ == "__main__":
    main()
