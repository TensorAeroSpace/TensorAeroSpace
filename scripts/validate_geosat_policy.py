"""Compare DQN training and frozen SAC/DQN policies on normalized GeoSat dynamics.

The regulation wrapper uses an explicit small-perturbation quadratic objective.
The independent nonlinear option propagates the orbital ODE, not GeoSat matrices.
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
    parser.add_argument("--plant-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--algorithm", choices=["dqn", "sac"], default="dqn")
    parser.add_argument("--nonlinear", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.integrate import solve_ivp
    from scipy.linalg import solve_continuous_are

    from tensoraerospace.agent.dqn import model as dqn
    from tensoraerospace.agent.sac.model import GaussianPolicy

    torch.set_num_threads(1)
    dqn._DEVICE = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = importlib.util.spec_from_file_location("fixed_geosat", args.plant_source)
    plant_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plant_module
    spec.loader.exec_module(plant_module)
    dt, horizon, limit = 0.1, 100, 0.05
    scales = np.array([0.05, 0.01, 0.01])
    q = np.diag(np.array([1, 0.5, 0.5]) / scales**2)
    r = np.array([[0.001 / limit**2]])
    levels = np.linspace(-limit, limit, 11)
    a = np.array([[0, 1, 0], [0.01036, 0, 0.7753], [0, -0.01774, 0]])
    b = np.array([[0], [0], [0.1512]])
    gain = np.linalg.solve(r, b.T @ solve_continuous_are(a, b, q, r))
    radius = 6.6108
    equilibrium = np.array([radius, 0, np.sqrt(1 / radius**3)])
    cases = [
        list(x)
        for x in itertools.product([-0.02, 0.02], [-0.004, 0.004], [-0.002, 0.002])
    ]
    cases += [[-0.04, 0, 0], [0.04, 0, 0], [0, 0, -0.004], [0, 0, 0.004]]

    class Regulation(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        action_space = gym.spaces.Discrete(len(levels))

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.steps = 0

        def obs(self):
            return np.append(self.state / scales, self.previous / limit).astype(
                np.float32
            )

        def reset(self, *, case=None, **kwargs):
            self.state = (
                self.rng.uniform(-1, 1, 3) * [0.025, 0.005, 0.0025]
                if case is None
                else np.array(case, dtype=float)
            )
            self.model = plant_module.GeoSat(self.state, horizon, dt=dt)
            self.previous = 0.0
            self.elapsed = 0
            return self.obs(), {}

        def step_control(self, command):
            applied = float(
                np.clip(
                    np.clip(command, -limit, limit),
                    self.previous - np.deg2rad(60) * dt,
                    self.previous + np.deg2rad(60) * dt,
                )
            )
            if args.nonlinear:

                def rhs(t, x):
                    rho, velocity, omega = x
                    return [
                        velocity,
                        rho * omega**2 - 1 / rho**2,
                        -2 * velocity * omega / rho + applied / rho,
                    ]

                solution = solve_ivp(
                    rhs,
                    (0, dt),
                    equilibrium + self.state,
                    method="DOP853",
                    rtol=1e-12,
                    atol=1e-14,
                )
                if not solution.success:
                    raise RuntimeError("Nonlinear integration failed")
                self.state = solution.y[:, -1] - equilibrium
            else:
                self.state = self.model.run_step([command]).reshape(-1)
                applied = self.model.store_input[0, self.model.time_step - 1]
            if not np.isfinite(self.state).all() or not np.isfinite(applied):
                raise FloatingPointError("Nonfinite plant state or command")
            self.previous = float(applied)
            self.elapsed += 1
            self.steps += 1
            state_cost = float(self.state @ q @ self.state)
            failed = bool(np.any(abs(self.state) > [0.2, 0.05, 0.02]))
            return (
                self.obs(),
                -state_cost - 0.001 * (applied / limit) ** 2 - 10 * failed,
                failed,
                self.elapsed >= horizon,
                {"state_cost": state_cost},
            )

        def step(self, action):
            return self.step_control(float(levels[int(action)]))

    env = Regulation(args.seed)
    env.action_space.seed(args.seed)
    agent = None
    if args.algorithm == "dqn":
        agent = dqn.DQNAgent(
            dqn.Model(len(levels)),
            dqn.Model(len(levels)),
            env,
            learning_rate=0.0005,
            gamma=0.98,
            epsilon=0.8,
            epsilon_dacay=0.985,
            min_epsilon=0.05,
            batch_size=64,
            buffer_size=512,
            target_update_iter=1000,
            beta_increment_per_sample=0.00003,
            train_nums=args.frames,
            seed=args.seed,
            log_dir=str(args.output.with_suffix("")),
        )

        def policy(obs):
            action, values = agent.model.action_value(obs[None])
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite Q value")
            return float(levels[int(action)])

    else:
        if args.checkpoint is None:
            raise ValueError("SAC mode evaluates an existing checkpoint")
        actor = GaussianPolicy(4, 1, 64, gym.spaces.Box(-1, 1, (1,), np.float32))
        actor.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)["policy"]
        )
        actor.eval()

        def policy(obs):
            with torch.no_grad():
                mean, _ = actor(torch.as_tensor(obs).unsqueeze(0))
                return float(torch.tanh(mean).item()) * limit

    def evaluate(controller):
        evaluator = Regulation(20260917)
        costs, rmses, traces = [], [], []
        failures = 0
        peak = np.zeros(3)
        for case in cases:
            obs, _ = evaluator.reset(case=case)
            trace, cost = [], []
            for _ in range(horizon):
                command = controller(obs)
                if not np.isfinite(command):
                    raise FloatingPointError("Nonfinite policy action")
                obs, _, term, trunc, info = evaluator.step_control(command)
                trace.append(evaluator.state.tolist())
                cost.append(info["state_cost"])
                peak = np.maximum(peak, abs(evaluator.state))
                if term or trunc:
                    failures += int(term)
                    break
            costs.append(float(np.mean(cost)))
            rmses.append(np.sqrt(np.mean(np.array(trace) ** 2, axis=0)).tolist())
            traces.append(trace)
        return {
            "mean_state_cost": float(np.mean(costs)),
            "mean_rmse": np.mean(rmses, axis=0).tolist(),
            "episode_state_cost": costs,
            "episodes": len(cases),
            "failed_episodes": failures,
            "max_abs_state": peak.tolist(),
            "traces": traces,
        }

    def lqr(obs):
        return float(np.clip(-(gain @ (obs[:3] * scales)).item(), -limit, limit))

    def quantized_lqr(obs):
        return float(levels[np.argmin(abs(levels - lqr(obs)))])

    result = {
        "repo": str(args.repo.resolve()),
        "seed": args.seed,
        "algorithm": args.algorithm,
        "nonlinear": args.nonlinear,
        "plant_source": str(args.plant_source),
        "plant_sha256": hashlib.sha256(args.plant_source.read_bytes()).hexdigest(),
        "config": {
            "frames": args.frames,
            "dt_tau": dt,
            "horizon_tau": dt * horizon,
            "levels": levels.tolist(),
            "q": q.tolist(),
            "r": r.tolist(),
            "scales": scales.tolist(),
            "cases": cases,
            "dqn": {
                "lr": 0.0005,
                "gamma": 0.98,
                "epsilon": 0.8,
                "epsilon_decay": 0.985,
                "min_epsilon": 0.05,
                "batch": 64,
                "buffer": 512,
                "target_update": 1000,
            },
        },
        "zero": evaluate(lambda obs: 0),
        "lqr": evaluate(lqr),
        "quantized_lqr": evaluate(quantized_lqr),
    }
    if args.checkpoint and agent is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        agent.model.predict(np.zeros((1, 4), np.float32))
        agent.model.load_state_dict(state["model"])
    elif not args.checkpoint:
        result["untrained"] = evaluate(policy)
        curves = []
        original = agent.train_step
        maximum_loss = 0.0

        def checked_update():
            nonlocal maximum_loss
            loss = float(original())
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite DQN loss")
            maximum_loss = max(maximum_loss, loss)
            if agent.global_env_step % 5000 == 0:
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                metrics = evaluate(policy)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                curves.append(
                    {
                        "frames": agent.global_env_step,
                        "cost": metrics["mean_state_cost"],
                        "failures": metrics["failed_episodes"],
                    }
                )
                print(json.dumps(curves[-1]), flush=True)
            return loss

        agent.train_step = checked_update
        started = time.monotonic()
        agent.train(verbose=False)
        result.update(
            environment_steps=env.steps,
            updates=agent.global_step,
            epsilon=agent.epsilon,
            max_loss=maximum_loss,
            learning_curve=curves,
            elapsed_s=time.monotonic() - started,
        )
        result["parameters_finite"] = all(
            torch.isfinite(p).all().item()
            for net in [agent.model, agent.target_model]
            for p in net.parameters()
        )
        if not result["parameters_finite"]:
            raise FloatingPointError("Nonfinite DQN parameters")
        torch.save(
            {
                "model": agent.model.state_dict(),
                "target_model": agent.target_model.state_dict(),
            },
            args.output.with_suffix(".pt"),
        )
    result["evaluated"] = evaluate(policy)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if agent is not None:
        agent.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in result["evaluated"].items()
                if k not in ["traces", "episode_state_cost"]
            }
        )
    )


if __name__ == "__main__":
    main()
