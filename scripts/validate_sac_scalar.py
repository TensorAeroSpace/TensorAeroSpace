"""Compare scalar SAC collection and learning on one shared, audited B747 environment.

Run in separate processes for each repository revision. The environment source
is fixed by --env-source so that algorithm comparisons use identical physics.
Results include deterministic held-out tracking, failures, losses and weights;
finite losses alone are not interpreted as policy convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--env-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--buffer-mode", choices=["copied", "reused"], default="reused")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = "tensoraerospace.envs.validation_b747"
    spec = importlib.util.spec_from_file_location(name, args.env_source.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    Env = module.ImprovedB747VecEnvTorch

    def make_env(n, seed, auto_reset=True):
        return Env(
            num_envs=n,
            dt=0.05,
            tn=10.0,
            device="cpu",
            seed=seed,
            auto_reset=auto_reset,
            include_reference_in_obs=True,
            reward_mode="tracking",
            step_randomization={
                "signal_type": "step",
                "amplitude_deg_range": (-4.0, 4.0),
                "min_abs_amplitude_deg": 1.0,
                "step_time_sec_range": (0.5, 1.5),
            },
        )

    import gymnasium as gym

    from tensoraerospace.agent.sac.sac import SAC

    class ScalarEnv(gym.Env):
        def __init__(self):
            self.env = make_env(1, args.seed, auto_reset=False)
            self.buffer = np.zeros(6, dtype=np.float32)
            self.steps = 0
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def observation(self, obs):
            self.buffer[:] = obs[0].numpy()
            return self.buffer if args.buffer_mode == "reused" else self.buffer.copy()

        def reset(self, **kwargs):
            obs, info = self.env.reset()
            return self.observation(obs), info

        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(
                torch.tensor(action).reshape(1, 1)
            )
            self.steps += 1
            return (
                self.observation(obs),
                float(reward[0]),
                bool(term[0]),
                bool(trunc[0]),
                info,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = ScalarEnv()
    agent = SAC(
        env,
        device="cpu",
        seed=args.seed,
        hidden_size=64,
        batch_size=128,
        memory_capacity=50000,
        alpha=0.01,
        automatic_entropy_tuning=False,
        log_dir=str(args.output.with_suffix("")),
        log_every_updates=100,
    )
    networks = [agent.policy, agent.critic]
    # The experiment spans repeated one-episode calls. Check the metrics
    # contract once after the complete budget, including the warmup episodes.
    assert_metrics = agent.writer.assert_contract_satisfied
    agent.writer.assert_contract_satisfied = lambda: None

    def policy(obs):
        return agent.select_action_batch(obs, evaluate=True, return_tensor=True)

    def evaluate(control):
        evaluation = make_env(24, 20260916, auto_reset=False)
        evaluation.reset()
        amplitudes = torch.tensor([-4.0, -2.0, 2.0, 4.0]).repeat(6)
        starts = torch.linspace(0.5, 1.5, 24)
        evaluation.reference_signal[:] = torch.deg2rad(amplitudes[:, None]) * (
            evaluation.tps[None, :] >= starts[:, None]
        )
        obs = evaluation._get_obs()
        rewards = np.zeros(24)
        squares = np.zeros(24)
        counts = np.zeros(24)
        active = np.ones(24, dtype=bool)
        failed = np.zeros(24, dtype=bool)
        peaks = np.zeros(24)
        saturation = np.zeros(24)
        traces = []
        for step in range(evaluation.number_time_steps - 2):
            with torch.no_grad():
                action = control(obs)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Nonfinite evaluation action")
            obs, reward, term, trunc, _ = evaluation.step(action)
            theta = evaluation.state[:, 3].numpy()
            target = evaluation.reference_signal[:, step + 1].numpy()
            error_deg = np.rad2deg(theta - target)
            rewards[active] += reward.numpy()[active]
            squares[active] += error_deg[active] ** 2
            counts[active] += 1
            peaks[active] = np.maximum(peaks[active], np.abs(np.rad2deg(theta[active])))
            saturation[active] += action.reshape(-1).abs().numpy()[active] >= 0.99
            failed |= active & term.numpy()
            active &= ~(term | trunc).numpy()
            traces.append([float(theta[0]), float(target[0]), float(action[0, 0])])
        return {
            "reward_mean": float(rewards.mean()),
            "rmse_deg_mean": float(np.sqrt(squares / np.maximum(counts, 1)).mean()),
            "failed_episodes": int(failed.sum()),
            "episodes": 24,
            "max_pitch_deg": float(peaks.max()),
            "saturation_fraction": float((saturation / np.maximum(counts, 1)).mean()),
            "episode_rewards": rewards.tolist(),
            "episode_rmse_deg": np.sqrt(squares / np.maximum(counts, 1)).tolist(),
            "first_episode_trace_rad": traces,
        }

    result = {
        "algorithm": "sac_scalar",
        "seed": args.seed,
        "repo": str(args.repo.resolve()),
        "environment_sha256": hashlib.sha256(args.env_source.read_bytes()).hexdigest(),
        "config": {
            "training_steps": args.frames,
            "buffer_mode": args.buffer_mode,
            "hidden_size": 64,
            "batch_size": 128,
            "alpha": 0.01,
            "automatic_entropy_tuning": False,
            "gamma": 0.99,
        },
        "untrained": evaluate(policy),
        "zero_control": evaluate(lambda obs: torch.zeros((24, 1))),
        "pd_control": evaluate(
            lambda obs: (-2 * obs[:, 0] + 0.6 * obs[:, 1]).clamp(-1, 1).unsqueeze(1)
        ),
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    losses = []
    curves = []
    original_scalar = agent.writer.add_scalar

    def monitor(tag, scalar_value, *pos, **kw):
        numeric = float(scalar_value)
        if not np.isfinite(numeric):
            raise FloatingPointError(f"Nonfinite metric {tag}: {numeric}")
        if tag.startswith("loss/"):
            losses.append({"step": env.steps, "tag": tag, "value": numeric})
        return original_scalar(tag, scalar_value, *pos, **kw)

    agent.writer.add_scalar = monitor
    started = time.monotonic()
    try:
        next_evaluation = 5000
        while env.steps < args.frames:
            agent.train(
                num_episodes=1, max_steps=args.frames - env.steps, verbose=False
            )
            if not all(
                bool(torch.isfinite(p).all()) for n in networks for p in n.parameters()
            ):
                raise FloatingPointError("Nonfinite weights")
            if env.steps >= next_evaluation:
                metrics = evaluate(policy)
                curves.append(
                    {
                        "step": env.steps,
                        "rmse_deg": metrics["rmse_deg_mean"],
                        "failed_episodes": metrics["failed_episodes"],
                    }
                )
                next_evaluation += 5000
                print(json.dumps(curves[-1]), flush=True)
        assert_metrics()
        result["trained"] = evaluate(policy)
        result["environment_steps"] = env.steps
        result["parameters_finite"] = True
        result["learning_curve"] = curves
        result["loss_samples"] = losses[:: max(1, len(losses) // 200)]
        torch.save(
            {"policy": agent.policy.state_dict(), "critic": agent.critic.state_dict()},
            args.output.with_suffix(".pt"),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        agent.writer.close()
    print(
        json.dumps(
            {k: v for k, v in result["trained"].items() if not isinstance(v, list)}
        )
    )


if __name__ == "__main__":
    main()
