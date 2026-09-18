"""Compare real PPO/SAC learning on one shared, audited B747 environment.

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
    parser.add_argument("--algorithm", choices=["ppo", "sac"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppo-updates", type=int, default=40)
    parser.add_argument("--sac-steps", type=int, default=6000)
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

    env = make_env(16, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.with_suffix("")
    if args.algorithm == "ppo":
        from tensoraerospace.agent.ppo.model import PPO

        agent = PPO(
            env,
            device="cpu",
            seed=args.seed,
            actor_hidden_dim=64,
            critic_hidden_dim=64,
            rollout_len=256,
            max_episodes=args.ppo_updates,
            num_epochs=4,
            batch_size=512,
            actor_lr=3e-4,
            critic_lr=1e-3,
            normalize_obs=False,
            normalize_reward=False,
            save_best_model=False,
            target_kl=0.03,
            log_dir=log_dir,
            entropy_coef=0.001,
        )

        def policy(obs):
            return agent.act(obs, deterministic=True)[0]

        networks = [agent.actor, agent.critic]
    else:
        from tensoraerospace.agent.sac.sac import SAC

        agent = SAC(
            env,
            device="cpu",
            seed=args.seed,
            hidden_size=64,
            batch_size=128,
            memory_capacity=150000,
            alpha=0.01,
            automatic_entropy_tuning=False,
            log_dir=log_dir,
            log_every_updates=100,
        )

        def policy(obs):
            return agent.select_action_batch(obs, evaluate=True, return_tensor=True)

        networks = [agent.policy, agent.critic]

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
        "algorithm": args.algorithm,
        "seed": args.seed,
        "repo": str(args.repo.resolve()),
        "environment_sha256": hashlib.sha256(args.env_source.read_bytes()).hexdigest(),
        "config": {
            "num_envs": 16,
            "dt": 0.05,
            "horizon_s": 10.0,
            "ppo_updates": args.ppo_updates,
            "sac_vector_steps": args.sac_steps,
        },
        "untrained": evaluate(policy),
        "zero_control": evaluate(lambda obs: torch.zeros((24, 1))),
        "pd_control": evaluate(
            lambda obs: (-2.0 * obs[:, 0] + 0.6 * obs[:, 1]).clamp(-1, 1).unsqueeze(1)
        ),
    }
    # Evaluation must not change the random stream used to collect training data.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    losses = []
    if args.algorithm == "ppo":
        original = agent.learn

        def monitored(*a, **kw):
            metrics = original(*a, **kw)
            numeric = {k: float(v) for k, v in metrics.items()}
            if not all(np.isfinite(v) for v in numeric.values()):
                raise FloatingPointError(f"Nonfinite PPO update: {numeric}")
            losses.append(numeric)
            return metrics

        agent.learn = monitored
    else:
        original = agent.update_parameters

        def monitored(*a, **kw):
            metrics = original(*a, **kw)
            values = np.asarray(metrics, dtype=float)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite SAC update")
            losses.append(values.tolist())
            return metrics

        agent.update_parameters = monitored
    started = time.monotonic()
    try:
        if args.algorithm == "ppo":
            agent.train(verbose=False)
            result["environment_steps"] = agent.global_env_step
        else:
            agent.train_vector(
                total_steps=args.sac_steps,
                warmup_steps=250,
                log_every=1000,
                save_best=False,
            )
            result["environment_steps"] = 16 * args.sac_steps
        result["trained"] = evaluate(policy)
        result["parameters_finite"] = all(
            bool(torch.isfinite(p).all())
            for network in networks
            for p in network.parameters()
        )
        result["weight_norm"] = float(
            np.sqrt(
                sum(
                    float(p.detach().square().sum())
                    for network in networks
                    for p in network.parameters()
                )
            )
        )
        result["updates"] = len(losses)
        result["loss_samples"] = losses[:: max(1, len(losses) // 100)]
        torch.save(
            {str(i): network.state_dict() for i, network in enumerate(networks)},
            args.output.with_suffix(".pt"),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        agent.close()
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "algorithm",
                    "seed",
                    "environment_steps",
                    "parameters_finite",
                    "updates",
                    "elapsed_seconds",
                ]
            }
        )
    )
    print(
        json.dumps(
            {k: v for k, v in result["trained"].items() if not isinstance(v, list)}
        )
    )


if __name__ == "__main__":
    main()
