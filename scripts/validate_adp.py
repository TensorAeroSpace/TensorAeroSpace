"""Exercise IM-GDHP online learning on the common B747 dynamics over fixed seeds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--env-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cov-init", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch

    from tensoraerospace.agent.im_gdhp.model import IMGDHPAgent, IMGDHPConfig

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = "tensoraerospace.envs.validation_b747"
    spec = importlib.util.spec_from_file_location(name, args.env_source.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    env = module.ImprovedB747VecEnvTorch(
        num_envs=1,
        dt=0.05,
        tn=10.0,
        seed=args.seed,
        device="cpu",
        auto_reset=False,
        reward_mode="tracking",
        step_randomization={
            "signal_type": "step",
            "amplitude_deg_range": (-2.0, 2.0),
            "step_time_sec_range": (1.0, 1.0),
            "min_abs_amplitude_deg": 1.0,
        },
    )
    cfg = IMGDHPConfig(
        gamma=args.gamma,
        cov_init=args.cov_init,
        actor_hidden=(16, 16),
        critic_hidden=(32, 32),
        actor_lr=1e-4,
        critic_lr=1e-3,
        track_Q=(1.0,),
        obs_scale=(1.0, 1.0, 10.0, 10.0),
        warmup_steps=50,
        critic_only_steps=50,
        beta_lambda=0.1,
        target_update_tau=0.01,
        exploration_noise_std=0.002,
        u_max=0.04,
        device="cpu",
        seed=args.seed,
    )
    agent = IMGDHPAgent(4, 1, reference_size=1, tracking_indices=[3], config=cfg)

    def evaluate():
        values = []
        failures = 0
        for amplitude in [-2.0, -1.0, 1.0, 2.0]:
            env.reset()
            reference = np.deg2rad(amplitude) * (np.arange(202)[None, :] * 0.05 >= 1.0)
            agent.reset()
            squares = []
            for step in range(199):
                state = env.state[0].numpy().copy()
                action = agent.predict(state, reference, step, deterministic=True)
                _, _, terminated, _, _ = env.step(np.asarray(action).reshape(1, 1))
                if not np.isfinite(action).all() or not torch.isfinite(env.state).all():
                    raise FloatingPointError("Nonfinite IM-GDHP action or trajectory")
                squares.append(
                    float(np.rad2deg(env.state[0, 3].item() - reference[0, step + 1]))
                    ** 2
                )
                if terminated[0]:
                    failures += 1
                    break
            values.append(float(np.sqrt(np.mean(squares))))
        return {
            "rmse_deg_mean": float(np.mean(values)),
            "episode_rmse_deg": values,
            "failed_episodes": failures,
            "episodes": 4,
        }

    initial = evaluate()
    losses = []
    failures = 0
    steps = 0
    curves = []
    for episode in range(20):
        env.reset()
        agent.reset()
        reference = env.reference_signal.numpy().copy()
        squared = []
        for step in range(199):
            state = env.state[0].numpy().copy()
            action = agent.predict(state, reference, step)
            _, _, term, _, _ = env.step(np.asarray(action).reshape(1, 1))
            next_state = env.state[0].numpy().copy()
            metrics = agent.learn(next_state, reference, step)
            if not np.isfinite(next_state).all() or not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite training transition")
            # These agents report NaN for losses before the corresponding update starts.
            finite = {k: float(v) for k, v in metrics.items() if np.isfinite(v)}
            if step >= 101 and not np.isfinite(metrics["critic_loss"]):
                raise FloatingPointError("Nonfinite active critic loss")
            losses.append(finite)
            squared.append(
                float(np.rad2deg(next_state[3] - reference[0, step + 1])) ** 2
            )
            steps += 1
            if term[0]:
                failures += 1
                break
        curves.append(float(np.sqrt(np.mean(squared))))
    result = {
        "algorithm": "imgdhp",
        "seed": args.seed,
        "repo": str(args.repo.resolve()),
        "config": {k: v for k, v in vars(cfg).items() if k != "history"},
        "environment_steps": steps,
        "training_failures": failures,
        "untrained": initial,
        "trained": evaluate(),
        "training_episode_rmse_deg": curves,
        "identified_A": agent.incremental_model.A.tolist(),
        "identified_B": agent.incremental_model.B.tolist(),
        "physical_B": (env.Bd.numpy() * env._max_elev_rad).tolist(),
        "loss_samples": losses[::100],
        "parameters_finite": all(
            bool(torch.isfinite(p).all())
            for net in [agent.actor, agent.critic]
            for p in net.parameters()
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["trained"]))


if __name__ == "__main__":
    main()
