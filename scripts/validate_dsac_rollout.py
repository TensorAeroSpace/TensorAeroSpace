"""Compare DSAC learning on a fixed B747 vector environment and validate replay."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=80000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch

    from tensoraerospace.agent.dsac import DSAC

    torch.set_num_threads(1)
    assert args.frames % 4 == 0
    spec = importlib.util.spec_from_file_location("fixed_b747", args.env_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def make_plant(n, seed, auto_reset=False):
        return module.ImprovedB747VecEnvTorch(
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

    env = make_plant(4, args.seed, True)
    agent = DSAC(
        env,
        hidden_size=64,
        batch_size=128,
        memory_capacity=100000,
        learning_starts=512,
        num_quantiles=8,
        embedding_dim=32,
        alpha=0.01,
        automatic_entropy_tuning=False,
        caps_lambda_smoothness=0.0,
        caps_lambda_temporal=0.0,
        lr=0.0003,
        policy_lr=0.0003,
        device="cpu",
        seed=args.seed,
        log_every_updates=100,
        log_dir=args.output.with_suffix(""),
    )

    def evaluate(controller):
        plant = make_plant(24, 20260916)
        plant.reset()
        amplitudes = torch.tensor([-4.0, -2.0, 2.0, 4.0]).repeat(6)
        starts = torch.linspace(0.5, 1.5, 24)
        plant.reference_signal[:] = torch.deg2rad(amplitudes[:, None]) * (
            plant.tps[None, :] >= starts[:, None]
        )
        obs = plant._get_obs()
        squares, counts = np.zeros(24), np.zeros(24)
        active, failed = np.ones(24, bool), np.zeros(24, bool)
        peak = 0.0
        for step in range(plant.number_time_steps - 2):
            actions = controller(obs)
            if not torch.isfinite(actions).all():
                raise FloatingPointError("Nonfinite policy action")
            obs, _, term, trunc, _ = plant.step(actions)
            pitch = np.rad2deg(plant.state[:, 3].numpy())
            error = pitch - np.rad2deg(plant.reference_signal[:, step + 1].numpy())
            squares[active] += error[active] ** 2
            counts[active] += 1
            if active.any():
                peak = max(peak, float(abs(pitch[active]).max()))
            failed |= active & term.numpy()
            active &= ~(term | trunc).numpy()
        errors = np.sqrt(squares / np.maximum(counts, 1))
        return {
            "rmse_deg_mean": float(errors.mean()),
            "episode_rmse_deg": errors.tolist(),
            "episodes": 24,
            "failed_episodes": int(failed.sum()),
            "max_pitch_deg": peak,
        }

    def policy(obs):
        return agent.select_action_batch(obs, evaluate=True, return_tensor=True)

    result = {
        "repo": str(args.repo.resolve()),
        "seed": args.seed,
        "frames": args.frames,
        "environment_sha256": hashlib.sha256(args.env_source.read_bytes()).hexdigest(),
        "config": {
            "num_envs": 4,
            "learning_starts": 512,
            "warmup_vector_steps": 128,
            "hidden_size": 64,
            "batch_size": 128,
            "num_quantiles": 8,
            "embedding_dim": 32,
            "alpha": 0.01,
            "automatic_entropy_tuning": False,
            "caps": 0.0,
            "lr": 0.0003,
        },
        "untrained": evaluate(policy),
        "zero": evaluate(lambda obs: torch.zeros(24, 1)),
        "pd": evaluate(
            lambda obs: (-2 * obs[:, 0] + 0.6 * obs[:, 1]).clamp(-1, 1).unsqueeze(1)
        ),
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    step_env = env.step
    expected = {}
    frames = 0

    def monitored_step(action):
        nonlocal frames
        expected["state"] = env._get_obs().clone().numpy()
        output = step_env(action)
        next_obs, _, term, trunc, info = output
        expected["next"] = next_obs.clone().numpy()
        done = (term | trunc).numpy()
        if done.any():
            expected["next"][done] = info["final_observation"][done].numpy()
        expected["terminal"] = term.numpy().astype(float)
        frames += 4
        if not torch.isfinite(next_obs).all():
            raise FloatingPointError("Nonfinite observation")
        return output

    env.step = monitored_step
    update = agent.update_parameters
    maximum_loss = np.zeros(5)
    replay_errors = np.zeros(3)
    curves = []

    def checked_update(memory, batch_size, updates):
        recent = memory.buffer[-4:]
        actual = [
            np.stack([t[0] for t in recent]),
            np.stack([t[3] for t in recent]),
            np.array([t[4] for t in recent]),
        ]
        for index, key in enumerate(["state", "next", "terminal"]):
            replay_errors[index] = max(
                replay_errors[index], float(abs(actual[index] - expected[key]).max())
            )
        losses = np.asarray(update(memory, batch_size, updates))
        if not np.isfinite(losses).all():
            raise FloatingPointError("Nonfinite update loss")
        maximum_loss[:] = np.maximum(maximum_loss, abs(losses))
        if updates and updates % 5000 == 0:
            rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
            metrics = evaluate(policy)
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            curves.append({"frames": frames, "updates": updates, **metrics})
            print(
                json.dumps(
                    {
                        "frames": frames,
                        "rmse": metrics["rmse_deg_mean"],
                        "failures": metrics["failed_episodes"],
                    }
                ),
                flush=True,
            )
        return tuple(losses)

    agent.update_parameters = checked_update
    started = time.monotonic()
    try:
        agent.train_vector(
            total_steps=args.frames // 4, warmup_steps=128, log_every=5000
        )
        result.update(
            trained=evaluate(policy),
            learning_curve=curves,
            environment_steps=frames,
            max_abs_losses=maximum_loss.tolist(),
            replay_max_errors=dict(
                zip(["state", "next_state", "terminal_mask"], replay_errors.tolist())
            ),
        )
        names = ["policy", "Z1", "Z2", "Z1_target", "Z2_target"]
        result["parameters_finite"] = all(
            torch.isfinite(p).all().item()
            for name in names
            for p in getattr(agent, name).parameters()
        )
        if not result["parameters_finite"]:
            raise FloatingPointError("Nonfinite parameters")
        result["weight_sha256"] = {
            name: hashlib.sha256(
                b"".join(
                    v.detach().cpu().numpy().tobytes()
                    for v in getattr(agent, name).state_dict().values()
                )
            ).hexdigest()
            for name in names
        }
        torch.save(
            {name: getattr(agent, name).state_dict() for name in names},
            args.output.with_suffix(".pt"),
        )
    finally:
        result["elapsed_s"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        agent.close()
    print(json.dumps(result["trained"]))


if __name__ == "__main__":
    main()
