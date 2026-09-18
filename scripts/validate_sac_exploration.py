"""Measure bounded SAC exploration with an exact environment-step budget.

Run one bounded CPU experiment per process. Compare native evaluation with an
external RNG-preserving control on the same fixed B747 environment source.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--preserve-rng", action="store_true")
    parser.add_argument(
        "--policy-type", choices=["Gaussian", "Deterministic"], default="Deterministic"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.agent.sac import SAC

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("fixed_b747", args.env_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def make_plant(n, seed):
        return module.ImprovedB747VecEnvTorch(
            num_envs=n,
            dt=0.05,
            tn=10.0,
            device="cpu",
            seed=seed,
            auto_reset=False,
            include_reference_in_obs=True,
            reward_mode="tracking",
            step_randomization={
                "signal_type": "step",
                "amplitude_deg_range": (-4.0, 4.0),
                "min_abs_amplitude_deg": 1.0,
                "step_time_sec_range": (0.5, 1.5),
            },
        )

    class Scalar(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def __init__(self):
            self.plant = make_plant(1, args.seed)
            self.steps = 0

        def reset(self, **kwargs):
            obs, info = self.plant.reset()
            return obs[0].numpy().copy(), info

        def step(self, action):
            obs, reward, term, trunc, info = self.plant.step(
                torch.as_tensor(action).reshape(1, 1)
            )
            self.steps += 1
            return (
                obs[0].numpy().copy(),
                float(reward[0]),
                bool(term[0]),
                bool(trunc[0]),
                info,
            )

    env = Scalar()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent = SAC(
        env,
        device="cpu",
        seed=args.seed,
        policy_type=args.policy_type,
        hidden_size=64,
        batch_size=128,
        memory_capacity=50000,
        alpha=0.01,
        log_every_updates=100,
        log_dir=args.output.with_suffix(""),
    )
    contract = agent.writer.assert_contract_satisfied
    agent.writer.assert_contract_satisfied = lambda: None

    def evaluate(controller):
        plant = make_plant(24, 20260916)
        plant.reset()
        amplitudes = torch.tensor([-4.0, -2.0, 2.0, 4.0]).repeat(6)
        starts = torch.linspace(0.5, 1.5, 24)
        plant.reference_signal[:] = torch.deg2rad(amplitudes[:, None]) * (
            plant.tps[None, :] >= starts[:, None]
        )
        obs = plant._get_obs()
        squares = np.zeros(24)
        counts = np.zeros(24)
        active = np.ones(24, bool)
        failed = np.zeros(24, bool)
        peak = 0.0
        for step in range(plant.number_time_steps - 2):
            action = controller(obs)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Nonfinite policy action")
            obs, _, term, trunc, _ = plant.step(action)
            pitch = np.rad2deg(plant.state[:, 3].numpy())
            error = pitch - np.rad2deg(plant.reference_signal[:, step + 1].numpy())
            squares[active] += error[active] ** 2
            counts[active] += 1
            if active.any():
                peak = max(peak, float(np.abs(pitch[active]).max()))
            failed |= active & term.numpy()
            active &= ~(term | trunc).numpy()
        errors = np.sqrt(squares / np.maximum(counts, 1))
        return {
            "rmse_deg_mean": float(errors.mean()),
            "episode_rmse_deg": errors.tolist(),
            "failed_episodes": int(failed.sum()),
            "episodes": 24,
            "max_pitch_deg": peak,
        }

    def policy(obs):
        return agent.select_action_batch(obs, evaluate=True, return_tensor=True)

    result = {
        "policy_type": args.policy_type,
        "seed": args.seed,
        "repo": str(args.repo.resolve()),
        "frames": args.frames,
        "preserve_rng": args.preserve_rng,
        "environment_sha256": hashlib.sha256(args.env_source.read_bytes()).hexdigest(),
        "untrained": evaluate(policy),
        "zero": evaluate(lambda obs: torch.zeros((24, 1))),
        "pd": evaluate(
            lambda obs: (-2 * obs[:, 0] + 0.6 * obs[:, 1]).clamp(-1, 1).unsqueeze(1)
        ),
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    curves = []
    update = agent.update_parameters
    max_losses = np.zeros(5)

    def checked_update(*pos, **kw):
        values = np.asarray(update(*pos, **kw))
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite training loss")
        max_losses[:] = np.maximum(max_losses, np.abs(values))
        return tuple(values)

    agent.update_parameters = checked_update
    started = time.monotonic()
    next_evaluation = 5000
    try:
        while env.steps < args.frames:
            agent.train(
                num_episodes=1, max_steps=args.frames - env.steps, verbose=False
            )
            if env.steps >= next_evaluation:
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                metrics = evaluate(policy)
                changed = not torch.equal(rng[2], torch.get_rng_state())
                if args.preserve_rng:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                curves.append(
                    {"step": env.steps, "rng_changed_by_evaluation": changed, **metrics}
                )
                next_evaluation += 5000
                print(
                    json.dumps(
                        {
                            "step": env.steps,
                            "rmse": metrics["rmse_deg_mean"],
                            "failed": metrics["failed_episodes"],
                        }
                    ),
                    flush=True,
                )
        contract()
        result["trained"] = evaluate(policy)
        result["learning_curve"] = curves
        result["max_abs_losses"] = max_losses.tolist()
        result["environment_steps"] = env.steps
        result["gradient_updates"] = (
            agent.total_updates if hasattr(agent, "total_updates") else None
        )
        result["parameters_finite"] = all(
            torch.isfinite(p).all().item()
            for network in [agent.policy, agent.critic, agent.critic_target]
            for p in network.parameters()
        )
        if not result["parameters_finite"]:
            raise FloatingPointError("Nonfinite network parameters")
        result["weight_sha256"] = {
            name: hashlib.sha256(
                b"".join(
                    t.detach().cpu().numpy().tobytes()
                    for t in getattr(agent, name).state_dict().values()
                )
            ).hexdigest()
            for name in ["policy", "critic", "critic_target"]
        }
        torch.save(
            {
                name: getattr(agent, name).state_dict()
                for name in ["policy", "critic", "critic_target"]
            },
            args.output.with_suffix(".pt"),
        )
    finally:
        result["elapsed_s"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        agent.close()
    print(json.dumps(result["trained"]))


if __name__ == "__main__":
    main()
