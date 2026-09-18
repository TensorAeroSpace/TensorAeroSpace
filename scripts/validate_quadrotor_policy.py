"""SAC altitude tracking on a nonlinear quadrotor with symmetric rotor faults.

Effectiveness is an observed simulation variable. This scoped experiment does
not claim flight-ready fault detection or recovery from an asymmetric rotor loss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-weights", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.aerospacemodel.quadrotor.damage import (
        DamageProfile,
        MotorEfficiencyDecay,
        RotorDamageEvent,
    )
    from tensoraerospace.agent.sac.sac import SAC
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dt = 0.02
    horizon = 200

    class TrackingEnv(gym.Env):
        def __init__(self, seed):
            initial = np.zeros(12)
            initial[2] = -10.0
            self.physical = NonlinearQuadrotorEnv(initial, horizon, dt=dt)
            self.rng = np.random.default_rng(seed)
            self.steps = 0
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def observe(self):
            self.height = -float(self.state[2])
            self.velocity = -float(self.state[5])
            self.target = 10.0 + (self.amplitude if self.index * dt >= 0.5 else 0.0)
            self.mu = float(self.physical.damage_manager.state.mu.mean())
            return np.array(
                [(self.height - self.target) / 2.0, self.velocity / 3.0, self.mu],
                np.float32,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            options = options or {}
            self.amplitude = float(
                options.get("amplitude", self.rng.uniform(-1.0, 1.0))
            )
            kind = options.get("kind", self.rng.choice(["healthy", "loss", "decay"]))
            trigger = float(options.get("trigger", self.rng.uniform(0.7, 1.3)))
            fraction = float(options.get("fraction", self.rng.uniform(0.7, 0.9)))
            if kind == "loss":
                events = [RotorDamageEvent(trigger, i, mu=fraction) for i in range(4)]
            elif kind == "decay":
                events = [
                    MotorEfficiencyDecay(trigger, i, tau=0.5, mu_floor=fraction)
                    for i in range(4)
                ]
            else:
                events = []
            self.state, _ = self.physical.reset(
                options={"damage_profile": DamageProfile(events)}
            )
            self.index = 0
            return self.observe(), {}

        def step(self, action):
            action = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
            thrust = (action + 1.0) * 15.0
            self.state, _, _, _, _ = self.physical.step([thrust, 0.0, 0.0, 0.0])
            self.steps += 1
            self.index += 1
            if not np.all(np.isfinite(self.state)):
                raise FloatingPointError("Nonfinite physical state")
            obs = self.observe()
            failed = (
                abs(self.height - 10.0) > 5.0
                or abs(self.velocity) > 8.0
                or np.max(np.abs(self.state[6:8])) > 0.5
            )
            reward = (
                -((self.height - self.target) ** 2)
                - 0.05 * self.velocity**2
                - 0.005 * action**2
            )
            return obs, float(reward), bool(failed), bool(self.index >= horizon), {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = TrackingEnv(args.seed)
    agent = SAC(
        env,
        hidden_size=64,
        batch_size=128,
        memory_capacity=50000,
        automatic_entropy_tuning=True,
        alpha=0.1,
        seed=args.seed,
        device="cpu",
        log_dir=args.output.with_suffix(""),
    )
    if args.load_weights:
        weights = torch.load(args.load_weights, map_location="cpu", weights_only=True)
        agent.policy.load_state_dict(weights["policy"])
        agent.critic.load_state_dict(weights["critic"])
    policy = lambda obs: agent.select_action(obs, evaluate=True)

    def evaluate(control):
        # Evaluation must not consume training RNG state through SAC.sample().
        rng_state = torch.random.get_rng_state()
        try:
            evaluation = TrackingEnv(20260917)
            rows = []
            trace = []
            for kind in ["healthy", "loss", "decay"]:
                for amplitude in [-1.0, -0.5, 0.5, 1.0]:
                    obs, _ = evaluation.reset(
                        options=dict(
                            kind=kind, amplitude=amplitude, trigger=1.015, fraction=0.75
                        )
                    )
                    errors = []
                    heights = []
                    velocities = []
                    saturated = 0
                    for step in range(horizon):
                        action = control(obs)
                        if not np.all(np.isfinite(action)):
                            raise FloatingPointError("Nonfinite policy action")
                        obs, _, term, trunc, _ = evaluation.step(action)
                        errors.append(evaluation.height - evaluation.target)
                        heights.append(evaluation.height)
                        velocities.append(evaluation.velocity)
                        saturated += abs(float(action[0])) >= 0.99
                        if kind == "loss" and amplitude == 1.0:
                            trace.append(
                                [
                                    (step + 1) * dt,
                                    evaluation.height,
                                    evaluation.target,
                                    evaluation.mu,
                                    float(action[0]),
                                ]
                            )
                        if term or trunc:
                            break
                    rows.append(
                        dict(
                            kind=kind,
                            amplitude_m=amplitude,
                            rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                            failed=term,
                            steps=len(errors),
                            min_height_m=min(heights),
                            max_height_m=max(heights),
                            max_abs_velocity_m_s=max(np.abs(velocities)),
                            saturation_fraction=saturated / len(errors),
                        )
                    )
            return dict(
                rmse_m_mean=float(np.mean([r["rmse_m"] for r in rows])),
                failed_episodes=sum(r["failed"] for r in rows),
                episodes=len(rows),
                episode_metrics=rows,
                loss_episode_trace=trace,
            )
        finally:
            torch.random.set_rng_state(rng_state)

    # error = obs[0]*2, velocity = obs[1]*3; compensate observed effectiveness.
    pd = lambda obs: np.array(
        [
            np.clip(
                1.5
                * (9.81 - 6.0 * obs[0] - 7.5 * obs[1])
                / max(float(obs[2]), 0.1)
                / 15.0
                - 1.0,
                -1.0,
                1.0,
            )
        ]
    )
    result = dict(
        seed=args.seed,
        repo=str(args.repo.resolve()),
        config=dict(
            training_steps=args.frames,
            dt_s=dt,
            horizon_s=dt * horizon,
            hidden_size=64,
            batch_size=128,
            alpha=0.1,
            automatic_entropy_tuning=True,
            observed_effectiveness=True,
        ),
        initial=evaluate(policy),
        hover_control=evaluate(lambda obs: np.array([14.715 / 15.0 - 1.0])),
        pd_control=evaluate(pd),
    )
    losses = []
    curves = []
    original = agent.writer.add_scalar

    def monitor(tag, value, *pos, **kwargs):
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"Nonfinite metric {tag}")
        if tag.startswith("loss/"):
            losses.append(dict(step=env.steps, tag=tag, value=float(value)))
        return original(tag, value, *pos, **kwargs)

    agent.writer.add_scalar = monitor
    started = time.monotonic()
    next_eval = 5000
    try:
        if not args.load_weights:
            while env.steps < args.frames:
                agent.train(
                    num_episodes=1, max_steps=args.frames - env.steps, verbose=False
                )
                if not all(
                    bool(torch.isfinite(p).all())
                    for n in [agent.policy, agent.critic, agent.critic_target]
                    for p in n.parameters()
                ):
                    raise FloatingPointError("Nonfinite weights")
                if env.steps >= next_eval:
                    m = evaluate(policy)
                    curves.append(
                        dict(
                            step=env.steps,
                            rmse_m=m["rmse_m_mean"],
                            failed_episodes=m["failed_episodes"],
                        )
                    )
                    print(json.dumps(curves[-1]), flush=True)
                    next_eval += 5000
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "critic": agent.critic.state_dict(),
                },
                args.output.with_suffix(".pt"),
            )
        result["evaluated"] = evaluate(policy)
        result["learning_curve"] = curves
        result["environment_steps"] = env.steps
        result["parameters_finite"] = True
        result["loss_samples"] = losses[:: max(1, len(losses) // 200)]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        agent.writer.close()
    print(
        json.dumps(
            {k: v for k, v in result["evaluated"].items() if not isinstance(v, list)}
        )
    )


if __name__ == "__main__":
    main()
