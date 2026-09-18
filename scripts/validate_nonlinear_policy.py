"""Train and evaluate SAC pitch tracking on the full nonlinear B737-800.

A scoped longitudinal task at 37,000 ft, with trim throttle held fixed and an
additional elevator authority of +/-3 degrees. This checks the current simulator
and freshly trained policies; it does not certify existing policies or flight data.
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
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import gymnasium as gym
    import numpy as np
    import torch

    from tensoraerospace.aerospacemodel.b737.nonlinear import B737Configuration, trim
    from tensoraerospace.agent.sac.sac import SAC
    from tensoraerospace.envs.b737_nonlinear import NonlinearB737Env

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = B737Configuration.B737_800
    trimmed = trim(altitude_ft=37000.0, V_ft_s=800.0, config=config)
    assert trimmed.converged
    initial = trimmed.to_state()
    command = np.array([trimmed.elevator_rad, 0.0, 0.0, trimmed.throttle])
    dt = 0.05
    horizon = 200

    class TrackingEnv(gym.Env):
        def __init__(self, seed):
            self.physical = NonlinearB737Env(
                initial_state=initial, dt=dt, number_time_steps=horizon, config=config
            )
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            self.rng = np.random.default_rng(seed)
            self.steps = 0

        def observe(self):
            self.theta = float(np.rad2deg(self.state[7] - initial[7]))
            self.q = float(np.rad2deg(self.state[4]))
            self.speed = float(np.linalg.norm(self.state[:3]))
            alpha = float(
                np.rad2deg(np.arctan2(self.state[2], self.state[0]) - trimmed.alpha_rad)
            )
            self.target = self.amplitude if self.index * dt >= self.start else 0.0
            return np.array(
                [
                    (self.theta - self.target) / 5.0,
                    self.q / 10.0,
                    self.theta / 5.0,
                    (self.speed - 800.0) / 50.0,
                    alpha / 5.0,
                ],
                np.float32,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            options = options or {}
            self.amplitude = float(
                options.get("amplitude", self.rng.uniform(-2.0, 2.0))
            )
            self.start = float(options.get("start", self.rng.uniform(0.5, 1.5)))
            self.state, _ = self.physical.reset()
            self.index = 0
            return self.observe(), {}

        def step(self, action):
            action = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
            u = command.copy()
            u[0] += np.deg2rad(3.0) * action
            self.state, _, _, _, _ = self.physical.step(u)
            self.steps += 1
            self.index += 1
            if not np.all(np.isfinite(self.state)):
                raise FloatingPointError("Nonfinite physical state")
            obs = self.observe()
            failed = (
                abs(self.theta) > 15.0
                or not 650.0 < self.speed < 950.0
                or -self.state[11] < 35000.0
            )
            reward = (
                -(((self.theta - self.target) / 2.0) ** 2)
                - 0.02 * action**2
                - 0.02 * self.q**2
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
        alpha=0.2,
        seed=args.seed,
        device="cpu",
        log_dir=args.output.with_suffix(""),
    )
    networks = [agent.policy, agent.critic, agent.critic_target]

    def evaluate(policy):
        evaluation = TrackingEnv(20260917)
        rows = []
        first_trace = []
        for amplitude in (-2.0, -1.0, 1.0, 2.0):
            for start in (0.5, 1.0, 1.5):
                obs, _ = evaluation.reset(
                    options={"amplitude": amplitude, "start": start}
                )
                errors = []
                speeds = []
                altitudes = []
                actions = []
                pitches = []
                reward_total = 0.0
                for step in range(horizon):
                    action = policy(obs)
                    if not np.all(np.isfinite(action)):
                        raise FloatingPointError("Nonfinite policy action")
                    obs, reward, term, trunc, _ = evaluation.step(action)
                    errors.append(evaluation.theta - evaluation.target)
                    speeds.append(evaluation.speed)
                    altitudes.append(-evaluation.state[11])
                    actions.append(float(action[0]))
                    pitches.append(evaluation.theta)
                    reward_total += reward
                    if not rows:
                        first_trace.append(
                            [
                                dt * (step + 1),
                                evaluation.theta,
                                evaluation.target,
                                float(action[0]),
                            ]
                        )
                    if term or trunc:
                        break
                rows.append(
                    dict(
                        amplitude_deg=amplitude,
                        start_s=start,
                        steps=len(errors),
                        failed=term,
                        rmse_deg=float(np.sqrt(np.mean(np.square(errors)))),
                        reward=reward_total,
                        min_speed_ft_s=min(speeds),
                        max_speed_ft_s=max(speeds),
                        min_altitude_ft=min(altitudes),
                        max_altitude_ft=max(altitudes),
                        max_abs_pitch_deviation_deg=max(np.abs(pitches)),
                        saturation_fraction=float(np.mean(np.abs(actions) >= 0.99)),
                    )
                )
        return dict(
            rmse_deg_mean=float(np.mean([r["rmse_deg"] for r in rows])),
            failed_episodes=sum(r["failed"] for r in rows),
            episodes=len(rows),
            episode_metrics=rows,
            first_episode_trace=first_trace,
        )

    policy = lambda obs: agent.select_action(obs, evaluate=True)
    result = dict(
        seed=args.seed,
        repo=str(args.repo.resolve()),
        config=dict(
            training_steps=args.frames,
            altitude_ft=37000.0,
            speed_ft_s=800.0,
            dt_s=dt,
            horizon_s=dt * horizon,
            elevator_delta_limit_deg=3.0,
            hidden_size=64,
            batch_size=128,
            throttle=trimmed.throttle,
            trim_elevator_rad=trimmed.elevator_rad,
        ),
        untrained=evaluate(policy),
        zero_control=evaluate(lambda obs: np.zeros(1)),
        pd_control=evaluate(
            lambda obs: np.array([np.clip(2.0 * obs[0] + 1.5 * obs[1], -1.0, 1.0)])
        ),
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
        while env.steps < args.frames:
            agent.train(
                num_episodes=1, max_steps=args.frames - env.steps, verbose=False
            )
            if not all(
                bool(torch.isfinite(p).all()) for n in networks for p in n.parameters()
            ):
                raise FloatingPointError("Nonfinite weights")
            if env.steps >= next_eval:
                metric = evaluate(policy)
                curves.append(
                    dict(
                        step=env.steps,
                        rmse_deg=metric["rmse_deg_mean"],
                        failed_episodes=metric["failed_episodes"],
                    )
                )
                print(json.dumps(curves[-1]), flush=True)
                next_eval += 5000
        result["trained"] = evaluate(policy)
        result["learning_curve"] = curves
        result["environment_steps"] = env.steps
        result["parameters_finite"] = True
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
