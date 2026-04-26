"""DSAC training on ImprovedB747 for step response.

This script is a practical alternative to the notebook when you want stable,
repeatable training runs.

Key ideas (mirrors what worked for PPO):
- Bootstrap with easier reward, then switch to full *step_response* reward.
- Use vectorized environment (ImprovedB747VecEnvTorch) to decorrelate data.
- Use curriculum: start with moderate steps, then widen to the eval difficulty.
- Prevent the "terminate early to avoid negative rewards" hack:
  - Keep terminal penalties un-clipped in DSAC (implemented in the agent).
  - Add an early-termination penalty per remaining step in the env.

Run:
    python example/reinforcement_learning/train_dsac_b747_step_response.py

TensorBoard:
    tensorboard --logdir runs
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensoraerospace.agent import DSAC
from tensoraerospace.agent.sac.replay_memory import ReplayMemory
from tensoraerospace.envs import ImprovedB747VecEnvTorch
from tensoraerospace.envs.b747 import ImprovedB747Env


def load_dsac_checkpoint(folder: Path, env: Any, *, device: str = "cpu") -> DSAC:
    """Minimal loader to warm-start from a previous best_eval checkpoint."""
    folder = Path(folder)
    config_path = folder / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {folder}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    params = dict(cfg["policy"]["params"])
    params["device"] = device

    agent = DSAC(env=env, **params)

    # Load state_dict-based checkpoints (dsac-flight style)
    policy_sd = torch.load(
        folder / "policy.pth", map_location=device, weights_only=True
    )
    critic_sd = torch.load(
        folder / "critic.pth", map_location=device, weights_only=True
    )
    critic_t_sd = torch.load(
        folder / "critic_target.pth", map_location=device, weights_only=True
    )

    agent.policy.load_state_dict(policy_sd)
    agent.Z1.load_state_dict(critic_sd["Z1"])
    agent.Z2.load_state_dict(critic_sd["Z2"])
    agent.Z1_target.load_state_dict(critic_t_sd["Z1_target"])
    agent.Z2_target.load_state_dict(critic_t_sd["Z2_target"])
    agent.policy = agent.policy.to(device)
    agent.Z1 = agent.Z1.to(device)
    agent.Z2 = agent.Z2.to(device)
    agent.Z1_target = agent.Z1_target.to(device)
    agent.Z2_target = agent.Z2_target.to(device)

    # Alpha
    log_alpha_path = folder / "log_alpha.pth"
    if getattr(agent, "automatic_entropy_tuning", False) and log_alpha_path.exists():
        loaded_alpha = torch.load(
            log_alpha_path, map_location=device, weights_only=True
        )
        log_alpha_t = getattr(agent, "log_alpha", None)
        if (
            isinstance(loaded_alpha, dict)
            and "log_alpha" in loaded_alpha
            and log_alpha_t is not None
        ):
            log_alpha_t.data.copy_(loaded_alpha["log_alpha"].to(agent.device))
            agent.alpha = float(log_alpha_t.exp().item())

    agent.policy.eval()
    agent.Z1.eval()
    agent.Z2.eval()
    agent.Z1_target.eval()
    agent.Z2_target.eval()
    return agent


def find_latest_metrics(runs_root: Path) -> Path | None:
    candidates = sorted(
        runs_root.glob("dsac_b747_step_response_*/best_eval/metrics.json")
    )
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def make_reference(
    *,
    dt: float,
    tn: float,
    step_deg: float,
    step_time_sec: float,
) -> np.ndarray:
    """Step reference in radians, shape (1, T)."""
    num_steps = int(tn / dt) + 1
    ref = np.zeros((1, num_steps), dtype=np.float32)
    step_idx = int(np.clip(round(step_time_sec / dt), 0, num_steps - 1))
    ref[:, step_idx:] = np.deg2rad(step_deg)
    return ref


def eval_one_episode(agent: DSAC, env: ImprovedB747Env) -> dict:
    obs, _ = env.reset()
    rewards: list[float] = []
    terminated = False
    truncated = False
    last_info: dict = {}
    while not (terminated or truncated):
        action = agent.select_action(obs, evaluate=True)
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(float(r))
        last_info = info
    return {
        "return_sum": float(np.sum(rewards)),
        "return_mean": float(np.mean(rewards)) if rewards else 0.0,
        "steps": int(len(rewards)),
        "settled": bool(last_info.get("settled", False)),
        "settle_time_s": float(last_info.get("settle_time_s", -1.0)),
        "overshoot_ratio": float(last_info.get("overshoot_ratio", 0.0)),
        "sign_changes": int(last_info.get("sign_changes", 0)),
    }


def save_eval_best(agent: DSAC, best_root: Path, metrics: dict) -> Path:
    """Save a single "best" checkpoint (overwrite previous).

    DSAC.save() always creates a timestamped subfolder. We keep only one by
    clearing best_root before saving, then writing metrics.json.
    """
    best_root.mkdir(parents=True, exist_ok=True)
    for p in list(best_root.iterdir()):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    agent.save(path=best_root, save_gradients=False)
    subdirs = [p for p in best_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(
            f"Expected a checkpoint folder under {best_root}, but found none."
        )
    saved_dir = max(subdirs, key=lambda p: p.stat().st_mtime)

    payload = dict(metrics)
    payload["checkpoint_dir"] = str(saved_dir.resolve())
    with open(best_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return saved_dir


def main() -> None:
    # --------------------------
    # Global config (PPO-like)
    # --------------------------
    seed = 0
    dt = 0.1
    tn = 20.0
    step_time_sec = 5.0
    num_envs = 128

    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------
    # Train env (vectorized) — start in tracking mode, then switch to step_response
    # --------------------------
    # Shaping to discourage "finish early" hacks (applies only on termination).
    completion_bonus = 5.0
    early_term_penalty_per_step = (
        1.0  # increase to 2..5 if agent still terminates early
    )

    env_train = ImprovedB747VecEnvTorch(
        num_envs=num_envs,
        dt=dt,
        tn=tn,
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        device=device,
        seed=seed,
        auto_reset=True,
        include_reference_in_obs=True,
        reward_mode="tracking",  # Stage 0: start with tracking for stability
        early_termination_penalty=0.0,
        survival_bonus=0.0,
        completion_bonus=0.0,  # off for tracking
        early_termination_penalty_per_step=0.0,  # off for tracking
        step_randomization={
            # Stage 0: pure tracking uses sine; these are placeholders, will switch later
            "amplitude_deg_range": (-5.0, 5.0),
            "min_abs_amplitude_deg": 0.1,
            "step_time_sec_range": (1, 15),
        },
    )

    # --------------------------
    # Eval envs (single, fixed)
    # --------------------------
    env_eval_tracking = ImprovedB747Env(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        reference_signal=make_reference(
            dt=dt, tn=tn, step_deg=0.0, step_time_sec=0.0
        ),  # will be overridden by tracking sine via include_reference
        number_time_steps=int(tn / dt) + 1,
        dt=dt,
        reward_mode="tracking",
        include_reference_in_obs=True,
    )
    env_eval_step = ImprovedB747Env(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        reference_signal=make_reference(
            dt=dt, tn=tn, step_deg=1.0, step_time_sec=step_time_sec
        ),
        number_time_steps=int(tn / dt) + 1,
        dt=dt,
        reward_mode="step_response",
        include_reference_in_obs=True,
    )

    # --------------------------
    # DSAC agent
    # --------------------------
    run_name = f"dsac_b747_step_response_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir = log_dir / "best_checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_best_dir = log_dir / "best_eval"
    eval_best = float("-inf")

    # Optional: warm start from previous best_eval checkpoint
    resume_metrics = find_latest_metrics(Path("runs"))
    resume_checkpoint_dir = None
    if resume_metrics is not None:
        try:
            with open(resume_metrics, "r", encoding="utf-8") as f:
                m = json.load(f)
            resume_checkpoint_dir = Path(m["checkpoint_dir"])
            print(f"Resuming from best_eval: {resume_checkpoint_dir}")
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"Could not load resume metrics ({resume_metrics}): {exc}")

    agent = None
    if resume_checkpoint_dir is not None and resume_checkpoint_dir.exists():
        try:
            agent = load_dsac_checkpoint(
                resume_checkpoint_dir, env_train, device=device
            )
            print("Loaded agent weights from previous best_eval checkpoint.")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Failed to load previous best_eval checkpoint: {exc}")
            agent = None

    if agent is None:
        agent = DSAC(
            env_train,
            # Replay / updates
            batch_size=256,
            memory_capacity=1_000_000,
            # Delay updates for stability on harsh step_response reward
            learning_starts=100_000,
            updates_per_step=4,
            lr=4.4e-4,
            policy_lr=4.4e-4,
            gamma=0.995,
            tau=0.005,
            # IQN params (dsac-flight defaults)
            num_quantiles=8,
            num_quantiles_exp=8,
            embedding_dim=64,
            hidden_layers=[64, 64],
            huber_threshold=1.0,
            # Entropy / temperature
            automatic_entropy_tuning=True,
            target_entropy_scale=1.0,
            min_alpha=0.0,
            # Helps on step_response with large penalties
            reward_clip=50.0,
            # No extra exploration noise
            exploration_noise_std=0.0,
            # CAPS: env already penalizes u/du/ddu for B747 -> keep off
            caps_lambda_smoothness=0.0,
            caps_lambda_temporal=0.0,
            caps_noise_std=0.05,
            # Risk distortion: keep neutral for now
            risk_distortion="neutral",
            # Misc
            device=device,
            log_dir=log_dir,
            log_every_updates=200,
            seed=seed,
        )
    else:
        # If we resumed from an old checkpoint, redirect TensorBoard to this run's log_dir
        try:
            agent.writer.close()
        except Exception:
            pass
        from torch.utils.tensorboard import SummaryWriter

        agent.log_dir = log_dir
        agent.writer = SummaryWriter(log_dir=str(log_dir))

    # --------------------------
    # Training curriculum
    # --------------------------
    print(f"Device: {device}")
    print(f"Log dir: {log_dir.resolve()}")
    print(f"Best checkpoints dir: {best_dir.resolve()}")
    print("Stage 0: tracking reward (bootstrap, sine-like behavior via ref in obs)")
    env_train.reward_mode = "tracking"
    env_train.completion_bonus = 0.0
    env_train.early_termination_penalty_per_step = 0.0
    env_train.early_termination_penalty = 0.0
    agent.train_vector(
        total_steps=10_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m0 = eval_one_episode(agent, env_eval_tracking)
    print("Eval after Stage 0:", m0)
    if float(m0["return_sum"]) > float(eval_best):
        eval_best = float(m0["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m0)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    # IMPORTANT: switching reward changes the MDP;
    # drop old replay to avoid mixing
    print("Resetting replay buffer before step_response...")
    agent.memory = ReplayMemory(agent.memory.capacity, seed=seed)
    # Tame outliers from step_response penalties
    agent.reward_clip = 20.0

    # Stage 1: step_response reward (lite -> full)
    print("Stage 1a: step_response (lite penalties, looser band)")
    env_train.reward_mode = "step_response"
    env_train.completion_bonus = float(completion_bonus)
    env_train.early_termination_penalty_per_step = float(early_term_penalty_per_step)
    # Loosen step-response targets to reduce reward discontinuities early on
    env_train.overshoot_limit_ratio = 0.2
    env_train.settle_band_ratio = 0.05
    env_train.settle_band_min_rad = float(np.deg2rad(0.2))
    env_train.w_abs = 0.2
    env_train.w_time = 0.2
    env_train.w_osc = 0.2
    env_train.w_overshoot = 50.0

    env_train.step_rand.amplitude_deg_range = (-5.0, 5.0)
    env_train.step_rand.min_abs_amplitude_deg = 0.5
    env_train.step_rand.step_time_sec_range = (1.0, 15.0)
    agent.train_vector(
        total_steps=10_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )

    m1a = eval_one_episode(agent, env_eval_step)
    print("Eval after Stage 1a:", m1a)
    if float(m1a["return_sum"]) > float(eval_best):
        eval_best = float(m1a["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m1a)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    print("Stage 1b: step_response (full penalties)")
    env_train.overshoot_limit_ratio = 0.05
    env_train.settle_band_ratio = 0.01
    env_train.settle_band_min_rad = float(np.deg2rad(0.05))
    env_train.w_abs = 0.6
    env_train.w_time = 0.6
    env_train.w_osc = 1.0
    env_train.w_overshoot = 300.0
    agent.train_vector(
        total_steps=20_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )

    m1b = eval_one_episode(agent, env_eval_step)
    print("Eval after Stage 1b:", m1b)
    if float(m1b["return_sum"]) > float(eval_best):
        eval_best = float(m1b["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m1b)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    # Stage 2: step_response without extra completion/early penalties (robustness)
    print("Stage 2: step_response without completion/early penalties")
    env_train.completion_bonus = 0.0
    env_train.early_termination_penalty_per_step = 0.0
    env_train.early_termination_penalty = 0.0
    agent.reward_clip = 20.0
    agent.train_vector(
        total_steps=30_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )

    m2 = eval_one_episode(agent, env_eval_step)
    print("Eval after Stage 2:", m2)
    if float(m2["return_sum"]) > float(eval_best):
        eval_best = float(m2["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m2)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    agent.close()

    print("\nDone.")
    print("Tip: open TensorBoard and watch Performance/RewardMedian200.")


if __name__ == "__main__":
    main()
