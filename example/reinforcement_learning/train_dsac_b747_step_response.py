"""DSAC training on ImprovedB747 (step_response) that avoids the common -12 plateau.

This script is a practical alternative to the notebook when you want stable,
repeatable training runs.

Key ideas (mirrors what worked for PPO):
- Train on *step_response* directly (the objective we care about).
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

import numpy as np
import torch

from tensoraerospace.agent import DSAC
from tensoraerospace.envs import ImprovedB747VecEnvTorch
from tensoraerospace.envs.b747 import ImprovedB747Env


def load_dsac_checkpoint(folder: Path, env: ImprovedB747Env, *, device: str = "cpu") -> DSAC:
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

    # Weights
    agent.critic = torch.load(folder / "critic.pth", map_location=device, weights_only=False)
    agent.policy = torch.load(folder / "policy.pth", map_location=device, weights_only=False)
    agent.critic_target = torch.load(folder / "critic_target.pth", map_location=device, weights_only=False)
    agent.critic = agent.critic.to(device)
    agent.policy = agent.policy.to(device)
    agent.critic_target = agent.critic_target.to(device)

    # Alpha
    log_alpha_path = folder / "log_alpha.pth"
    if getattr(agent, "automatic_entropy_tuning", False) and log_alpha_path.exists():
        loaded_alpha = torch.load(log_alpha_path, map_location=device, weights_only=False)
        if isinstance(loaded_alpha, dict) and "log_alpha" in loaded_alpha:
            agent.log_alpha.data.copy_(loaded_alpha["log_alpha"].to(agent.device))
            agent.alpha = float(agent.log_alpha.exp().item())

    # Validate shapes: policy_target and policy must align; same for critics.
    def _same_shapes(m1: torch.nn.Module, m2: torch.nn.Module) -> bool:
        return all(p1.shape == p2.shape for p1, p2 in zip(m1.parameters(), m2.parameters()))

    if not _same_shapes(agent.policy_target, agent.policy):
        raise ValueError("Loaded policy and policy_target shapes differ; likely incompatible checkpoint.")
    if not _same_shapes(agent.critic_target, agent.critic):
        raise ValueError("Loaded critic and critic_target shapes differ; likely incompatible checkpoint.")

    agent.policy.eval()
    agent.critic.eval()
    agent.critic_target.eval()
    return agent


def find_latest_metrics(runs_root: Path) -> Path | None:
    candidates = sorted(runs_root.glob("dsac_b747_step_response_*/best_eval/metrics.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
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
        raise RuntimeError(f"Expected a checkpoint folder under {best_root}, but found none.")
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
    # Train env (vectorized)
    # --------------------------
    # Shaping to discourage "finish early" hacks (applies only on termination).
    completion_bonus = 5.0
    early_term_penalty_per_step = 1.0  # increase to 2..5 if agent still terminates early

    env_train = ImprovedB747VecEnvTorch(
        num_envs=num_envs,
        dt=dt,
        tn=tn,
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        device=device,
        seed=seed,
        auto_reset=True,
        reward_mode="step_response",
        survival_bonus=0.0,
        completion_bonus=float(completion_bonus),
        early_termination_penalty=0.0,
        early_termination_penalty_per_step=float(early_term_penalty_per_step),
        step_randomization={
            # Stage 1: learn meaningful control (avoid tiny steps)
            "amplitude_deg_range": (-5.0, 5.0),
            "min_abs_amplitude_deg": 0.1,
            "step_time_sec_range": (1, 15),
        },
    )

    # --------------------------
    # Eval env (single, fixed)
    # --------------------------
    env_eval = ImprovedB747Env(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        reference_signal=make_reference(
            dt=dt, tn=tn, step_deg=1.0, step_time_sec=step_time_sec
        ),
        number_time_steps=int(tn / dt) + 1,
        dt=dt,
        reward_mode="step_response",
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
        except Exception as exc:
            print(f"Could not load resume metrics ({resume_metrics}): {exc}")

    agent = None
    if resume_checkpoint_dir is not None and resume_checkpoint_dir.exists():
        try:
            agent = load_dsac_checkpoint(resume_checkpoint_dir, env_train, device=device)
            print("Loaded agent weights from previous best_eval checkpoint.")
        except Exception as exc:
            print(f"Failed to load previous best_eval checkpoint: {exc}")
            agent = None

    if agent is None:
        agent = DSAC(
            env_train,
            # Replay / updates
            batch_size=256,
            memory_capacity=1_000_000,
            learning_starts=50_000,  # ~3k vec-steps with num_envs=16
            warmup_action_scale=0.6,
            updates_per_step=4,
            reward_clip=50.0,  # clips non-terminal rewards only (terminal stays un-clipped)
            # Critics (distributional)
            num_quantiles=32,
            embedding_dim=32,
            hidden_layers=[64, 64],
            huber_threshold=1.0,
            # Actor net (SAC-style)
            hidden_size=64,
            # Optim
            lr=1e-4,
            policy_lr=1e-4,
            max_grad_norm=10.0,
            # Discount
            gamma=0.995,
            tau=0.005,
            # Exploration / entropy
            alpha=0.2,
            automatic_entropy_tuning=True,
            target_entropy_scale=1.0,
            min_alpha=0.01,
            exploration_noise_std=0.01,
            # CAPS: env already penalizes u/du/ddu -> keep off for B747
            caps_lambda_smoothness=0.0,
            caps_lambda_temporal=0.0,
            caps_noise_std=0.05,
            # Misc
            device=device,
            log_dir=log_dir,
            log_every_updates=200,
            seed=seed,
        )

    # --------------------------
    # Training curriculum
    # --------------------------
    print(f"Device: {device}")
    print(f"Log dir: {log_dir.resolve()}")
    print(f"Best checkpoints dir: {best_dir.resolve()}")
    print("Stage 1: ±3deg @ 5s")
    agent.train_vector(
        total_steps=40_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m1 = eval_one_episode(agent, env_eval)
    print("Eval after Stage 1:", m1)
    if float(m1["return_sum"]) > float(eval_best):
        eval_best = float(m1["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m1)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    # Stage 2: widen to match eval difficulty (±5deg and random step time)
    print("Stage 2: ±5deg and random step time (3..7s)")
    env_train.step_rand.amplitude_deg_range = (-5.0, 5.0)
    env_train.step_rand.min_abs_amplitude_deg = 0.5
    env_train.step_rand.step_time_sec_range = (1.0, 15.0)
    # Reduce extra action noise for fine-tuning
    agent.exploration_noise_std = 0.0
    agent.train_vector(
        total_steps=80_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m2 = eval_one_episode(agent, env_eval)
    print("Eval after Stage 2:", m2)
    if float(m2["return_sum"]) > float(eval_best):
        eval_best = float(m2["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m2)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    agent.close()

    print("\nDone.")
    print("Tip: open TensorBoard and watch Performance/RewardMedian200 (robust).")


if __name__ == "__main__":
    main()


