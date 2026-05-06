"""DSAC training on ImprovedB747 for sine tracking (+/-1 deg), dt=0.1s,
f=0.05 Hz.

What this does:
- Builds a fixed sinusoid in [-1 deg, 1 deg] (dt=0.1 s, f=0.05 Hz).
- Saves a reference preview (reference_signal.png) before training starts.
- Trains DSAC in tracking mode on a vectorized sine environment.
- Evaluates on the fixed sine and saves best checkpoint in log_dir/best_eval.

Run:
    python example/reinforcement_learning/train_dsac_b747_tracking.py

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
from matplotlib import pyplot as plt

from tensoraerospace.agent import DSAC
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
        runs_root.glob("dsac_b747_sine_tracking_refobs_*/best_eval/metrics.json")
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


def make_sine_reference(
    *,
    dt: float,
    tn: float,
    amplitude_deg: float,
    frequency_hz: float,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """Sine reference in radians, shape (1, T)."""
    num_steps = int(tn / dt) + 1
    t = np.arange(num_steps, dtype=np.float32) * float(dt)
    amp_rad = np.deg2rad(float(amplitude_deg))
    phase_rad = np.deg2rad(float(phase_deg))
    ref = amp_rad * np.sin(2.0 * np.pi * float(frequency_hz) * t + phase_rad)
    return np.asarray(ref, dtype=np.float32).reshape(1, -1)


def save_reference_plot(ref_rad: np.ndarray, dt: float, path: Path) -> Path:
    """Save a quick preview of the reference signal in degrees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(ref_rad.shape[1]) * float(dt)
    ref_deg = np.rad2deg(ref_rad.squeeze(0))
    plt.figure(figsize=(8, 3))
    plt.plot(t, ref_deg, label="theta_ref (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (deg)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.title("Sine reference (tracking)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def show_reference_plot(ref_rad: np.ndarray, dt: float) -> None:
    """Display the reference plot and block until the window is closed."""
    t = np.arange(ref_rad.shape[1]) * float(dt)
    ref_deg = np.rad2deg(ref_rad.squeeze(0))
    plt.figure(figsize=(8, 3))
    plt.plot(t, ref_deg, label="theta_ref (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (deg)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.title("Sine reference (tracking)")
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    # Global config
    # --------------------------
    seed = 0
    dt = 0.1
    tn = 20.0  # 1 period at 0.05 Hz
    sine_amplitude_deg = 1.0
    sine_frequency_hz = 0.05
    num_envs = 128

    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------
    # DSAC agent / logging
    # --------------------------
    run_name = f"dsac_b747_sine_tracking_refobs_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir = log_dir / "best_checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_best_dir = log_dir / "best_eval"
    eval_best = float("-inf")

    # Reference (fixed) + preview
    reference_eval = make_sine_reference(
        dt=dt,
        tn=tn,
        amplitude_deg=sine_amplitude_deg,
        frequency_hz=sine_frequency_hz,
    )
    plot_path = save_reference_plot(
        reference_eval, dt, log_dir / "reference_signal.png"
    )
    print(
        f"Reference preview saved to {plot_path.resolve()} "
        f"(+/-{sine_amplitude_deg} deg, "
        f"f={sine_frequency_hz} Hz, dt={dt}s, T={tn}s)"
    )
    # Block here until the user closes the preview window
    show_reference_plot(reference_eval, dt)

    # --------------------------
    # Train env (vectorized)
    # --------------------------
    train_amp_stage1_deg = 1.0
    train_amp_stage2_deg = 1.0
    env_train = ImprovedB747VecEnvTorch(
        num_envs=num_envs,
        dt=dt,
        tn=tn,
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        device=device,
        seed=seed,
        auto_reset=True,
        include_reference_in_obs=True,
        reward_mode="tracking",
        survival_bonus=0.0,
        completion_bonus=0.0,
        early_termination_penalty=0.0,
        early_termination_penalty_per_step=0.0,
        step_randomization={
            "signal_type": "sine",
            "amplitude_deg_range": (
                -train_amp_stage1_deg,
                train_amp_stage1_deg,
            ),
            "min_abs_amplitude_deg": 0.5 * train_amp_stage1_deg,
            "frequency_hz_range": (sine_frequency_hz, sine_frequency_hz),
        },
    )

    # --------------------------
    # Eval env (single, fixed)
    # --------------------------
    env_eval = ImprovedB747Env(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        reference_signal=reference_eval,
        number_time_steps=int(tn / dt) + 1,
        dt=dt,
        include_reference_in_obs=True,
        reward_mode="tracking",
    )

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
            # Params aligned with dsac-flight/config.toml
            batch_size=256,
            memory_capacity=500_000,
            # dsac-flight starts updates once replay >= batch_size
            learning_starts=256,
            updates_per_step=4,
            lr=4.4e-4,
            policy_lr=4.4e-4,
            gamma=0.99,
            tau=0.005,  # polyak_step_size=0.995 -> tau=0.005
            # IQN params
            num_quantiles=8,
            num_quantiles_exp=8,
            embedding_dim=64,  # n_cos
            hidden_layers=[64, 64],
            huber_threshold=1.0,
            # Entropy / temperature
            automatic_entropy_tuning=True,
            target_entropy_scale=1.0,
            min_alpha=0.0,
            # No extra exploration noise in dsac-flight
            exploration_noise_std=0.0,
            # No reward clipping in dsac-flight
            reward_clip=None,
            # CAPS: env already penalizes u/du/ddu for B747 -> keep off
            caps_lambda_smoothness=0.0,
            caps_lambda_temporal=0.0,
            caps_noise_std=0.05,
            # Risk distortion: start neutral for tracking stability
            risk_distortion="neutral",
            # Misc
            device=device,
            log_dir=log_dir,
            log_every_updates=200,
            seed=seed,
        )

    # --------------------------
    # Training
    # --------------------------
    print(f"Device: {device}")
    print(f"Log dir: {log_dir.resolve()}")
    print(f"Best checkpoints dir: {best_dir.resolve()}")
    print(
        "Training on sine: "
        f"+/-{sine_amplitude_deg} deg @ {sine_frequency_hz} Hz, "
        f"dt={dt}s, T={tn}s"
    )
    # Stage 1: bigger sine to learn control authority
    print(f"Stage 1: sine +/-{train_amp_stage1_deg} deg")
    agent.train_vector(
        total_steps=20_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m1 = eval_one_episode(agent, env_eval)
    print("Eval after Stage 1 (fixed sine +/-1deg):", m1)
    if float(m1["return_sum"]) > float(eval_best):
        eval_best = float(m1["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m1)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    # Stage 2: match eval amplitude
    env_train.step_rand.amplitude_deg_range = (
        -train_amp_stage2_deg,
        train_amp_stage2_deg,
    )
    env_train.step_rand.min_abs_amplitude_deg = 0.5 * train_amp_stage2_deg
    print(f"Stage 2: sine +/-{train_amp_stage2_deg} deg")
    agent.train_vector(
        total_steps=40_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m2 = eval_one_episode(agent, env_eval)
    print("Eval after Stage 2 (fixed sine +/-1deg):", m2)
    if float(m2["return_sum"]) > float(eval_best):
        eval_best = float(m2["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m2)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    # Stage 3: fine-tune without extra exploration noise
    agent.exploration_noise_std = 0.0
    print("Stage 3: exploration_noise_std set to 0.0 (fine-tune)")
    agent.train_vector(
        total_steps=20_000,
        warmup_steps=0,
        log_every=1_000,
        reward_window=200,
        save_best=True,
        save_path=best_dir,
    )
    m_eval = eval_one_episode(agent, env_eval)
    print("Eval on fixed sine:", m_eval)
    if float(m_eval["return_sum"]) > float(eval_best):
        eval_best = float(m_eval["return_sum"])
        saved = save_eval_best(agent, eval_best_dir, metrics=m_eval)
        print("Saved new eval-best checkpoint to:", saved.resolve())

    agent.close()

    print("\nDone.")
    print("Tip: open TensorBoard; watch Performance/RewardMedian200.")


if __name__ == "__main__":
    main()
