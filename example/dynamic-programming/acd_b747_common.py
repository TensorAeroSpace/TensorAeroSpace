"""Shared utilities for ACD/ADP examples on ImprovedB747Env.

These helpers are intentionally lightweight and notebook-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

MODEL_BASED_DESIGNS = {"hdp", "dhp", "gdhp", "addhp", "adgdhp"}


def make_reference_signal_sine(
    *,
    n_steps: int,
    dt: float,
    amp_deg: float,
    freq_hz: float,
    phase_deg: float = 0.0,
    offset_deg: float = 0.0,
) -> np.ndarray:
    """Build a sinus reference for pitch theta (radians), shape (1, n_steps)."""
    n_steps = int(n_steps)
    dt = float(dt)
    t = np.arange(n_steps, dtype=float) * dt

    amp = float(np.deg2rad(amp_deg))
    phase = float(np.deg2rad(phase_deg))
    offset = float(np.deg2rad(offset_deg))

    ref = offset + amp * np.sin(2.0 * np.pi * float(freq_hz) * t + phase)
    return ref.reshape(1, -1)


def make_env_b747_sine(
    *,
    dt: float = 0.1,
    n_steps: int = 300,
    reward_mode: str = "tracking",
    initial_state: Optional[np.ndarray] = None,
    include_reference_in_obs: bool = False,
    sine_amp_deg: float = 1.0,
    sine_freq_hz: float = 0.05,
    sine_phase_deg: float = 0.0,
    sine_offset_deg: float = 0.0,
):
    """Create ImprovedB747Env with a sine pitch reference."""
    from tensoraerospace.envs.b747 import ImprovedB747Env

    if initial_state is None:
        initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    ref = make_reference_signal_sine(
        n_steps=int(n_steps),
        dt=float(dt),
        amp_deg=float(sine_amp_deg),
        freq_hz=float(sine_freq_hz),
        phase_deg=float(sine_phase_deg),
        offset_deg=float(sine_offset_deg),
    )
    return ImprovedB747Env(
        initial_state=np.array(initial_state, dtype=float).reshape(-1),
        reference_signal=ref,
        number_time_steps=int(n_steps),
        dt=float(dt),
        reward_mode=str(reward_mode),
        include_reference_in_obs=bool(include_reference_in_obs),
    )


def _ref_values(env) -> tuple[float, float]:
    """Return (theta_ref, q_ref) at current step."""
    ref = np.asarray(env.reference_signal, dtype=float)
    idx = int(np.clip(env.current_step, 0, ref.shape[1] - 1))
    idx_prev = int(np.clip(env.current_step - 1, 0, ref.shape[1] - 1))
    theta_ref = float(ref[0, idx])
    theta_ref_prev = float(ref[0, idx_prev])
    q_ref = float((theta_ref - theta_ref_prev) / float(env.dt))
    return theta_ref, q_ref


def build_r_t(env) -> np.ndarray:
    """Build R(t)=[x, theta_ref, q_ref] for model-based ACD designs."""
    theta_ref, q_ref = _ref_values(env)
    x = np.asarray(env.state, dtype=float).reshape(-1)
    return np.concatenate([x, [theta_ref, q_ref]]).astype(np.float32)


@dataclass
class RolloutResult:
    t_s: np.ndarray
    theta_deg: np.ndarray
    theta_ref_deg: np.ndarray
    q_deg_s: np.ndarray
    elev_deg: np.ndarray
    reward: np.ndarray
    terminated: bool
    truncated: bool
    metrics: Dict[str, float]


def rollout(
    env,
    agent=None,
    *,
    deterministic: bool = True,
    baseline: str | None = None,
    pid_kp: float = -24.6295,
    pid_ki: float = -0.2486,
    pid_kd: float = -7.8179,
    pid_mode: str = "deg",
    pd_kp: float = -24.6295,
    pd_kd: float = -7.8179,
):
    """Run one episode and collect time-series + tracking metrics.

    baseline (when agent is None):
      - None or "random": random actions
      - "pid": PID on pitch tracking (uses tensoraerospace.agent.pid.PID)
      - "pd":  simple PD on (theta,q) tracking (normalized)
    """
    obs, _ = env.reset()

    if agent is not None and hasattr(agent, "reset"):
        try:
            agent.reset()
        except Exception:
            pass

    baseline = None if baseline is None else str(baseline).lower().strip()
    if baseline in ("", "none"):
        baseline = None

    pid = None
    if agent is None and baseline == "pid":
        from tensoraerospace.agent.pid import PID

        pid_mode = str(pid_mode).lower().strip()
        if pid_mode not in ("deg", "norm"):
            raise ValueError("pid_mode must be 'deg' or 'norm'")

        pid_env = None if pid_mode == "deg" else env
        pid = PID(
            env=pid_env,
            kp=float(pid_kp),
            ki=float(pid_ki),
            kd=float(pid_kd),
            dt=float(env.dt),
        )
        pid.reset()

    t_s = []
    theta_deg = []
    theta_ref_deg = []
    q_deg_s = []
    elev_deg = []
    rewards = []

    done = False
    steps = 0
    while not done:
        if agent is None:
            if baseline in (None, "random"):
                action = env.action_space.sample()
            else:
                theta_ref, q_ref = _ref_values(env)
                theta = float(env.state[env._idx_theta])
                q = float(env.state[env._idx_q])

                if baseline == "pid":
                    pid_mode_l = str(pid_mode).lower().strip()
                    if pid_mode_l == "deg":
                        sp = float(np.rad2deg(theta_ref))
                        meas = float(np.rad2deg(theta))
                        control_deg = float(pid.select_action(sp, meas))
                        u = float(control_deg) / float(env.max_stabilizer_angle_deg)
                    else:
                        u = float(pid.select_action(theta_ref, theta))
                    action = np.asarray([np.clip(u, -1.0, 1.0)], dtype=np.float32)
                elif baseline == "pd":
                    e_theta_n = (float(theta_ref) - float(theta)) / float(
                        env.max_pitch_rad
                    )
                    e_q_n = (float(q_ref) - float(q)) / float(env.max_pitch_rate_rad_s)
                    u = float(pd_kp) * float(e_theta_n) + float(pd_kd) * float(e_q_n)
                    action = np.asarray([np.clip(u, -1.0, 1.0)], dtype=np.float32)
                else:
                    raise ValueError(
                        f"Unknown baseline={baseline!r}. Use 'pid', 'pd', or None."
                    )
        else:
            design = str(getattr(agent, "design", "adhdp")).lower().strip()
            if design in MODEL_BASED_DESIGNS:
                r_t = build_r_t(env)
                action = agent.select_action(r_t, evaluate=deterministic)
            else:
                action = agent.select_action(
                    np.asarray(obs, dtype=np.float32), evaluate=deterministic
                )

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        ts = float(env.current_step) * float(env.dt)
        t_s.append(ts)

        theta_deg.append(float(np.rad2deg(info.get("theta_rad", 0.0))))
        theta_ref_deg.append(float(np.rad2deg(info.get("target_theta_rad", 0.0))))

        q_rad_s = float(env.state[env._idx_q])
        q_deg_s.append(float(np.rad2deg(q_rad_s)))

        u_norm = float(info.get("u_norm", 0.0))
        elev_deg.append(float(u_norm * float(env.max_stabilizer_angle_deg)))

        rewards.append(float(reward))

        steps += 1
        if steps >= int(getattr(env, "number_time_steps", 0) or 0) + 5:
            break

    theta = np.asarray(theta_deg)
    theta_ref = np.asarray(theta_ref_deg)
    err = theta_ref - theta
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    max_abs = float(np.max(np.abs(err))) if err.size else float("nan")

    metrics = {
        "rmse_theta_deg": rmse,
        "mae_theta_deg": mae,
        "max_abs_theta_deg": max_abs,
        "episode_return": float(np.sum(rewards)),
    }

    return RolloutResult(
        t_s=np.asarray(t_s),
        theta_deg=np.asarray(theta_deg),
        theta_ref_deg=np.asarray(theta_ref_deg),
        q_deg_s=np.asarray(q_deg_s),
        elev_deg=np.asarray(elev_deg),
        reward=np.asarray(rewards),
        terminated=bool(terminated),
        truncated=bool(truncated),
        metrics=metrics,
    )


def plot_rollout(
    data: RolloutResult, *, title: str = "", figsize: tuple[int, int] = (12, 6)
):
    import matplotlib.pyplot as plt

    t = data.t_s
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax[0].plot(t, data.theta_ref_deg, label="theta_ref", linewidth=2)
    ax[0].plot(t, data.theta_deg, label="theta", linewidth=2)
    ax[0].set_ylabel("Pitch θ (deg)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(t, data.elev_deg, label="elevator", linewidth=2)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Elevator (deg)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    if title:
        fig.suptitle(title)
    plt.show()

    print("metrics:", data.metrics)
