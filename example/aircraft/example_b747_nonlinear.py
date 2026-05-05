"""B-747 nonlinear 6-DoF model: trim, step response, and damage demo.

Run:
    poetry run python example/aircraft/example_b747_nonlinear.py

Demonstrates:
1. Newton-Raphson trim at FL200, V=674 ft/s.
2. Open-loop elevator step response from trim (short-period mode).
3. Effect of 50% elevator effectiveness loss at t = 5 s on the same step.

Reference: NASA CR-2144 §IX (Heffley & Jewell 1972) — Aircraft
Handling Qualities Data.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from tensoraerospace.aerospacemodel.b747.nonlinear import (
    B747Configuration,
    NonlinearB747,
    trim,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile,
    SurfaceEffectivenessEvent,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env


def run_step_response(
    *,
    elevator_step_deg: float = -1.0,
    duration_s: float = 10.0,
    dt: float = 0.01,
    damage_profile: DamageProfile | None = None,
):
    """Apply a constant elevator step from trim and log the response."""
    n_steps = int(duration_s / dt)
    env = NonlinearB747Env(
        trim_at=(20_000.0, 674.0),
        number_time_steps=n_steps,
        dt=dt,
        action_space="virtual",
        config=B747Configuration.NOMINAL,
        damage_profile=damage_profile,
    )
    obs, _ = env.reset()

    # Trim throttle from the trim-finder
    r = trim(altitude_ft=20_000.0, V_ft_s=674.0)
    de_cmd = np.deg2rad(elevator_step_deg)
    u = np.array([de_cmd, 0.0, 0.0, r.throttle])

    log_t = np.arange(n_steps) * dt
    log_theta = np.zeros(n_steps)
    log_q = np.zeros(n_steps)
    log_alt = np.zeros(n_steps)
    log_V = np.zeros(n_steps)
    triggers: list[tuple[float, str]] = []

    for k in range(n_steps):
        log_theta[k] = obs[7]
        log_q[k] = obs[4]
        log_alt[k] = -obs[11]
        log_V[k] = float(np.linalg.norm(obs[:3]))
        obs, _, _, trunc, info = env.step(u)
        if "damage_events_triggered" in info:
            for label in info["damage_events_triggered"]:
                triggers.append((k * dt, label))
        if trunc:
            log_t = log_t[: k + 1]
            log_theta = log_theta[: k + 1]
            log_q = log_q[: k + 1]
            log_alt = log_alt[: k + 1]
            log_V = log_V[: k + 1]
            break
    return dict(
        t=log_t,
        theta=np.rad2deg(log_theta),
        q=np.rad2deg(log_q),
        altitude=log_alt,
        V=log_V,
        triggers=triggers,
    )


def main() -> None:
    print("=== B-747 Nonlinear 6-DoF — example ===\n")

    # 1. Trim
    r = trim(altitude_ft=20_000.0, V_ft_s=674.0)
    print(f"Trim @ FL200, V=674 ft/s:")
    print(f"  α = {np.rad2deg(r.alpha_rad):+.3f}°")
    print(f"  δ_e = {np.rad2deg(r.elevator_rad):+.3f}°")
    print(f"  δ_T = {r.throttle:.4f}")
    print(f"  residual = {r.residual:.2e}, converged = {r.converged}\n")

    # 2. Healthy elevator step response
    healthy = run_step_response(elevator_step_deg=-1.0, duration_s=10.0)
    print("Healthy step response (δ_e = −1° for 10 s):")
    print(f"  Δθ_max = {np.max(healthy['theta']) - healthy['theta'][0]:+.2f}°")
    print(f"  V change: {healthy['V'][-1] - healthy['V'][0]:+.2f} ft/s")
    print(f"  Δh = {healthy['altitude'][-1] - healthy['altitude'][0]:+.0f} ft\n")

    # 3. Same step with 50% elevator loss at t=5 s
    profile = DamageProfile(
        events=[
            SurfaceEffectivenessEvent(
                trigger_time=5.0,
                surface="elevator",
                mu=0.5,
                label="elevator_50pct_loss",
            ),
        ]
    )
    damaged = run_step_response(
        elevator_step_deg=-1.0,
        duration_s=10.0,
        damage_profile=profile,
    )
    print("Damaged step response (elevator → 50% effectiveness at t=5s):")
    for t, label in damaged["triggers"]:
        print(f"  t = {t:.2f} s — fired: {label}")
    print(f"  Δθ_max = {np.max(damaged['theta']) - damaged['theta'][0]:+.2f}°")
    print(f"  V change: {damaged['V'][-1] - damaged['V'][0]:+.2f} ft/s")
    print(f"  Δh = {damaged['altitude'][-1] - damaged['altitude'][0]:+.0f} ft")

    # 4. Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(healthy["t"], healthy["theta"], "tab:blue", label="healthy")
    axes[0].plot(
        damaged["t"],
        damaged["theta"],
        "tab:red",
        linestyle="--",
        label="50% elev loss @ 5s",
    )
    axes[0].axvline(5.0, color="k", linestyle=":", linewidth=0.6, alpha=0.5)
    axes[0].set_ylabel(r"pitch $\theta$, deg")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(healthy["t"], healthy["q"], "tab:blue")
    axes[1].plot(damaged["t"], damaged["q"], "tab:red", linestyle="--")
    axes[1].axvline(5.0, color="k", linestyle=":", linewidth=0.6, alpha=0.5)
    axes[1].set_ylabel(r"pitch rate $q$, deg/s")
    axes[1].grid(True)

    axes[2].plot(healthy["t"], healthy["altitude"], "tab:blue")
    axes[2].plot(damaged["t"], damaged["altitude"], "tab:red", linestyle="--")
    axes[2].axvline(5.0, color="k", linestyle=":", linewidth=0.6, alpha=0.5)
    axes[2].set_xlabel("time, s")
    axes[2].set_ylabel("altitude, ft")
    axes[2].grid(True)
    fig.suptitle(
        "B-747 nonlinear: open-loop elevator step (−1°)\n"
        "Healthy vs. 50% elevator effectiveness loss"
    )
    plt.tight_layout()
    out = "/tmp/b747_nonlinear_step_response.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved plot to: {out}")


if __name__ == "__main__":
    main()
